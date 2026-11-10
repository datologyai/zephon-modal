import dataclasses
from datetime import datetime, timezone

import pytest

from zephon_modal.adapters import Torchtitan
from zephon_modal.placement import Placement
from zephon_modal.reconcile import Backend, Call, Observation, Plan, Policy, RunState, gpu_hours, new_state, tick
from zephon_modal.runner import RunSpec
from zephon_modal.zephon import Zephon

PLACEMENTS = [Placement("H100:8", nodes=2), Placement("H100:8", fallbacks=("H100:4",))]
CLUSTER, SINGLE = (placement.name for placement in PLACEMENTS)


def _plan(**policy) -> Plan:
    specs = [
        RunSpec(
            run_id="r1",
            steps=100,
            seed=42,
            launch_id="",
            placement=placement,
            adapter=Torchtitan(module="recipes", config="smoke"),
            data=Zephon(canonical_replicas=16),
            runs_mount="/runs",
            runs_volume_name="runs",
            registry_name="job-runs",
        )
        for placement in PLACEMENTS
    ]
    return Plan(run_id="r1", steps=100, specs=specs, policy=Policy(**policy))


class FakeBackend(Backend):
    def __init__(self) -> None:
        self.observation = Observation()
        self.spawned: list[RunSpec] = []
        self.cancelled: list[str] = []
        self.alerts: list[str] = []
        self.saved_current: list[dict | None] = []
        self.pruned: list[int] = []
        self.fail_spawn = False

    def observe(self, plan, state):
        return self.observation

    def save(self, state):
        self.saved_current.append(dict(state.current) if state.current else None)

    def spawn(self, spec):
        if self.fail_spawn:
            raise ConnectionError("spawn RPC failed")
        self.spawned.append(spec)
        return f"fc-{len(self.spawned)}"

    def cancel(self, call_id):
        self.cancelled.append(call_id)

    def prune(self, plan, keep):
        self.pruned.append(keep)
        return [1]

    def alert(self, run_id, message):
        self.alerts.append(message)


class Harness:
    """A run under the reconciler, on a fake clock, one tick at a time."""

    def __init__(self, **policy) -> None:
        self.plan = _plan(**policy)
        self.backend = FakeBackend()
        self.now = 1000.0
        self.state: RunState = new_state(self.plan, self.now)

    def tick(self, seconds: float = 60, **observation) -> RunState:
        self.now += seconds
        if observation:
            self.backend.observation = dataclasses.replace(Observation(), **observation)
        tick(self.plan, self.state, self.backend, self.now, **self._requests)
        self._requests = {}
        return self.state

    _requests: dict = {}

    def request(self, **requests) -> None:
        self._requests = requests

    @property
    def placement(self) -> str:
        assert self.state.current is not None
        return self.state.current["placement"]

    @property
    def events(self) -> list[str]:
        return [event["event"] for event in self.state.events]


def test_launches_on_the_preferred_placement_and_completes():
    run = Harness()
    run.tick()
    assert run.placement == CLUSTER
    assert run.backend.spawned[0].steps == 100 and run.backend.spawned[0].launch_id
    run.tick(call=Call.RUNNING, task_id="ta-1", checkpoint=50)
    assert run.state.checkpoint == 50
    # Already ended and confirmed: finished in the same tick.
    run.tick(call=Call.SUCCEEDED, checkpoint=100)
    assert run.state.status == "completed"
    assert run.backend.cancelled == []
    assert run.events == ["launched", "checkpoint", "running", "checkpoint", "launch finished", "drained"]
    assert run.backend.alerts == ["r1: completed 100 steps"]


def test_state_is_saved_with_the_new_launch_before_it_is_spawned():
    run = Harness()
    run.tick()
    assert run.backend.saved_current[0]["launch_id"] == run.backend.spawned[0].launch_id
    assert run.backend.saved_current[0]["call_id"] is None


def test_a_failed_spawn_leaves_no_current_launch():
    run = Harness()
    run.backend.fail_spawn = True
    with pytest.raises(ConnectionError):
        run.tick()
    assert run.state.current is None
    run.backend.fail_spawn = False
    run.tick()
    assert run.placement == CLUSTER


def test_queue_timeout_cools_the_placement_and_moves_on_after_draining():
    run = Harness(queue_timeout=600)
    run.tick()
    run.tick(call=Call.QUEUED)
    run.tick(seconds=600, call=Call.QUEUED)
    assert run.state.status == "draining"
    assert run.backend.cancelled == ["fc-1"]
    # The cancelled call hasn't ended yet: no successor.
    run.tick(call=Call.QUEUED)
    assert len(run.backend.spawned) == 1
    run.tick(call=Call.CANCELLED)
    assert run.placement == SINGLE
    assert run.state.cooling_until[CLUSTER] > run.now


def test_cooled_down_placement_is_preferred_again_after_cooldown():
    run = Harness(queue_timeout=60, cooldown=600)
    run.tick()
    run.tick(seconds=60, call=Call.QUEUED)
    run.tick(call=Call.CANCELLED)
    assert run.placement == SINGLE
    run.tick(seconds=600, call=Call.FAILED, error="boom")
    assert run.placement == CLUSTER


def test_waits_for_every_container_to_confirm_before_a_successor():
    run = Harness()
    run.tick()
    run.tick(call=Call.RUNNING, task_id="ta-1")
    run.tick(call=Call.FAILED, error="CalledProcessError", drained=False)
    assert run.state.status == "draining"
    run.tick(call=Call.FAILED, drained=False)
    assert len(run.backend.spawned) == 1
    run.tick(call=Call.FAILED, drained=True)
    assert len(run.backend.spawned) == 2


def test_unconfirmed_launch_needs_an_operator():
    run = Harness(drain_timeout=300)
    run.tick()
    run.tick(call=Call.RUNNING, task_id="ta-1")
    run.request(stop=True)
    run.tick(call=Call.RUNNING, task_id="ta-1")
    run.tick(seconds=300, call=Call.CANCELLED, drained=False)
    assert run.state.status == "needs recovery"
    assert "zephon modal recover" in run.backend.alerts[-1]
    run.tick(seconds=3600, call=Call.CANCELLED, drained=False)
    assert run.state.status == "needs recovery"
    run.request(recover=True)
    run.tick(call=Call.CANCELLED, drained=False)
    # The stop request still stands.
    assert run.state.status == "stopped"


def test_recovery_relaunches():
    run = Harness(drain_timeout=300)
    run.tick()
    run.tick(call=Call.FAILED, error="boom", drained=False)
    run.tick(seconds=300, call=Call.FAILED, drained=False)
    assert run.state.status == "needs recovery"
    run.request(recover=True)
    run.tick(call=Call.FAILED, drained=False)
    assert run.state.status == "running"
    assert len(run.backend.spawned) == 2


def test_too_many_restarts_moves_on():
    run = Harness(max_restarts=1)
    run.tick()
    run.tick(call=Call.RUNNING, task_id="ta-1")
    run.tick(call=Call.QUEUED)
    run.tick(call=Call.RUNNING, task_id="ta-2")
    assert run.state.status == "running"
    run.tick(call=Call.QUEUED)
    run.tick(call=Call.RUNNING, task_id="ta-3")
    assert run.state.status == "draining"
    assert run.backend.cancelled == ["fc-1"]
    run.tick(call=Call.CANCELLED)
    assert run.placement == SINGLE
    assert run.events.count("restarted") == 2


def test_function_timeout_relaunches_on_the_same_placement():
    run = Harness()
    run.tick()
    run.tick(call=Call.RUNNING, task_id="ta-1", checkpoint=10)
    run.tick(call=Call.TIMED_OUT, checkpoint=10)
    assert len(run.backend.spawned) == 2 and run.placement == CLUSTER
    assert run.backend.cancelled == [] and run.state.failures == 0


def test_gives_up_after_failures_without_progress():
    run = Harness(max_failures=2)
    run.tick()
    run.tick(call=Call.FAILED, error="boom")
    assert run.state.failures == 1 and len(run.backend.spawned) == 2
    run.tick(call=Call.FAILED, error="boom again")
    assert run.state.status == "gave up"
    assert any("giving up after 2 failures" in alert for alert in run.backend.alerts)


def test_a_new_checkpoint_resets_the_failure_count():
    run = Harness(max_failures=2)
    run.tick()
    run.tick(call=Call.FAILED, error="boom")
    run.tick(call=Call.RUNNING, task_id="ta-1", checkpoint=10)
    assert run.state.failures == 0


def test_stalled_launch_is_cancelled_and_counts_as_a_failure():
    run = Harness(stall_timeout=1800)
    run.tick()
    run.tick(call=Call.RUNNING, task_id="ta-1", checkpoint=10)
    run.tick(seconds=1700, call=Call.RUNNING, task_id="ta-1", checkpoint=10)
    assert run.state.status == "running"
    run.tick(seconds=200, call=Call.RUNNING, task_id="ta-1", checkpoint=10)
    assert run.state.status == "draining" and run.state.failures == 1
    assert run.backend.cancelled == ["fc-1"]


def test_a_restart_gets_a_fresh_stall_budget():
    run = Harness(stall_timeout=1800)
    run.tick()
    run.tick(call=Call.RUNNING, task_id="ta-1")
    run.tick(seconds=1700, call=Call.QUEUED)
    run.tick(seconds=200, call=Call.RUNNING, task_id="ta-2")
    assert run.state.status == "running"


def test_unresponsive_launch_is_cancelled_after_the_heartbeat_timeout():
    run = Harness(heartbeat_timeout=300)
    run.tick()
    run.tick(call=Call.RUNNING, task_id="ta-1")
    run.tick(call=Call.UNKNOWN, error="no heartbeat")
    run.tick(seconds=200, call=Call.UNKNOWN, error="no heartbeat")
    assert run.state.status == "running"
    run.tick(seconds=100, call=Call.UNKNOWN, error="no heartbeat")
    assert run.state.status == "draining" and run.backend.cancelled == ["fc-1"]


def test_stop_cancels_and_waits():
    run = Harness()
    run.tick()
    run.tick(call=Call.RUNNING, task_id="ta-1")
    run.request(stop=True)
    run.tick(call=Call.RUNNING, task_id="ta-1")
    assert run.state.status == "draining" and run.backend.cancelled == ["fc-1"]
    run.tick(call=Call.CANCELLED)
    assert run.state.status == "stopped"
    run.tick()
    assert len(run.backend.spawned) == 1


def test_stop_while_draining_to_relaunch_stops_instead():
    run = Harness()
    run.tick()
    run.tick(call=Call.FAILED, error="boom", drained=False)
    run.request(stop=True)
    run.tick(call=Call.FAILED, drained=False)
    run.tick(call=Call.FAILED, drained=True)
    assert run.state.status == "stopped"
    assert len(run.backend.spawned) == 1


def test_gpu_budget_gives_up():
    run = Harness(max_gpu_hours=10)
    run.tick()
    run.tick(call=Call.RUNNING, task_id="ta-1", gpu_hours=10.5)
    run.tick(call=Call.CANCELLED, gpu_hours=10.5)
    assert run.state.status == "gave up"


def test_prunes_when_a_checkpoint_lands():
    run = Harness(keep_checkpoints=3)
    run.tick()
    run.tick(call=Call.RUNNING, task_id="ta-1", checkpoint=10)
    run.tick(call=Call.RUNNING, task_id="ta-1", checkpoint=10)
    assert run.backend.pruned == [3]


def test_alerts_when_quiet_and_repeats_at_the_interval():
    run = Harness(quiet_alert=3600, queue_timeout=10**6)
    run.tick()
    run.tick(seconds=3600, call=Call.QUEUED)
    run.tick(call=Call.QUEUED)
    run.tick(seconds=3600, call=Call.QUEUED)
    quiet = [alert for alert in run.backend.alerts if "no new checkpoint" in alert]
    assert len(quiet) == 2


def test_finished_runs_are_left_alone():
    run = Harness()
    run.state.status = "completed"
    run.tick()
    assert run.backend.spawned == []


def test_gpu_hours_counts_every_container():
    records = [
        {"started": "2026-11-09T00:00:00+00:00", "ended": "2026-11-09T01:00:00+00:00", "allocated": "2xh100x8"},
        {"started": "2026-11-09T00:00:00+00:00", "heartbeat": "2026-11-09T00:30:00+00:00", "allocated": "h100x4"},
    ]
    assert gpu_hours(records, now=0) == pytest.approx(8 + 2)


def test_gpu_hours_stop_counting_containers_that_died_silently():
    now = datetime(2026, 11, 12, tzinfo=timezone.utc).timestamp()  # days later
    records = [
        # A container of an old launch that died without "ended": up to its last heartbeat.
        {
            "launch_id": "L1",
            "started": "2026-11-09T00:00:00+00:00",
            "heartbeat": "2026-11-09T01:00:00+00:00",
            "allocated": "t4x1",
        },
        # Never even heartbeated: nothing.
        {"launch_id": "L1", "started": "2026-11-09T02:00:00+00:00", "allocated": "t4x1"},
        # The current launch's running container: up to now.
        {"launch_id": "L2", "started": "2026-11-11T23:00:00+00:00", "allocated": "t4x1"},
    ]
    assert gpu_hours(records, now, current_launch="L2") == pytest.approx(1 + 0 + 1)


def test_state_serializes_for_the_volume_mirror():
    run = Harness()
    run.tick()
    record = run.state.to_json()
    assert record["status"] == "running" and record["current"]["placement"] == CLUSTER
    assert record["started"].endswith("+00:00")


def test_a_failure_moves_the_retry_to_the_next_placement():
    run = Harness()
    run.tick()
    run.tick(call=Call.FAILED, error="boom")
    assert run.placement == SINGLE
    assert run.state.cooling_until[CLUSTER] > run.now


def test_a_timeout_without_progress_since_the_last_one_is_a_failure():
    run = Harness()
    run.tick()
    run.tick(call=Call.RUNNING, task_id="ta-1", checkpoint=10)
    run.tick(call=Call.TIMED_OUT, checkpoint=10)
    assert run.state.failures == 0 and len(run.backend.spawned) == 2
    run.tick(call=Call.RUNNING, task_id="ta-2", checkpoint=10)
    run.tick(call=Call.TIMED_OUT, checkpoint=10)
    assert run.state.failures == 1
    assert "timed out without progress" in run.events


def test_timeouts_with_progress_in_between_are_boundaries():
    run = Harness()
    run.tick()
    for checkpoint in (10, 20, 30):
        run.tick(call=Call.RUNNING, task_id=f"ta-{checkpoint}", checkpoint=checkpoint)
        run.tick(call=Call.TIMED_OUT, checkpoint=checkpoint)
    assert run.state.failures == 0 and run.state.status == "running"
    assert len(run.backend.spawned) == 4


def test_finishing_below_the_step_target_says_so():
    run = Harness()
    run.tick()
    run.tick(call=Call.SUCCEEDED, checkpoint=60)
    run.tick(call=Call.SUCCEEDED, checkpoint=60)
    assert run.state.status == "completed"
    assert "latest checkpoint is 60 of 100" in run.backend.alerts[-1]


def test_an_ended_launch_still_waits_for_its_containers():
    run = Harness()
    run.tick()
    run.tick(call=Call.FAILED, error="boom", drained=False)
    assert run.state.status == "draining" and len(run.backend.spawned) == 1
    run.tick(call=Call.FAILED, drained=True)
    assert run.state.status == "running" and len(run.backend.spawned) == 2


def test_a_restart_between_ticks_still_counts():
    run = Harness(max_restarts=1)
    run.tick()
    run.tick(call=Call.RUNNING, task_id="ta-1", attempts=1)
    # Attempts 2 came and went between ticks; the next tick sees attempt 3 running.
    run.tick(call=Call.RUNNING, task_id="ta-3", attempts=3)
    assert "restarted" in run.events
    assert run.state.status == "draining" and run.backend.cancelled == ["fc-1"]
