import json
from collections.abc import Callable

import pytest

from zephon_modal.adapters import Torchtitan
from zephon_modal.data import DataContract
from zephon_modal.placement import Placement
from zephon_modal.runner import RunSpec, launch_attempts
from zephon_modal.supervisor import (
    Backend,
    CallState,
    CallStatus,
    GaveUp,
    Plan,
    Policy,
    Supervisor,
    state_from_attempts,
)

QUEUED = CallState(CallStatus.QUEUED)
DONE = CallState(CallStatus.SUCCEEDED, result=["step-20"])
FAILED = CallState(CallStatus.FAILED, error="CalledProcessError: exit 1")
TIMED_OUT = CallState(CallStatus.TIMED_OUT)


def running(task: str) -> CallState:
    return CallState(CallStatus.RUNNING, task_id=task)


class InputCancellation(BaseException):
    """Stands in for modal.exception.InputCancellation (matched by name)."""


class FakeBackend(Backend):
    """Each spawned call plays a script of states, one per poll; the clock moves only in sleep()."""

    def __init__(self, scripts: dict[str, list[CallState]], on_poll: Callable[[int], None] | None = None) -> None:
        self.scripts = scripts
        self.clock = 0.0
        self.spawned: list[str] = []
        self.cancelled: list[str] = []
        self.saved: list[dict] = []
        self.polls = 0
        self.on_poll = on_poll

    def spawn(self, spec: RunSpec) -> str:
        call_id = f"{spec.placement.name}#{sum(s.startswith(spec.placement.name) for s in self.spawned)}"
        self.spawned.append(call_id)
        return call_id

    def state(self, call_id: str) -> CallState:
        script = self.scripts[call_id]
        return script.pop(0) if len(script) > 1 else script[0]

    def cancel(self, call_id: str) -> None:
        self.cancelled.append(call_id)

    def save(self, record) -> None:
        self.saved.append(dict(record))

    def now(self) -> float:
        return self.clock

    def sleep(self, seconds: float) -> None:
        self.polls += 1
        if self.on_poll:
            self.on_poll(self.polls)
        self.clock += seconds


POLICY = Policy(queue_timeout=60, max_restarts=1, max_failures=1, cooldown=600, poll_interval=10)


def _supervisor(backend: FakeBackend, policy: Policy = POLICY) -> Supervisor:
    specs = [
        RunSpec(
            run_id="r",
            steps=20,
            seed=42,
            launch_id="",
            placement=placement,
            adapter=Torchtitan(module="m", config="c"),
            data=DataContract(),
            runs_mount="/runs",
            runs_volume_name="runs",
        )
        for placement in (Placement("H100:8", nodes=2), Placement("H100:8"), Placement("H100:4"))
    ]
    return Supervisor(Plan(run_id="r", steps=20, specs=specs, policy=policy), backend)


def _events(supervisor: Supervisor) -> list[tuple[str, str | None]]:
    return [(event["event"], event.get("placement")) for event in supervisor.events]


def test_runs_on_the_preferred_placement():
    backend = FakeBackend({"2xh100x8#0": [QUEUED, running("ta-1"), running("ta-1"), DONE]})
    supervisor = _supervisor(backend)
    assert supervisor.run() == ["step-20"]
    assert _events(supervisor) == [("launched", "2xh100x8"), ("running", "2xh100x8"), ("completed", "2xh100x8")]
    assert backend.saved[-1]["status"] == "completed"


def test_queue_timeout_moves_to_the_next_placement():
    backend = FakeBackend({"2xh100x8#0": [QUEUED], "h100x8#0": [running("ta-1"), DONE]})
    supervisor = _supervisor(backend)
    supervisor.run()
    assert backend.cancelled == ["2xh100x8#0"]
    assert ("queue timeout", "2xh100x8") in _events(supervisor)
    assert backend.spawned == ["2xh100x8#0", "h100x8#0"]


def test_restarts_are_left_to_modal_until_there_are_too_many():
    backend = FakeBackend(
        {
            # One restart is allowed; the second moves the run on.
            "2xh100x8#0": [running("ta-1"), QUEUED, running("ta-2"), running("ta-3")],
            "h100x8#0": [running("ta-4"), DONE],
        }
    )
    supervisor = _supervisor(backend)
    supervisor.run()
    events = _events(supervisor)
    assert events.count(("restarted", "2xh100x8")) == 2
    assert ("too many restarts", "2xh100x8") in events
    assert backend.cancelled == ["2xh100x8#0"]
    assert backend.spawned[-1] == "h100x8#0"


def test_a_restart_stuck_in_the_queue_moves_on():
    backend = FakeBackend({"2xh100x8#0": [running("ta-1"), QUEUED], "h100x8#0": [running("ta-2"), DONE]})
    supervisor = _supervisor(backend)
    supervisor.run()
    assert ("queue timeout", "2xh100x8") in _events(supervisor)


def test_function_timeout_relaunches_on_the_preferred_placement():
    backend = FakeBackend({"2xh100x8#0": [running("ta-1"), TIMED_OUT], "2xh100x8#1": [running("ta-2"), DONE]})
    supervisor = _supervisor(backend)
    supervisor.run()
    assert backend.spawned == ["2xh100x8#0", "2xh100x8#1"]
    assert backend.cancelled == []


def test_gives_up_after_max_failures():
    backend = FakeBackend({"2xh100x8#0": [running("ta-1"), FAILED]})
    supervisor = _supervisor(backend)
    with pytest.raises(GaveUp, match="exit 1"):
        supervisor.run()
    assert backend.saved[-1]["status"] == "failed"
    assert backend.spawned == ["2xh100x8#0"]


def test_failures_reset_once_a_launch_runs():
    policy = Policy(queue_timeout=60, max_restarts=1, max_failures=2, cooldown=600, poll_interval=10)
    backend = FakeBackend(
        {
            "2xh100x8#0": [FAILED],
            "2xh100x8#1": [running("ta-1"), FAILED],
            "2xh100x8#2": [running("ta-2"), DONE],
        }
    )
    assert _supervisor(backend, policy).run() == ["step-20"]


def test_cooled_down_placements_come_back():
    backend = FakeBackend(
        {
            "2xh100x8#0": [QUEUED],  # times out at t=70 and cools down until t=670
            "h100x8#0": [running("ta-1")] * 60 + [TIMED_OUT],  # times out at t~680
            "2xh100x8#1": [running("ta-2"), DONE],
        }
    )
    supervisor = _supervisor(backend)
    supervisor.run()
    assert backend.spawned == ["2xh100x8#0", "h100x8#0", "2xh100x8#1"]


def test_when_everything_is_cooling_down_picks_the_first_free():
    backend = FakeBackend({})
    supervisor = _supervisor(backend)
    supervisor.cooling_until = {"2xh100x8": 500, "h100x8": 100, "h100x4": 300}
    assert supervisor.choose().placement.name == "h100x8"


def test_cancel_cancels_the_current_launch():
    def cancel_on_third_poll(poll: int) -> None:
        if poll == 3:
            raise InputCancellation()

    backend = FakeBackend({"2xh100x8#0": [running("ta-1")]}, on_poll=cancel_on_third_poll)
    supervisor = _supervisor(backend)
    with pytest.raises(InputCancellation):
        supervisor.run()
    assert backend.cancelled == ["2xh100x8#0"]
    assert backend.saved[-1]["status"] == "cancelled"


def test_launches_get_fresh_launch_ids_and_the_target(monkeypatch):
    ids = iter(["20261106T000000", "20261106T000100"])
    monkeypatch.setattr("zephon_modal.supervisor.new_launch_id", lambda: next(ids))
    specs = []
    backend = FakeBackend({"2xh100x8#0": [QUEUED], "h100x8#0": [DONE]})
    original_spawn = backend.spawn
    backend.spawn = lambda spec: specs.append(spec) or original_spawn(spec)
    _supervisor(backend).run()
    assert [(s.launch_id, s.steps) for s in specs] == [("20261106T000000", 20), ("20261106T000100", 20)]


def _attempt(task: str, started: str, outcome: str, node_rank: int = 0) -> dict:
    return {"task_id": task, "node_rank": node_rank, "started": started, "outcome": outcome}


def test_state_from_attempts():
    assert state_from_attempts([]).status is CallStatus.QUEUED
    first = _attempt("ta-1", "2026-11-06T19:00:00+00:00", "running")
    assert state_from_attempts([first]) == running("ta-1")
    # Stopped, and Modal's restart hasn't started yet.
    stopped = first | {"outcome": "stopped"}
    assert state_from_attempts([stopped]).status is CallStatus.QUEUED
    assert state_from_attempts([stopped, _attempt("ta-2", "2026-11-06T19:05:00+00:00", "running")]) == running("ta-2")


def test_launch_attempts_reads_node_zero_records_in_order(tmp_path):
    launch_dir = tmp_path / "launches" / "20261106T190000"
    launch_dir.mkdir(parents=True)
    (launch_dir / "launch.json").write_text("{}")
    for record in (
        _attempt("ta-2", "2026-11-06T19:05:00+00:00", "running"),
        _attempt("ta-1", "2026-11-06T19:00:00+00:00", "stopped"),
        _attempt("ta-3", "2026-11-06T19:05:01+00:00", "running", node_rank=1),
    ):
        (launch_dir / f"{record['task_id']}.json").write_text(json.dumps(record))
    assert [a["task_id"] for a in launch_attempts(str(tmp_path), "20261106T190000")] == ["ta-1", "ta-2"]
    assert launch_attempts(str(tmp_path), "20261106T000000") == []
