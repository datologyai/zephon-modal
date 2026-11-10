# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Keeps every started run going until it reaches its step target.

Nothing here runs for long. A deployed job app has a ``reconcile`` function
that Modal calls every ``reconcile_minutes`` (5 by default, and right after
``start``, ``stop`` and ``recover``); it looks at each
active run once and takes at most one decision, then exits. Modal runs one
call at a time (``max_containers=1``), so there is a single decision-maker
without any locking, and there's no long-lived process to outlive Modal's
24-hour function limit or to die holding a run.

For each run, :func:`tick` compares the plan with what it observes:

- No current launch: launch on the most preferred placement that isn't
  cooling down.
- Queued longer than ``queue_timeout``, or restarted by Modal (preempted)
  more than ``max_restarts`` times: cancel, cool the placement down, move on.
- Running with no new checkpoint for ``stall_timeout``: cancel; it counts
  as a failure.
- Hit the function timeout: launch again; the run resumes from its checkpoint.
- Failed: cool the placement down and launch again (on the next placement
  if there is one), until ``max_failures`` failures in a row without a new
  checkpoint, then give up.
- Finished: done.

Before any successor, the old launch must *drain*: its call has ended and
every container it started has recorded its final commit. Until then no
second launch can write the run directory. If that doesn't happen within
``drain_timeout``, the run waits for an operator (``zephon modal recover``)
instead of guessing.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from zephon_modal.runner import RunSpec, new_launch_id

STATE_FILE = "reconciler.json"
MAX_EVENTS = 500
# The reconcile function's settings, set on its image by ElasticJob.
CONFIG_ENV = "ZEPHON_MODAL_RECONCILER"
ALERT_WEBHOOK_ENV = "ZEPHON_MODAL_ALERT_WEBHOOK"


@dataclass(frozen=True)
class Policy:
    """When the reconciler moves, retries or gives up on a run.

    Attributes:
        queue_timeout: Seconds a launch may wait for containers (at first, or
            after a restart) before moving to the next placement.
        max_restarts: Modal restarts (preemptions) of one launch before moving on.
        cooldown: Seconds a placement is skipped after the reconciler moves off it.
        stall_timeout: Seconds a running launch may go without a new complete
            checkpoint before it's treated as hung and cancelled.
        max_failures: Failed launches in a row, without a new checkpoint in
            between, before giving up.
        drain_timeout: Seconds to wait for a finished or cancelled launch's
            containers to record their final commit.
        heartbeat_timeout: Seconds without a heartbeat from a running launch
            before it's treated as gone.
        keep_checkpoints: Complete checkpoints to keep; older ones are deleted.
            ``None`` keeps everything (as trainers must on multi-node launches).
        max_gpu_hours: Give up once the run's attempts add up to this many GPU hours.
        quiet_alert: Seconds without a new checkpoint before alerting (repeated
            at that interval while it lasts).
    """

    queue_timeout: float = 15 * 60
    max_restarts: int = 2
    cooldown: float = 30 * 60
    stall_timeout: float = 60 * 60
    max_failures: int = 3
    drain_timeout: float = 10 * 60
    heartbeat_timeout: float = 5 * 60
    keep_checkpoints: int | None = None
    max_gpu_hours: float | None = None
    quiet_alert: float = 2 * 60 * 60

    def __post_init__(self) -> None:
        durations = (
            self.queue_timeout,
            self.cooldown,
            self.stall_timeout,
            self.drain_timeout,
            self.heartbeat_timeout,
            self.quiet_alert,
        )
        if min(durations) <= 0:
            raise ValueError("Policy durations must be positive")
        if self.max_restarts < 0 or self.max_failures < 1:
            raise ValueError("max_restarts must be nonnegative and max_failures positive")
        if self.keep_checkpoints is not None and self.keep_checkpoints < 1:
            raise ValueError("keep_checkpoints must be positive")


@dataclass(frozen=True)
class Plan:
    """A run's target: a spec per placement, in order of preference."""

    run_id: str
    steps: int
    specs: Sequence[RunSpec]
    policy: Policy = Policy()


class Call(enum.Enum):
    """Where the current launch's Modal call stands."""

    QUEUED = "queued"  # no container running: not started yet, or a restart pending
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed out"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"  # couldn't tell: an API error, or a silent container


ENDED = (Call.SUCCEEDED, Call.FAILED, Call.TIMED_OUT, Call.CANCELLED)


@dataclass(frozen=True)
class Observation:
    """What the reconciler sees of a run in one tick.

    Attributes:
        call: The current launch's call, if there is one.
        error: Why the call failed or couldn't be observed.
        drained: Every container the current launch started has recorded its final commit.
        checkpoint: The latest complete checkpoint's step.
        gpu_hours: GPU hours across every attempt of the run so far.
        attempts: Attempts the current launch has started (node 0's records),
            so a restart that came and went between ticks still counts.
    """

    call: Call | None = None
    error: str | None = None
    drained: bool = True
    checkpoint: int | None = None
    gpu_hours: float = 0.0
    attempts: int = 0


@dataclass
class RunState:
    """Everything the reconciler remembers about a run between ticks.

    ``status`` is one of: ``running``, ``draining``, ``needs recovery``,
    ``completed``, ``gave up``, ``stopped``.
    """

    run_id: str
    steps: int
    status: str = "running"
    started: float = 0.0
    current: dict[str, Any] | None = None
    draining: dict[str, Any] | None = None
    cooling_until: dict[str, float] = field(default_factory=dict)
    failures: int = 0
    checkpoint: int | None = None
    progress_at: float = 0.0
    quiet_alerted_at: float = 0.0
    stop_requested: bool = False
    # Checkpoint when a launch last hit the function timeout; -1 for never.
    timeout_checkpoint: int = -1
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def active(self) -> bool:
        return self.status in ("running", "draining", "needs recovery")

    def to_json(self) -> dict[str, Any]:
        record = dataclasses.asdict(self)
        record["cooling_until"] = {name: _iso(until) for name, until in self.cooling_until.items()}
        for key in ("started", "progress_at", "quiet_alerted_at"):
            record[key] = _iso(record[key]) if record[key] else None
        return record


class Backend:
    """How the reconciler acts on Modal; tests use a fake."""

    def observe(self, plan: Plan, state: RunState) -> Observation:
        raise NotImplementedError

    def save(self, state: RunState) -> None:
        """Persist the state now, before a spawn: the new launch checks it's current."""
        raise NotImplementedError

    def spawn(self, spec: RunSpec) -> str:
        """Start a launch; returns its call ID."""
        raise NotImplementedError

    def cancel(self, call_id: str) -> None:
        raise NotImplementedError

    def prune(self, plan: Plan, keep: int) -> list[int]:
        """Delete old checkpoints; returns the steps deleted."""
        raise NotImplementedError

    def alert(self, run_id: str, message: str) -> None:
        raise NotImplementedError

    def mirror(self, plan: Plan, state: RunState) -> None:
        """Keep a durable copy of the state next to the run."""


def _iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat(timespec="seconds")


def _event(state: RunState, now: float, event: str, **detail: Any) -> None:
    state.events.append({"time": _iso(now), "event": event, **detail})
    del state.events[:-MAX_EVENTS]
    print(f"[zephon_modal reconcile] {state.run_id}: {event} {detail}", flush=True)


def new_state(plan: Plan, now: float) -> RunState:
    """A just-started run's state."""
    return RunState(run_id=plan.run_id, steps=plan.steps, started=now, progress_at=now)


def choose(plan: Plan, state: RunState, now: float) -> RunSpec:
    """The most preferred placement not cooling down; if all are, the one free soonest."""
    for spec in plan.specs:
        if state.cooling_until.get(spec.placement.name, 0) <= now:
            return spec
    return min(plan.specs, key=lambda spec: state.cooling_until[spec.placement.name])


def _launch(plan: Plan, state: RunState, backend: Backend, now: float) -> None:
    spec = dataclasses.replace(choose(plan, state, now), steps=plan.steps, launch_id=new_launch_id())
    # Current before the spawn: the launch checks it's current before touching the run.
    state.current = {
        "placement": spec.placement.name,
        "launch_id": spec.launch_id,
        "call_id": None,
        "spawned": now,
        "queued_since": now,
        "running_since": None,
    }
    backend.save(state)
    try:
        state.current["call_id"] = backend.spawn(spec)
    except Exception:
        # If Modal created the call anyway, its launch finds it isn't current and stops.
        state.current = None
        raise
    _event(state, now, "launched", placement=spec.placement.name, launch_id=spec.launch_id)


def _drain(
    state: RunState, backend: Backend, now: float, reason: str, then: str, *, cancel: bool, **detail: Any
) -> None:
    """Stop using the current launch: optionally cancel it, then wait for its containers before ``then``."""
    current = state.current or {}
    if cancel and current.get("call_id"):
        backend.cancel(current["call_id"])
    state.status = "draining"
    state.draining = {"reason": reason, "then": then, "since": now}
    _event(state, now, reason, placement=current.get("placement"), then=then, **detail)


def _finish_draining(state: RunState, backend: Backend, now: float) -> bool:
    """The old launch is gone: do what draining was for. Returns whether to launch next."""
    assert state.draining is not None
    then = state.draining["then"]
    if then == "launch" and state.stop_requested:
        then = "stop"
    _event(state, now, "drained", placement=(state.current or {}).get("placement"))
    state.current, state.draining = None, None
    state.status = {"launch": "running", "stop": "stopped", "give up": "gave up", "complete": "completed"}[then]
    if then == "give up":
        backend.alert(state.run_id, f"{state.run_id}: gave up at checkpoint {state.checkpoint}")
    if then == "complete":
        if (state.checkpoint or 0) >= state.steps:
            backend.alert(state.run_id, f"{state.run_id}: completed {state.steps} steps")
        else:
            backend.alert(
                state.run_id,
                f"{state.run_id}: the trainer finished, but the latest checkpoint is {state.checkpoint} "
                f"of {state.steps} steps; check the recipe's step target and save interval",
            )
    return then == "launch"


def tick(
    plan: Plan, state: RunState, backend: Backend, now: float, *, stop: bool = False, recover: bool = False
) -> None:
    """Take this tick's decision for one run, updating ``state`` in place."""
    policy = plan.policy
    if not state.active:
        return
    if stop and not state.stop_requested:
        state.stop_requested = True
        _event(state, now, "stop requested")
    observation = backend.observe(plan, state)

    if observation.checkpoint is not None and (state.checkpoint is None or observation.checkpoint > state.checkpoint):
        state.checkpoint = observation.checkpoint
        state.progress_at = now
        state.failures = 0
        _event(state, now, "checkpoint", step=observation.checkpoint)
        if policy.keep_checkpoints is not None:
            deleted = backend.prune(plan, policy.keep_checkpoints)
            if deleted:
                _event(state, now, "pruned checkpoints", steps=deleted)

    if state.status == "needs recovery":
        if not recover:
            return
        _event(state, now, "recovered by operator")
        state.status, state.current, state.draining = "running", None, None

    if state.status == "draining":
        assert state.draining is not None
        if observation.call in (Call.QUEUED, Call.RUNNING, Call.UNKNOWN) or not observation.drained:
            if now - state.draining["since"] >= policy.drain_timeout:
                state.status = "needs recovery"
                _event(state, now, "needs recovery", call=observation.call and observation.call.value)
                backend.alert(
                    state.run_id,
                    f"{state.run_id}: launch {(state.current or {}).get('launch_id')} didn't confirm it stopped "
                    f"within {policy.drain_timeout:.0f}s. Check it's gone, then run `zephon modal recover`.",
                )
            return
        if not _finish_draining(state, backend, now):
            return

    if state.stop_requested:
        if state.current is None:
            state.status = "stopped"
            _event(state, now, "stopped")
        else:
            _drain(state, backend, now, "stopping", "stop", cancel=True)
        return

    if policy.max_gpu_hours is not None and observation.gpu_hours >= policy.max_gpu_hours:
        if state.current is None:
            state.status = "gave up"
            _event(state, now, "gave up", reason="GPU budget")
        else:
            _drain(state, backend, now, "GPU budget spent", "give up", cancel=True)
        return

    if now - state.progress_at >= policy.quiet_alert and now - state.quiet_alerted_at >= policy.quiet_alert:
        state.quiet_alerted_at = now
        backend.alert(state.run_id, f"{state.run_id}: no new checkpoint for {(now - state.progress_at) / 3600:.1f}h")

    if state.current is None:
        _launch(plan, state, backend, now)
        return

    current = state.current
    placement = current["placement"]
    call = observation.call

    def cool_down() -> None:
        state.cooling_until[placement] = now + policy.cooldown

    def failed(reason: str, error: str | None) -> None:
        state.failures += 1
        # The next launch tries another placement: a bad node or shape shouldn't take every retry.
        cool_down()
        then = "give up" if state.failures >= policy.max_failures else "launch"
        _drain(state, backend, now, reason, then, cancel=True, error=error, failures=state.failures)
        if then == "give up":
            backend.alert(
                state.run_id, f"{state.run_id}: {reason} ({error}); giving up after {state.failures} failures"
            )

    if call is Call.SUCCEEDED:
        _drain(state, backend, now, "launch finished", "complete", cancel=False)
    elif call is Call.TIMED_OUT:
        progress = state.checkpoint or 0
        if progress == state.timeout_checkpoint:
            failed("timed out without progress", f"still at checkpoint {state.checkpoint} since the last timeout")
        else:
            state.timeout_checkpoint = progress
            _drain(state, backend, now, "function timeout", "launch", cancel=False)
    elif call in (Call.FAILED, Call.CANCELLED):
        failed(f"launch {call.value}", observation.error)
    elif call is Call.UNKNOWN:
        unknown_since = current.setdefault("unknown_since", now)
        if now - unknown_since >= policy.heartbeat_timeout:
            failed("launch unresponsive", observation.error)
    elif call is Call.QUEUED:
        current.pop("unknown_since", None)
        current["running_since"] = None
        current["queued_since"] = current.get("queued_since") or now
        if now - current["queued_since"] >= policy.queue_timeout:
            cool_down()
            _drain(state, backend, now, "queue timeout", "launch", cancel=True)
    elif call is Call.RUNNING:
        current.pop("unknown_since", None)
        current["queued_since"] = None
        # Ticks are minutes apart, so attempts come from the containers' records:
        # a restart can start and end between ticks.
        attempts = observation.attempts
        if attempts > current.get("attempts", 0):
            current["attempts"] = attempts
            current["running_since"] = now
            restarts = attempts - 1
            _event(state, now, "running" if not restarts else "restarted", placement=placement, restarts=restarts)
            if restarts > policy.max_restarts:
                cool_down()
                _drain(state, backend, now, "too many restarts", "launch", cancel=True)
                return
        current["running_since"] = current.get("running_since") or now
        if now - max(current["running_since"], state.progress_at) >= policy.stall_timeout:
            failed("stalled", f"no new checkpoint in {policy.stall_timeout:.0f}s")

    # A launch that has already ended and confirmed it's gone needn't wait a tick to be replaced.
    if state.status == "draining" and observation.call in ENDED and observation.drained:
        if _finish_draining(state, backend, now):
            _launch(plan, state, backend, now)


# --- Modal ---------------------------------------------------------------------------------


def _gpus(placement_name: str) -> int:
    """GPUs per node from a placement name such as ``2xh100x8`` or ``t4x1``."""
    return int(placement_name.rsplit("x", 1)[1])


def gpu_hours(records: Sequence[Mapping[str, Any]], now: float, current_launch: str | None = None) -> float:
    """GPU hours used by a run's attempt records (one per container).

    A container still running counts up to ``now``. That's only the current
    launch's: any other record without ``ended`` belongs to a container that
    died without saying so, and counts up to its last heartbeat.
    """
    total = 0.0
    for record in records:
        start = datetime.fromisoformat(record["started"]).timestamp()
        if record.get("ended"):
            end = datetime.fromisoformat(record["ended"]).timestamp()
        elif current_launch is not None and record.get("launch_id") == current_launch:
            end = now
        else:
            end = datetime.fromisoformat(record.get("heartbeat") or record["started"]).timestamp()
        total += max(0.0, end - start) * _gpus(record["allocated"]) / 3600
    return total


def observe_call(call: Any) -> tuple[Call | None, str | None]:
    """A Modal call's state if it has ended (or can't be read), else ``None``.

    Every exception Modal builds from a call's *result* means the call has
    ended: its timeout, an internal failure, a remote error, or the remote
    code's own exception (``CalledProcessError`` from the trainer, or the
    cancellation itself). Errors from *asking* (connection, service,
    client-side timeouts) say nothing about the call, so they're unknown,
    which never starts a successor.
    """
    import modal
    from modal import exception

    try:
        call.get(timeout=0)
    except TimeoutError:
        return None, None  # still going (the builtin TimeoutError; Modal's own means something else)
    except exception.FunctionTimeoutError as error:
        return Call.TIMED_OUT, str(error)
    except exception.InputCancellation as error:
        return Call.CANCELLED, str(error) or "cancelled"
    except (exception.InternalFailure, exception.RemoteError, exception.ExecutionError) as error:
        return Call.FAILED, f"{type(error).__name__}: {error}"
    except (modal.exception.Error, OSError) as error:
        return Call.UNKNOWN, f"{type(error).__name__}: {error}"
    except Exception as error:
        return Call.FAILED, f"{type(error).__name__}: {error}"
    return Call.SUCCEEDED, None


class ModalBackend(Backend):
    """Launches the deployed job's ``train-<placement>`` functions and reads the runs Volume."""

    def __init__(self, app_name: str, runs_volume_name: str, registry: Any) -> None:
        import modal

        self.app_name = app_name
        self.registry = registry
        self.volume = modal.Volume.from_name(runs_volume_name)
        self.webhook = os.environ.get(ALERT_WEBHOOK_ENV)

    def save(self, state: RunState) -> None:
        self.registry.save(state.run_id, state)

    def observe(self, plan: Plan, state: RunState) -> Observation:
        import modal

        spec = plan.specs[0]
        layout = spec.adapter.checkpoints
        checkpoint = layout.latest_complete(spec.run_dir) if layout is not None else None
        current = state.current
        # Each container's own, atomically updated record; "ended" is published
        # only after its last commit returned.
        run_records = self.registry.run_records(plan.run_id)
        hours = gpu_hours(run_records, time.time(), current["launch_id"] if current else None)
        if current is None:
            return Observation(checkpoint=checkpoint, gpu_hours=hours)
        records = [record for record in run_records if record["launch_id"] == current["launch_id"]]
        drained = all("ended" in record for record in records)
        call, error = observe_call(modal.FunctionCall.from_id(current["call_id"]))
        if call is not None:
            return Observation(call=call, error=error, drained=drained, checkpoint=checkpoint, gpu_hours=hours)
        attempts = [record for record in records if record.get("node_rank") == 0]
        if not attempts or "ended" in attempts[-1]:
            return Observation(
                call=Call.QUEUED, drained=drained, checkpoint=checkpoint, gpu_hours=hours, attempts=len(attempts)
            )
        latest = attempts[-1]
        beat = datetime.fromisoformat(latest.get("heartbeat") or latest["started"]).timestamp()
        if time.time() - beat > plan.policy.heartbeat_timeout:
            return Observation(
                call=Call.UNKNOWN, error="no heartbeat", drained=drained, checkpoint=checkpoint, gpu_hours=hours
            )
        return Observation(
            call=Call.RUNNING, drained=drained, checkpoint=checkpoint, gpu_hours=hours, attempts=len(attempts)
        )

    def spawn(self, spec: RunSpec) -> str:
        import modal

        return modal.Function.from_name(self.app_name, f"train-{spec.placement.name}").spawn(spec).object_id

    def cancel(self, call_id: str) -> None:
        import modal

        # Not terminate_containers: the launch gets InputCancellation, stops its trainer and commits.
        modal.FunctionCall.from_id(call_id).cancel()

    def prune(self, plan: Plan, keep: int) -> list[int]:
        from zephon_modal.checkpoints import prune

        spec = plan.specs[0]
        if spec.adapter.checkpoints is None:
            return []
        deleted = prune(spec.adapter.checkpoints, spec.run_dir, keep)
        if deleted:
            self.volume.commit()
        return deleted

    def alert(self, run_id: str, message: str) -> None:
        print(f"[zephon_modal reconcile] ALERT {message}", flush=True)
        if not self.webhook:
            return
        import urllib.request

        request = urllib.request.Request(
            self.webhook, data=json.dumps({"text": message}).encode(), headers={"Content-Type": "application/json"}
        )
        try:
            urllib.request.urlopen(request, timeout=10).close()
        except OSError as error:
            print(f"[zephon_modal reconcile] alert webhook failed: {error}", flush=True)

    def mirror(self, plan: Plan, state: RunState) -> None:
        """Write the state next to the run, the durable record once the registry entry expires."""
        path = f"{plan.specs[0].run_dir}/{STATE_FILE}"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w") as tmp_file:
            json.dump(state.to_json(), tmp_file, indent=2)
        os.replace(tmp_path, path)


def reconcile_all(registry: Any, backend: Backend, now: float) -> bool:
    """One tick for every run: adopt pending starts, then tick each active run.

    Returns whether anything was saved. Only this function writes plans and
    state, and Modal runs one reconciler call at a time.
    """
    changed = False
    for run_id in registry.run_ids():
        state: RunState | None = registry.state(run_id)
        plan: Plan | None = registry.plan(run_id)
        requested: Plan | None = registry.take("start", run_id)
        if requested is not None:
            if state is not None and state.active:
                _event(state, now, "start ignored", reason=f"already {state.status}")
            else:
                previous, plan = state, requested
                state = new_state(plan, now)
                if previous is not None:
                    state.events = list(previous.events)
                _event(state, now, "started", steps=plan.steps, placements=[s.placement.name for s in plan.specs])
            registry.save(run_id, state, plan)
            changed = True
        if state is None or plan is None or not state.active:
            continue
        stop, recover = registry.take("stop", run_id), registry.take("recover", run_id)
        try:
            tick(plan, state, backend, now, stop=bool(stop), recover=bool(recover))
        except Exception as error:
            # One run's trouble mustn't stop the others; the next tick tries again.
            _event(state, now, "reconcile error", error=f"{type(error).__name__}: {error}")
        registry.save(run_id, state)
        backend.mirror(plan, state)
        changed = True
    return changed


def reconcile() -> None:
    """The ``reconcile`` Modal function: one tick for every run of this job."""
    from zephon_modal.registry import Registry

    config = json.loads(os.environ[CONFIG_ENV])
    registry = Registry.from_name(config["registry"])
    backend = ModalBackend(config["app"], config["runs_volume"], registry)
    backend.volume.reload()
    if reconcile_all(registry, backend, time.time()):
        backend.volume.commit()
