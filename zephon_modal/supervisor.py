# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Keeps a run going across placements until it reaches its step target.

The supervisor is a small, non-preemptible CPU function in the job's app. It
spawns one launch at a time on the most preferred placement that isn't
cooling down, and watches it:

- Queued longer than ``queue_timeout``: cancel it, cool that placement down,
  and try the next one.
- Restarted by Modal more than ``max_restarts`` times (preemptions): cancel
  it, cool the placement down, and try the next one. Fewer restarts are left
  to Modal, which resumes on the same placement.
- Hit the function timeout: launch again (the run resumes from its checkpoint).
- Failed: launch again, up to ``max_failures`` times in a row, then give up.
- Finished: done.

Cooled-down placements become eligible again after ``cooldown``, so the next
launch goes back to a preferred placement once it may have capacity again.
The supervisor never stops a healthy launch to move up.

Its state lives in ``<run_dir>/supervisor.json`` on the runs Volume, which
``status`` reads; ``cancel`` cancels the supervisor's own call, and it cancels
its launch on the way out.
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

from zephon_modal.runner import RunSpec, launch_attempts, new_launch_id

SUPERVISOR_FILE = "supervisor.json"


@dataclass(frozen=True)
class Policy:
    """When the supervisor gives up on a placement.

    Attributes:
        queue_timeout: Seconds a launch may wait for containers (at first, or
            after a restart) before the supervisor tries the next placement.
        max_restarts: Modal restarts (preemptions) one launch may have before
            the supervisor moves it to the next placement.
        max_failures: Consecutive failed launches before the supervisor gives up.
        cooldown: Seconds a placement is skipped after the supervisor gives up on it.
        poll_interval: Seconds between checks on the current launch.
    """

    queue_timeout: float = 15 * 60
    max_restarts: int = 2
    max_failures: int = 1
    cooldown: float = 30 * 60
    poll_interval: float = 15


class CallStatus(enum.Enum):
    """Where a launch's Modal call stands."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed out"


@dataclass(frozen=True)
class CallState:
    """What a launch's Modal call is doing now."""

    status: CallStatus
    task_id: str | None = None
    result: Any = None
    error: str | None = None


def state_from_attempts(attempts: Sequence[Mapping[str, Any]]) -> CallState:
    """An unfinished launch's state, from node 0's attempt records (oldest first).

    Each container records its attempt when it starts and again when it ends,
    so no record means the launch is queued, and an attempt that was stopped
    with no newer one means Modal's restart is queued. (Modal's call graph
    would be the natural source, but its ``task_id`` stayed empty for a
    running launch.)
    """
    if not attempts:
        return CallState(CallStatus.QUEUED)
    latest = attempts[-1]
    if latest.get("outcome") in ("stopped", "cancelled"):
        return CallState(CallStatus.QUEUED)
    return CallState(CallStatus.RUNNING, task_id=latest["task_id"])


class Backend:
    """How the supervisor spawns, watches and cancels launches, and keeps its state.

    :class:`ModalBackend` is the real one; tests use a fake.
    """

    def spawn(self, spec: RunSpec) -> str:
        """Start a launch on ``spec.placement``; returns its call ID."""
        raise NotImplementedError

    def state(self, call_id: str) -> CallState:
        raise NotImplementedError

    def cancel(self, call_id: str) -> None:
        raise NotImplementedError

    def save(self, record: Mapping[str, Any]) -> None:
        """Persist the supervisor's state."""
        raise NotImplementedError

    def now(self) -> float:
        return time.time()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


@dataclass(frozen=True)
class Plan:
    """What the supervisor runs: a spec per placement, in order of preference."""

    run_id: str
    steps: int
    specs: Sequence[RunSpec]
    policy: Policy = Policy()


class GaveUp(Exception):
    """The supervisor stopped without reaching the step target."""


def _iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat(timespec="seconds")


@dataclass
class Supervisor:
    """The supervisor's decisions, independent of Modal."""

    plan: Plan
    backend: Backend
    supervisor_call_id: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    cooling_until: dict[str, float] = field(default_factory=dict)
    current: dict[str, Any] | None = None
    status: str = "running"

    @property
    def policy(self) -> Policy:
        return self.plan.policy

    def _event(self, event: str, **detail: Any) -> None:
        self.events.append({"time": _iso(self.backend.now()), "event": event, **detail})
        print(f"[zephon_modal supervisor] {event} {detail}", flush=True)
        self._save()

    def _save(self) -> None:
        self.backend.save(
            {
                "run_id": self.plan.run_id,
                "steps": self.plan.steps,
                "status": self.status,
                "supervisor_call_id": self.supervisor_call_id,
                "placements": [spec.placement.name for spec in self.plan.specs],
                "policy": dataclasses.asdict(self.policy),
                "current": self.current,
                "cooling_until": {name: _iso(until) for name, until in self.cooling_until.items()},
                "events": self.events,
            }
        )

    def choose(self) -> RunSpec:
        """The most preferred placement not cooling down; if all are, the one free soonest."""
        now = self.backend.now()
        for spec in self.plan.specs:
            if self.cooling_until.get(spec.placement.name, 0) <= now:
                return spec
        return min(self.plan.specs, key=lambda spec: self.cooling_until[spec.placement.name])

    def _cool_down(self, placement: str) -> None:
        self.cooling_until[placement] = self.backend.now() + self.policy.cooldown

    def _launch(self) -> str:
        spec = dataclasses.replace(self.choose(), steps=self.plan.steps, launch_id=new_launch_id())
        call_id = self.backend.spawn(spec)
        self.current = {
            "placement": spec.placement.name,
            "call_id": call_id,
            "launch_id": spec.launch_id,
            "started": _iso(self.backend.now()),
        }
        self._event("launched", placement=spec.placement.name, launch_id=spec.launch_id, call_id=call_id)
        return call_id

    def run(self) -> Any:
        """Supervise until the run reaches its step target; returns the final launch's result.

        Raises:
            GaveUp: after ``max_failures`` consecutive failed launches.
        """
        failures = 0
        call_id = self._launch()
        queued_since: float | None = self.backend.now()
        tasks: list[str] = []
        try:
            while True:
                self.backend.sleep(self.policy.poll_interval)
                state = self.backend.state(call_id)
                placement = self.current["placement"] if self.current else "?"
                relaunch = False
                if state.status is CallStatus.SUCCEEDED:
                    self.status = "completed"
                    self._event("completed", placement=placement)
                    return state.result
                if state.status is CallStatus.TIMED_OUT:
                    self._event("timed out", placement=placement)
                    relaunch = True
                elif state.status is CallStatus.FAILED:
                    failures += 1
                    self._event("failed", placement=placement, error=state.error, failures=failures)
                    if failures >= self.policy.max_failures:
                        self.status = "failed"
                        self._event("gave up", reason=f"{failures} failed launch(es) in a row")
                        raise GaveUp(state.error)
                    relaunch = True
                elif state.status is CallStatus.QUEUED:
                    queued_since = queued_since if queued_since is not None else self.backend.now()
                    if self.backend.now() - queued_since > self.policy.queue_timeout:
                        self.backend.cancel(call_id)
                        self._cool_down(placement)
                        self._event("queue timeout", placement=placement, queued_seconds=self.policy.queue_timeout)
                        relaunch = True
                elif state.status is CallStatus.RUNNING:
                    queued_since = None
                    if state.task_id not in tasks:
                        tasks.append(state.task_id or "")
                        if len(tasks) == 1:
                            failures = 0
                            self._event("running", placement=placement, task_id=state.task_id)
                        else:
                            self._event(
                                "restarted", placement=placement, task_id=state.task_id, restarts=len(tasks) - 1
                            )
                        if len(tasks) - 1 > self.policy.max_restarts:
                            self.backend.cancel(call_id)
                            self._cool_down(placement)
                            self._event("too many restarts", placement=placement, restarts=len(tasks) - 1)
                            relaunch = True
                if relaunch:
                    call_id = self._launch()
                    queued_since = self.backend.now()
                    tasks = []
        except BaseException as error:
            # Cancelled (InputCancellation), or a bug: don't leave a launch running unsupervised.
            if self.status == "running":
                self.backend.cancel(call_id)
                self.status = "cancelled" if type(error).__name__ == "InputCancellation" else "failed"
                self._event(self.status, placement=self.current["placement"] if self.current else None)
            raise


class ModalBackend(Backend):
    """Spawns launches as calls to the job's ``train-<placement>`` functions."""

    def __init__(self, functions: Mapping[str, Any], runs_mount: str, runs_volume_name: str, run_id: str) -> None:
        import modal

        self._functions = functions
        self._calls: dict[str, tuple[Any, RunSpec]] = {}
        self._path = f"{runs_mount}/{run_id}/{SUPERVISOR_FILE}"
        self._volume = modal.Volume.from_name(runs_volume_name)

    def spawn(self, spec: RunSpec) -> str:
        call = self._functions[spec.placement.name].spawn(spec)
        self._calls[call.object_id] = (call, spec)
        return call.object_id

    def state(self, call_id: str) -> CallState:
        from modal.exception import FunctionTimeoutError

        call, spec = self._calls[call_id]
        try:
            return CallState(CallStatus.SUCCEEDED, result=call.get(timeout=0))
        except TimeoutError:
            pass  # still going (the builtin TimeoutError; Modal's own is FunctionTimeoutError)
        except FunctionTimeoutError as error:
            return CallState(CallStatus.TIMED_OUT, error=str(error))
        except Exception as error:
            return CallState(CallStatus.FAILED, error=f"{type(error).__name__}: {error}")
        self._volume.reload()
        return state_from_attempts(launch_attempts(spec.run_dir, spec.launch_id))

    def cancel(self, call_id: str) -> None:
        # Not terminate_containers: the launch gets InputCancellation and stops
        # its trainer and commits, as on a preemption.
        self._calls[call_id][0].cancel()

    def save(self, record: Mapping[str, Any]) -> None:
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        tmp_path = f"{self._path}.tmp"
        with open(tmp_path, "w") as tmp_file:
            json.dump(record, tmp_file, indent=2)
        os.replace(tmp_path, self._path)
        self._volume.commit()


def supervise(plan: Plan, functions: Mapping[str, Any], runs_mount: str, runs_volume_name: str) -> Any:
    """The ``supervise`` Modal function: run ``plan`` to completion."""
    import modal

    backend = ModalBackend(functions, runs_mount, runs_volume_name, plan.run_id)
    return Supervisor(plan, backend, supervisor_call_id=modal.current_function_call_id()).run()
