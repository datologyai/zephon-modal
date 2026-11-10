"""The reconciler's Modal side, against a fake registry, a temporary Volume and fake calls."""

import dataclasses
import subprocess
from datetime import datetime, timedelta, timezone

import pytest
from modal import exception
from test_reconcile import FakeBackend, _plan

from zephon_modal.reconcile import Call, ModalBackend, RunState, new_state, observe_call, reconcile_all
from zephon_modal.registry import Registry


class DictStore(dict):
    def put(self, key, value, skip_if_exists=False):
        self[key] = value


def _iso(offset_seconds: float = 0) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=offset_seconds)).isoformat(timespec="seconds")


class Mirror(FakeBackend):
    def mirror(self, plan, state):
        pass


def test_a_start_request_is_adopted_by_the_reconciler():
    registry, backend = Registry(DictStore()), Mirror()
    registry.request("start", "r1", _plan())
    assert reconcile_all(registry, backend, now=1000)
    state = registry.state("r1")
    assert state.status == "running" and state.current["placement"] == "2xh100x8"
    assert registry.plan("r1").steps == 100
    assert registry.pending("start", "r1") is None
    assert [event["event"] for event in state.events][:2] == ["started", "launched"]


def test_a_start_request_for_an_active_run_is_ignored():
    registry, backend = Registry(DictStore()), Mirror()
    registry.request("start", "r1", _plan())
    reconcile_all(registry, backend, now=1000)
    registry.request("start", "r1", _plan())
    reconcile_all(registry, backend, now=1060)
    assert len(backend.spawned) == 1
    assert "start ignored" in [event["event"] for event in registry.state("r1").events]


def test_restarting_a_finished_run_keeps_its_history():
    registry, backend = Registry(DictStore()), Mirror()
    finished = new_state(_plan(), 0)
    finished.status = "stopped"
    finished.events = [{"time": "t", "event": "stopped"}]
    registry.save("r1", finished, _plan())
    registry.request("start", "r1", _plan())
    reconcile_all(registry, backend, now=1000)
    state = registry.state("r1")
    assert state.status == "running"
    assert [event["event"] for event in state.events][:2] == ["stopped", "started"]


def test_one_runs_error_doesnt_stop_the_others():
    registry, backend = Registry(DictStore()), Mirror()
    for run_id in ("a", "b"):
        registry.request("start", run_id, dataclasses.replace(_plan(), run_id=run_id))
    calls = []

    def observe(plan, state):
        calls.append(state.run_id)
        if state.run_id == "a":
            raise ConnectionError("Volume unavailable")
        return backend.observation

    backend.observe = observe
    reconcile_all(registry, backend, now=1000)
    assert calls == ["a", "b"]
    assert registry.state("a").events[-1]["event"] == "reconcile error"
    assert registry.state("b").current is not None


@pytest.fixture
def modal_backend(tmp_path, monkeypatch):
    calls = {}

    class FakeCall:
        def __init__(self, call_id):
            self.call_id = call_id

        def get(self, timeout=None):
            outcome = calls[self.call_id]
            if outcome is None:
                raise TimeoutError
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

    import modal

    monkeypatch.setattr(modal.FunctionCall, "from_id", staticmethod(lambda call_id: FakeCall(call_id)))
    backend = ModalBackend("app", "runs", Registry(DictStore()))
    plan = _plan()
    plan = type(plan)(
        run_id=plan.run_id,
        steps=plan.steps,
        specs=[type(spec)(**{**spec.__dict__, "runs_mount": str(tmp_path)}) for spec in plan.specs],
        policy=plan.policy,
    )
    return backend, plan, calls, tmp_path


def _record(backend, launch_id, task_id, **fields):
    record = {"task_id": task_id, "node_rank": 0, "started": _iso(-600), "allocated": "h100x8"} | fields
    backend.registry.record("r1", launch_id, task_id, record)


def _state_with_launch(plan, launch_id="L1", call_id="fc-1") -> RunState:
    state = new_state(plan, 0)
    state.current = {"placement": "h100x8", "launch_id": launch_id, "call_id": call_id}
    return state


def test_observe_queued_running_and_silent(modal_backend):
    backend, plan, calls, tmp_path = modal_backend
    calls["fc-1"] = None
    state = _state_with_launch(plan)
    assert backend.observe(plan, state).call is Call.QUEUED
    _record(backend, "L1", "ta-1", heartbeat=_iso(-10))
    observation = backend.observe(plan, state)
    assert (observation.call, observation.drained) == (Call.RUNNING, False)
    assert observation.attempts == 1
    _record(backend, "L1", "ta-1", heartbeat=_iso(-3600))
    assert backend.observe(plan, state).call is Call.SILENT


def test_an_api_error_falls_back_to_the_heartbeats(modal_backend):
    backend, plan, calls, tmp_path = modal_backend
    calls["fc-1"] = exception.ConnectionError("network")
    state = _state_with_launch(plan)
    # No container running: queued, or ended? Can't tell.
    assert backend.observe(plan, state).call is Call.UNKNOWN
    _record(backend, "L1", "ta-1", heartbeat=_iso(-10))
    observation = backend.observe(plan, state)
    assert (observation.call, observation.attempts) == (Call.RUNNING, 1)
    _record(backend, "L1", "ta-1", heartbeat=_iso(-3600))
    assert backend.observe(plan, state).call is Call.SILENT


def test_every_node_of_the_running_attempt_must_heartbeat(modal_backend):
    backend, plan, calls, tmp_path = modal_backend
    calls["fc-1"] = None
    state = _state_with_launch(plan)
    # The first attempt's node 1 died silently; it's not this attempt's.
    _record(backend, "L1", "ta-1", started=_iso(-900), ended=_iso(-800), attempt_tag=".a1")
    _record(backend, "L1", "ta-2", node_rank=1, started=_iso(-900), heartbeat=_iso(-850), attempt_tag=".a1")
    _record(backend, "L1", "ta-3", heartbeat=_iso(-10), attempt_tag=".a2")
    _record(backend, "L1", "ta-4", node_rank=1, heartbeat=_iso(-10), attempt_tag=".a2")
    assert backend.observe(plan, state).call is Call.RUNNING
    _record(backend, "L1", "ta-4", node_rank=1, heartbeat=_iso(-600), attempt_tag=".a2")
    observation = backend.observe(plan, state)
    assert observation.call is Call.SILENT and "node 1: no heartbeat" in observation.error
    _record(backend, "L1", "ta-4", node_rank=1, ended=_iso(-5), outcome="failed", attempt_tag=".a2")
    observation = backend.observe(plan, state)
    assert observation.call is Call.SILENT and "node 1 exited (failed)" in observation.error


def test_observe_a_restart_pending_after_a_preemption(modal_backend):
    backend, plan, calls, tmp_path = modal_backend
    calls["fc-1"] = None
    _record(backend, "L1", "ta-1", ended=_iso(-5), outcome="stopped")
    assert backend.observe(plan, _state_with_launch(plan)).call is Call.QUEUED


def test_drained_needs_every_container_of_every_node(modal_backend):
    backend, plan, calls, tmp_path = modal_backend
    calls["fc-1"] = RuntimeError("boom")
    _record(backend, "L1", "ta-1", ended=_iso(-5))
    _record(backend, "L1", "ta-2", node_rank=1)
    observation = backend.observe(plan, _state_with_launch(plan))
    assert (observation.call, observation.drained) == (Call.FAILED, False)
    _record(backend, "L1", "ta-2", node_rank=1, ended=_iso(-1))
    assert backend.observe(plan, _state_with_launch(plan)).drained


class _Raises:
    def __init__(self, error=None):
        self.error = error

    def get(self, timeout=None):
        if self.error is not None:
            raise self.error
        return ["step-10"]


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (None, Call.SUCCEEDED),
        (TimeoutError(), None),
        (exception.FunctionTimeoutError("24h"), Call.TIMED_OUT),
        (exception.InputCancellation(), Call.CANCELLED),
        (exception.RemoteError("cancelled"), Call.FAILED),
        (exception.InternalFailure("worker lost"), Call.LOST),
        (exception.ExecutionError("container crashed"), Call.FAILED),
        (subprocess.CalledProcessError(1, ["torchrun"]), Call.FAILED),
        (exception.ConnectionError("network"), Call.UNKNOWN),
        (exception.ServiceError("unavailable"), Call.UNKNOWN),
        (exception.TimeoutError("rpc deadline"), Call.UNKNOWN),
        (ConnectionResetError(), Call.UNKNOWN),
    ],
)
def test_call_outcomes_from_results_end_the_call_but_asking_errors_dont(error, expected):
    assert observe_call(_Raises(error))[0] is expected
