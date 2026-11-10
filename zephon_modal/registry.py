# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Where a deployed job keeps its runs' plans, state and operator requests.

One Modal Dict per job app, named ``<app>-runs``. The CLI only ever writes
*requests*; the reconciler is the only writer of plans and state, and Modal
runs one reconciler call at a time, so no entry has two writers. Keys are
tuples:

- ``("plan", run_id)``: the :class:`~zephon_modal.reconcile.Plan` in force.
- ``("state", run_id)``: the :class:`~zephon_modal.reconcile.RunState`.
- ``("start", run_id)``: a pending start, holding the requested plan.
- ``("stop", run_id)``, ``("recover", run_id)``: pending requests.
- ``("attempt", run_id, launch_id, task_id)``: one container's attempt record,
  written only by that container: when it starts, at every heartbeat, and
  with ``ended`` once its last Volume commit has returned. These are what
  the reconciler decides from, so they live here, where a write is atomic
  and immediately visible, not on the Volume, where a reader can catch a
  commit half-applied.

Modal expires a Dict entry after seven days with no reads or writes. The
reconciler reads an active run's plan and writes its state every few minutes, and
also mirrors the state to the run's directory on the Volume, which is the
durable record once a finished run's entries expire.
"""

from __future__ import annotations

from typing import Any

REQUESTS = ("start", "stop", "recover")


def registry_name(app_name: str) -> str:
    """The Modal Dict holding a job app's runs."""
    return f"{app_name}-runs"


class Registry:
    """A job's runs, over a Modal Dict (or any dict-like store, for tests)."""

    def __init__(self, store: Any) -> None:
        self.store = store

    @classmethod
    def from_name(cls, name: str) -> Registry:
        import modal

        return cls(modal.Dict.from_name(name, create_if_missing=True))

    def plan(self, run_id: str) -> Any:
        return self.store.get(("plan", run_id))

    def state(self, run_id: str) -> Any:
        return self.store.get(("state", run_id))

    def save(self, run_id: str, state: Any, plan: Any = None) -> None:
        """Write a run's state (and plan); only the reconciler calls this."""
        if plan is not None:
            self.store.put(("plan", run_id), plan)
        self.store.put(("state", run_id), state)

    def run_ids(self) -> list[str]:
        """Runs with state or a pending start."""
        return sorted({key[1] for key in self.store.keys() if isinstance(key, tuple) and key[0] in ("state", "start")})

    def record(self, run_id: str, launch_id: str, task_id: str, record: dict[str, Any]) -> None:
        """Publish one container's attempt record (only that container writes it)."""
        self.store.put(("attempt", run_id, launch_id, task_id), record)

    def _attempt_keys(self, run_id: str) -> list[tuple]:
        return [
            key
            for key in self.store.keys()
            if isinstance(key, tuple) and len(key) == 4 and key[0] == "attempt" and key[1] == run_id
        ]

    def launch_records(self, run_id: str, launch_id: str) -> list[dict[str, Any]]:
        """Every container's attempt record for one launch, all nodes, oldest first."""
        records = [self.store.get(key) for key in self._attempt_keys(run_id) if key[2] == launch_id]
        return sorted((record for record in records if record), key=lambda record: record["started"])

    def run_records(self, run_id: str) -> list[dict[str, Any]]:
        """Every attempt record of every launch of the run, each with its ``launch_id``."""
        records = []
        for key in self._attempt_keys(run_id):
            record = self.store.get(key)
            if record:
                records.append(record | {"launch_id": key[2]})
        return sorted(records, key=lambda record: record["started"])

    def pending(self, kind: str, run_id: str) -> Any:
        """A pending request's value, without consuming it."""
        return self.store.get((kind, run_id))

    def request(self, kind: str, run_id: str, value: Any = True) -> None:
        if kind not in REQUESTS:
            raise ValueError(f"Unknown request {kind!r}")
        self.store.put((kind, run_id), value)

    def take(self, kind: str, run_id: str) -> Any:
        """Consume a pending request; returns its value, or ``None`` if there was none."""
        return self.store.pop((kind, run_id), None)
