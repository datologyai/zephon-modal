# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""The job spec: one recipe, a data contract, and the placements it may run on."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import modal

from zephon_modal import reconcile, runner
from zephon_modal.adapters.base import Adapter
from zephon_modal.data import DataContract
from zephon_modal.placement import Placement
from zephon_modal.reconcile import Plan, Policy, RunState
from zephon_modal.registry import Registry, registry_name
from zephon_modal.runner import new_launch_id

RUNS_MOUNT = "/runs"
COMPILE_CACHE_MOUNT = "/compile-cache"
# Modal's longest function timeout. A launch that reaches it is relaunched and resumes.
MAX_TIMEOUT = 24 * 60 * 60
MIN_TIMEOUT = 10


def _volume_name(volume: modal.Volume) -> str:
    """Containers commit and reload these Volumes by name, so they must have one."""
    if not volume.name:
        raise ValueError("runs_volume and compile_cache_volume must be named Volumes (modal.Volume.from_name)")
    return volume.name


@dataclass(frozen=True, eq=False, kw_only=True)
class ElasticJob:
    """A training recipe that can run, and resume, on any of several placements.

    Building one registers, on :attr:`app`, a GPU function per placement, a
    scheduled CPU ``reconcile`` function that keeps started runs going, and
    a CPU ``compare`` function. Deploy it (``zephon modal start`` does) and
    start runs on it; the reconciler does the rest. It's frozen: the
    functions are built from the fields once, so changing a field afterwards
    would silently do nothing.

    Attributes:
        name: Modal app name. Its run registry is a Modal Dict named ``<name>-runs``.
        image: Training image. It needs the framework, Zephon >= 0.2.1 and
            the recipe modules, and must be able to import the adapter and
            data contract classes (the reconciler unpickles them too);
            ``zephon_modal`` itself is added here.
        adapter: How to launch the framework's trainer.
        placements: Physical shapes, in order of preference.
        runs_volume: Volume holding runs, mounted at ``/runs``. Use a named v2
            Volume (``modal.Volume.from_name(..., version=2)``): every rank
            writes its own checkpoint shard, and containers commit it by name.
        data: The data loader's contract: which DP degrees it allows and
            what settings trainers see. Use :class:`zephon_modal.zephon.Zephon`
            for Zephon runs; the default allows any DP degree.
        volumes: Extra mounts, e.g. training data.
        compile_cache_volume: Optional named v2 Volume for torch.compile
            caches, which every relaunch would otherwise rebuild.
        secrets: Modal secrets for the trainer.
        alert_secret: A Modal secret providing ``ZEPHON_MODAL_ALERT_WEBHOOK``,
            a Slack-compatible incoming webhook URL for the reconciler's
            alerts (give-ups, stalls, runs needing recovery, completions).
        cpu: CPU cores per node (Zephon tokenizes and packs alongside the trainers).
        memory: Memory per node, in MiB.
        timeout: Seconds before Modal stops a launch; the reconciler then
            relaunches it, so this bounds a launch, not a run.
    """

    name: str
    image: modal.Image
    adapter: Adapter
    placements: Sequence[Placement]
    runs_volume: modal.Volume
    data: DataContract = DataContract()
    volumes: Mapping[str, modal.Volume] = field(default_factory=dict)
    compile_cache_volume: modal.Volume | None = None
    secrets: Sequence[modal.Secret] = ()
    alert_secret: modal.Secret | None = None
    cpu: float = 32.0
    memory: int = 256 * 1024
    timeout: int = MAX_TIMEOUT
    app: modal.App = field(init=False)
    _functions: dict[str, Any] = field(init=False, repr=False)
    _compare: Any = field(init=False, repr=False)
    _replays: Any = field(init=False, repr=False)
    _reconcile: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.placements:
            raise ValueError("A job needs at least one placement")
        names = [placement.name for placement in self.placements]
        if len(set(names)) != len(names):
            raise ValueError(f"Placement names must be unique, got {names}")
        if not MIN_TIMEOUT <= self.timeout <= MAX_TIMEOUT:
            raise ValueError(f"timeout must be between {MIN_TIMEOUT} and {MAX_TIMEOUT} seconds (Modal's limits)")
        _volume_name(self.runs_volume)
        if self.compile_cache_volume is not None:
            _volume_name(self.compile_cache_volume)
        for placement in self.placements:
            for shape in placement.shapes:
                dp = shape.dp_size(self.adapter.model_parallel_size)
                try:
                    self.data.check_dp_size(dp)
                except ValueError as error:
                    raise ValueError(f"Placement {shape.name} gives DP={dp}: {error}") from error
        # Frozen, so derived state goes in through object.__setattr__.
        object.__setattr__(self, "placements", tuple(self.placements))
        object.__setattr__(self, "app", modal.App(self.name))
        functions = {placement.name: self._register(placement) for placement in self.placements}
        object.__setattr__(self, "_functions", functions)
        compare = self.app.function(
            name="compare",
            image=modal.Image.debian_slim(python_version="3.12").add_local_python_source("zephon_modal"),
            volumes={RUNS_MOUNT: self.runs_volume},
        )(runner.compare)
        object.__setattr__(self, "_compare", compare)
        replays = self.app.function(
            name="replays",
            image=modal.Image.debian_slim(python_version="3.12").add_local_python_source("zephon_modal"),
            volumes={RUNS_MOUNT: self.runs_volume},
        )(runner.replays)
        object.__setattr__(self, "_replays", replays)
        config = {"app": self.name, "registry": self.registry_name, "runs_volume": _volume_name(self.runs_volume)}
        reconcile_function = self.app.function(
            name="reconcile",
            # The training image, so plans unpickle with any adapter class the job defines.
            image=self.image.add_local_python_source("zephon_modal"),
            env={reconcile.CONFIG_ENV: json.dumps(config)},
            volumes={RUNS_MOUNT: self.runs_volume},
            secrets=[self.alert_secret] if self.alert_secret is not None else [],
            schedule=modal.Period(minutes=1),
            # One decision-maker: ticks (scheduled, or triggered by the CLI) queue rather than overlap.
            max_containers=1,
            cpu=1.0,
            memory=2048,
            nonpreemptible=True,
            timeout=10 * 60,
        )(reconcile.reconcile)
        object.__setattr__(self, "_reconcile", reconcile_function)

    def _register(self, placement: Placement) -> Any:
        volumes: dict[Any, Any] = {RUNS_MOUNT: self.runs_volume, **self.volumes}
        if self.compile_cache_volume is not None:
            volumes[COMPILE_CACHE_MOUNT] = self.compile_cache_volume
        target: Any = runner.train_node
        if placement.nodes > 1:
            target = modal.clustered(size=placement.nodes, rdma=True)(target)
        return self.app.function(
            name=f"train-{placement.name}",
            image=self.image.add_local_python_source("zephon_modal"),
            gpu=placement.modal_gpu,
            cpu=self.cpu,
            memory=self.memory,
            timeout=self.timeout,
            volumes=volumes,
            secrets=list(self.secrets),
        )(target)

    @property
    def functions(self) -> Mapping[str, Any]:
        """The training function for each placement, by placement name."""
        return self._functions

    @property
    def registry_name(self) -> str:
        return registry_name(self.name)

    def registry(self) -> Registry:
        return Registry.from_name(self.registry_name)

    def placement(self, name: str | None = None) -> Placement:
        """A placement by name, or the most preferred one."""
        if name is None:
            return self.placements[0]
        for placement in self.placements:
            if placement.name == name:
                return placement
        raise KeyError(f"No placement {name!r}; choose from {[p.name for p in self.placements]}")

    def run_spec(
        self, run_id: str, steps: int, placement: Placement, seed: int = 42, *, registry: str | None = None
    ) -> runner.RunSpec:
        return runner.RunSpec(
            run_id=run_id,
            steps=steps,
            seed=seed,
            data=self.data,
            launch_id=new_launch_id(),
            placement=placement,
            adapter=self.adapter,
            runs_mount=RUNS_MOUNT,
            runs_volume_name=_volume_name(self.runs_volume),
            compile_cache_mount=COMPILE_CACHE_MOUNT if self.compile_cache_volume else None,
            compile_cache_volume_name=_volume_name(self.compile_cache_volume) if self.compile_cache_volume else None,
            registry_name=registry,
        )

    def deploy(self) -> None:
        """Deploy the app, so its reconciler runs; needed again after changing the job or the library."""
        with modal.enable_output():
            self.app.deploy()

    def _tick(self) -> None:
        """Ask the reconciler to look now rather than within the minute."""
        modal.Function.from_name(self.name, "reconcile").spawn()

    def start(self, run_id: str, steps: int, seed: int = 42, policy: Policy = Policy(), deploy: bool = True) -> None:
        """Ask the reconciler to train ``run_id`` to ``steps`` across placements.

        A finished, stopped or abandoned run can be started again with a
        higher (or the same) step target; it resumes from its checkpoints.
        The reconciler adopts the request on its next tick, and ignores it
        (with an event in ``status``) if the run is still active.

        Raises:
            RuntimeError: if the run is already active.
        """
        registry = self.registry()
        existing: RunState | None = registry.state(run_id)
        if existing is not None and existing.active:
            raise RuntimeError(f"{run_id} is already {existing.status}; stop it first")
        if deploy:
            self.deploy()
        plan = Plan(
            run_id=run_id,
            steps=steps,
            specs=[
                self.run_spec(run_id, steps, placement, seed, registry=self.registry_name)
                for placement in self.placements
            ],
            policy=policy,
        )
        registry.request("start", run_id, plan)
        self._tick()

    def status(self, run_id: str) -> dict[str, Any] | None:
        """The reconciler's record of the run, or ``None`` if it was never started."""
        registry = self.registry()
        state = registry.state(run_id)
        requested = registry.pending("start", run_id)
        if requested is not None and (state is None or not state.active):
            return {"run_id": run_id, "status": "start requested", "steps": requested.steps, "checkpoint": None}
        if state is not None:
            return state.to_json()
        try:
            content = b"".join(self.runs_volume.read_file(f"{run_id}/{reconcile.STATE_FILE}"))
        except (FileNotFoundError, modal.exception.NotFoundError):
            return None
        return json.loads(content)

    def stop(self, run_id: str) -> None:
        """Ask the reconciler to stop the run: it cancels the launch, which stops its trainer and commits."""
        state = self.registry().state(run_id)
        if state is None or not state.active:
            raise RuntimeError(f"{run_id} isn't running")
        self.registry().request("stop", run_id)
        self._tick()

    def recover(self, run_id: str) -> None:
        """Let a run that's waiting on an unconfirmed launch continue.

        Only once you've checked (in Modal's dashboard, or with ``modal
        container list``) that none of that launch's containers is still running.
        """
        state = self.registry().state(run_id)
        if state is None or state.status != "needs recovery":
            raise RuntimeError(f"{run_id} isn't waiting for recovery")
        self.registry().request("recover", run_id)
        self._tick()

    def launch(
        self, run_id: str, steps: int, placement: str | None = None, seed: int = 42, detach: bool = False
    ) -> list[str]:
        """Train ``run_id`` up to ``steps`` on one placement, here and now, without the reconciler.

        For references and experiments. Returns the checkpoint directories
        present when the launch ends.
        """
        state = self.registry().state(run_id)
        if state is not None and state.active:
            raise RuntimeError(f"{run_id} is {state.status} under the reconciler; stop it first")
        chosen = self.placement(placement)
        spec = self.run_spec(run_id, steps, chosen, seed)
        with modal.enable_output(), self.app.run(detach=detach):
            return self._functions[chosen.name].remote(spec)

    def replays(self, run_id: str) -> list[dict]:
        """Check the steps ``run_id`` trained more than once (after resumes) against each other."""
        with self.app.run():
            return self._replays.remote(RUNS_MOUNT, _volume_name(self.runs_volume), run_id)

    def compare(
        self,
        reference: str,
        candidate: str,
        *,
        steps: int | None = None,
        start_step: int = 1,
        rows_per_step: int | None = None,
    ) -> list[dict]:
        """Compare two runs' per-step sample hashes."""
        with self.app.run():
            return self._compare.remote(
                RUNS_MOUNT, _volume_name(self.runs_volume), reference, candidate, steps, start_step, rows_per_step
            )
