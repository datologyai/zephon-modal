# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""The job spec: one recipe, a lane count, and the placements it may run on."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import modal

from zephon_modal import runner
from zephon_modal.adapters.base import Adapter
from zephon_modal.data import DataContract
from zephon_modal.placement import Placement

RUNS_MOUNT = "/runs"
COMPILE_CACHE_MOUNT = "/compile-cache"


def new_launch_id() -> str:
    """Sortable, so the latest launch's record wins for a re-trained step.

    Made locally and passed in the spec so every node of a cluster shares it.
    """
    return time.strftime("%Y%m%dT%H%M%S", time.gmtime())


def _volume_name(volume: modal.Volume) -> str:
    """Containers commit and reload these Volumes by name, so they must have one."""
    if not volume.name:
        raise ValueError("runs_volume and compile_cache_volume must be named Volumes (modal.Volume.from_name)")
    return volume.name


@dataclass(frozen=True, eq=False, kw_only=True)
class ElasticJob:
    """A training recipe that can run, and resume, on any of several placements.

    Building one registers a Modal function per placement (plus a CPU
    ``compare`` function) on :attr:`app`, so a job file is also a normal Modal
    app module. It's frozen: those functions are built from the fields once,
    so changing a field afterwards would silently do nothing.

    Attributes:
        name: Modal app name.
        image: Training image. It needs the framework, Zephon >= 0.2.1 and
            the recipe modules; ``zephon_modal`` itself is added here.
        adapter: How to launch the framework's trainer.
        placements: Physical shapes, in order of preference.
        runs_volume: Volume holding runs, mounted at ``/runs``. Use a named v2
            Volume (``modal.Volume.from_name(..., version=2)``): every rank
            writes its own checkpoint shard, and containers commit it by name.
        volumes: Extra mounts, e.g. training data.
        data: The data loader's contract: which DP degrees it allows and
            what settings trainers see. Use :class:`zephon_modal.zephon.Zephon`
            for Zephon runs; the default allows any DP degree.
        compile_cache_volume: Optional named v2 Volume for torch.compile
            caches, which every relaunch would otherwise rebuild (Inductor
            writes cache entries via rename from many ranks at once).
        secrets: Modal secrets for the trainer.
        cpu: CPU cores per node (Zephon tokenizes and packs alongside the trainers).
        memory: Memory per node, in MiB.
        timeout: Seconds before Modal stops a launch.
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
    cpu: float = 32.0
    memory: int = 256 * 1024
    timeout: int = 2 * 60 * 60
    app: modal.App = field(init=False)
    _functions: dict[str, Any] = field(init=False, repr=False)
    _compare: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.placements:
            raise ValueError("A job needs at least one placement")
        names = [placement.name for placement in self.placements]
        if len(set(names)) != len(names):
            raise ValueError(f"Placement names must be unique, got {names}")
        _volume_name(self.runs_volume)
        if self.compile_cache_volume is not None:
            _volume_name(self.compile_cache_volume)
        for placement in self.placements:
            dp = placement.dp_size(self.adapter.model_parallel_size)
            try:
                self.data.check_dp_size(dp)
            except ValueError as error:
                raise ValueError(f"Placement {placement.name} gives DP={dp}: {error}") from error
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
            gpu=placement.gpu,
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

    def placement(self, name: str | None = None) -> Placement:
        """A placement by name, or the most preferred one."""
        if name is None:
            return self.placements[0]
        for placement in self.placements:
            if placement.name == name:
                return placement
        raise KeyError(f"No placement {name!r}; choose from {[p.name for p in self.placements]}")

    def run_spec(self, run_id: str, steps: int, placement: Placement, seed: int = 42) -> runner.RunSpec:
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
        )

    def launch(
        self, run_id: str, steps: int, placement: str | None = None, seed: int = 42, detach: bool = False
    ) -> list[str]:
        """Train ``run_id`` up to ``steps`` on one placement, resuming if it exists.

        Returns the checkpoint steps present when the launch ends.
        """
        chosen = self.placement(placement)
        spec = self.run_spec(run_id, steps, chosen, seed)
        with modal.enable_output(), self.app.run(detach=detach):
            return self._functions[chosen.name].remote(spec)

    def compare(self, reference: str, candidate: str) -> list[dict]:
        """Compare two runs' per-step sample hashes."""
        with self.app.run():
            return self._compare.remote(RUNS_MOUNT, _volume_name(self.runs_volume), reference, candidate)
