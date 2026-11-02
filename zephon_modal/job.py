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
from zephon_modal.placement import Placement

RUNS_MOUNT = "/runs"
COMPILE_CACHE_MOUNT = "/compile-cache"


def new_launch_id() -> str:
    """Sortable, so the latest launch's record wins for a re-trained step.

    Made locally and passed in the spec so every node of a cluster shares it.
    """
    return time.strftime("%Y%m%dT%H%M%S", time.gmtime())


@dataclass
class ElasticJob:
    """A training recipe that can run, and resume, on any of several placements.

    Building one registers a Modal function per placement (plus a CPU
    ``compare`` function) on :attr:`app`, so a job file is also a normal Modal
    app module.

    Attributes:
        name: Modal app name.
        image: Training image. It needs the framework, Zephon >= 0.2.1 and
            the recipe modules; ``zephon_modal`` itself is added here.
        adapter: How to launch the framework's trainer.
        lanes: Zephon ``canonical_replicas``. Every placement's DP degree must
            divide it, and the recipe must consume whole lane windows per step.
        placements: Physical shapes, in order of preference.
        runs_volume: Name of the Volume holding runs (created as a v2 Volume
            if missing: every rank writes its own checkpoint shard).
        volumes: Extra mounts, e.g. training data.
        compile_cache_volume: Optional Volume name for torch.compile caches,
            which every relaunch would otherwise rebuild.
        secrets: Modal secrets for the trainer.
        cpu: CPU cores per node (Zephon tokenizes and packs alongside the trainers).
        memory: Memory per node, in MiB.
        timeout: Seconds before Modal stops a launch.
    """

    name: str
    image: modal.Image
    adapter: Adapter
    lanes: int
    placements: Sequence[Placement]
    runs_volume: str
    volumes: Mapping[str, modal.Volume] = field(default_factory=dict)
    compile_cache_volume: str | None = None
    secrets: Sequence[modal.Secret] = ()
    cpu: float = 32.0
    memory: int = 256 * 1024
    timeout: int = 2 * 60 * 60

    def __post_init__(self) -> None:
        if not self.placements:
            raise ValueError("A job needs at least one placement")
        names = [placement.name for placement in self.placements]
        if len(set(names)) != len(names):
            raise ValueError(f"Placement names must be unique, got {names}")
        for placement in self.placements:
            placement.check_lanes(self.lanes, self.adapter.model_parallel_size)
        self.app = modal.App(self.name)
        self._functions = {placement.name: self._register(placement) for placement in self.placements}
        self._compare = self.app.function(
            name="compare",
            image=modal.Image.debian_slim(python_version="3.12").add_local_python_source("zephon_modal"),
            volumes={RUNS_MOUNT: self._runs_volume()},
        )(runner.compare)

    def _runs_volume(self) -> modal.Volume:
        return modal.Volume.from_name(self.runs_volume, create_if_missing=True, version=2)

    def _register(self, placement: Placement) -> Any:
        volumes: dict[Any, Any] = {RUNS_MOUNT: self._runs_volume(), **self.volumes}
        if self.compile_cache_volume is not None:
            # Inductor writes cache entries via rename from many ranks at once.
            volumes[COMPILE_CACHE_MOUNT] = modal.Volume.from_name(
                self.compile_cache_volume, create_if_missing=True, version=2
            )
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
            nonpreemptible=placement.nonpreemptible,
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
            launch_id=new_launch_id(),
            placement=placement,
            adapter=self.adapter,
            runs_mount=RUNS_MOUNT,
            runs_volume=self.runs_volume,
            compile_cache_mount=COMPILE_CACHE_MOUNT if self.compile_cache_volume else None,
            compile_cache_volume=self.compile_cache_volume,
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
            return self._compare.remote(RUNS_MOUNT, self.runs_volume, reference, candidate)
