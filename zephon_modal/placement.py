# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Physical shapes a run can train on."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

# Modal clusters must request every GPU on each node.
GPUS_PER_FULL_NODE = 8


@dataclass(frozen=True)
class Placement:
    """One physical shape, a GPU spec per node and a node count, as one Modal function.

    There's no preemption setting: Modal doesn't offer non-preemptible GPU
    Functions, so placements differ in availability, not preemptibility.

    ``fallbacks`` turns a single-node placement into a Modal GPU fallback
    list: ``Placement("H100:8", fallbacks=("H100:4",))`` asks Modal for
    ``["H100:8", "H100:4"]``, and Modal's scheduler gives it the first one it
    can allocate. The launch sizes itself from the GPUs it got, so a run waits
    for whichever shape is free instead of for one in particular.

    Attributes:
        gpu: Modal GPU spec for each node, e.g. ``"H100:8"``.
        nodes: Number of nodes; more than one runs as a ``@modal.clustered`` function.
        fallbacks: Single-node GPU specs to accept, in order, when ``gpu`` isn't available.
    """

    gpu: str
    nodes: int = 1
    fallbacks: Sequence[str] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "fallbacks", tuple(self.fallbacks))
        if self.nodes < 1:
            raise ValueError(f"nodes must be at least 1, got {self.nodes}")
        if self.nodes > 1 and self.gpus_per_node != GPUS_PER_FULL_NODE:
            raise ValueError(f"Multi-node placements need whole nodes ({GPUS_PER_FULL_NODE} GPUs); got {self.gpu!r}")
        if self.nodes > 1 and self.fallbacks:
            raise ValueError("GPU fallbacks on clusters are untested here; make each cluster size its own placement")
        counts = [self.gpus_per_node, *(Placement(gpu).gpus_per_node for gpu in self.fallbacks)]
        if len(set(counts)) != len(counts):
            # The launch tells the shapes apart by how many GPUs it got.
            raise ValueError(f"GPU fallbacks need distinct GPU counts, got {[self.gpu, *self.fallbacks]}")

    @property
    def gpu_type(self) -> str:
        return self.gpu.split(":")[0]

    @property
    def gpus_per_node(self) -> int:
        _, _, count = self.gpu.partition(":")
        return int(count) if count else 1

    @property
    def world_size(self) -> int:
        return self.nodes * self.gpus_per_node

    @property
    def name(self) -> str:
        """A stable identifier, also used in the placement's Modal function name."""
        slug = re.sub(r"[^a-z0-9]+", "", self.gpu_type.lower())
        name = f"{slug}x{self.gpus_per_node}"
        if self.nodes > 1:
            name = f"{self.nodes}x{name}"
        return "-or-".join([name, *(Placement(gpu).name for gpu in self.fallbacks)])

    @property
    def shapes(self) -> tuple[Placement, ...]:
        """Every shape a launch here can get: this one without fallbacks, then each fallback."""
        return (Placement(self.gpu, self.nodes), *(Placement(gpu) for gpu in self.fallbacks))

    @property
    def modal_gpu(self) -> str | list[str]:
        """The ``gpu`` argument for this placement's Modal function."""
        return [self.gpu, *self.fallbacks] if self.fallbacks else self.gpu

    def allocated(self, gpus_per_node: int) -> Placement:
        """The shape a launch got, from the number of GPUs it sees."""
        for shape in self.shapes:
            if shape.gpus_per_node == gpus_per_node:
                return shape
        raise RuntimeError(f"Placement {self.name} got {gpus_per_node} GPUs per node, matching none of its shapes")

    def dp_size(self, model_parallel_size: int = 1) -> int:
        """Data-parallel degree on this placement's preferred shape, given TP x PP x CP."""
        dp, remainder = divmod(self.world_size, model_parallel_size)
        if remainder or dp == 0:
            raise ValueError(
                f"Placement {self.name} has {self.world_size} GPUs, not a multiple of "
                f"model_parallel_size={model_parallel_size}"
            )
        return dp
