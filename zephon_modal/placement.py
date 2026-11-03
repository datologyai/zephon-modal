# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Physical shapes a run can train on."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Modal clusters must request every GPU on each node.
GPUS_PER_FULL_NODE = 8


@dataclass(frozen=True)
class Placement:
    """One physical shape: a GPU spec per node and a node count.

    There's no preemption setting: Modal doesn't offer non-preemptible GPU
    Functions, so placements differ in availability, not preemptibility.

    Attributes:
        gpu: Modal GPU spec for each node, e.g. ``"H100:8"``.
        nodes: Number of nodes; more than one runs as a ``@modal.clustered`` function.
    """

    gpu: str
    nodes: int = 1

    def __post_init__(self) -> None:
        if self.nodes < 1:
            raise ValueError(f"nodes must be at least 1, got {self.nodes}")
        if self.nodes > 1 and self.gpus_per_node != GPUS_PER_FULL_NODE:
            raise ValueError(f"Multi-node placements need whole nodes ({GPUS_PER_FULL_NODE} GPUs); got {self.gpu!r}")

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
        return name

    def dp_size(self, model_parallel_size: int = 1) -> int:
        """Data-parallel degree on this placement, given TP x PP x CP."""
        dp, remainder = divmod(self.world_size, model_parallel_size)
        if remainder or dp == 0:
            raise ValueError(
                f"Placement {self.name} has {self.world_size} GPUs, not a multiple of "
                f"model_parallel_size={model_parallel_size}"
            )
        return dp
