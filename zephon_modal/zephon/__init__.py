# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Zephon as a job's data loader: which DP degrees it allows, and its settings.

Pass :class:`Zephon` as an :class:`~zephon_modal.ElasticJob`'s ``data``, and
read the settings in recipes with :func:`settings`. The hashing loader and
Megatron entry point that let ``compare`` check a run live alongside this
module (``zephon_modal.zephon.torchtitan``, ``zephon_modal.zephon.megatron_entry``)
and only import inside training images.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from zephon_modal.context import LaunchContext, context
from zephon_modal.data import DataContract

# Zephon merges lane state across ranks at checkpoint time. Modal Volume writes
# only reach other containers after commit + reload, so a shared aggregate_dir
# can't work across nodes, and on one node it adds nothing (restore reads the
# merged state from the trainer's checkpoint). So every launch, single- or
# multi-node, uses torchrun's own c10d TCPStore.
AGGREGATE_STORE = "torch://"


@dataclass(frozen=True)
class Zephon(DataContract):
    """Zephon's elastic-determinism contract.

    Attributes:
        canonical_replicas: Zephon's ``canonical_replicas`` (the run's lanes).
            Every placement's DP degree must divide it, and the recipe must
            consume whole lane windows per optimizer step.
    """

    canonical_replicas: int

    def check_dp_size(self, dp: int) -> None:
        if self.canonical_replicas % dp:
            raise ValueError(
                f"doesn't divide canonical_replicas={self.canonical_replicas}; "
                "Zephon can only train and resume on DP degrees that divide it"
            )

    def context(self) -> dict[str, Any]:
        return {"canonical_replicas": self.canonical_replicas, "aggregate_store": AGGREGATE_STORE}

    def describe_dp_size(self, dp: int) -> str:
        return f"lanes/rank {self.canonical_replicas // dp}"


@dataclass(frozen=True)
class ZephonSettings:
    """What a recipe passes to Zephon for this launch.

    Attributes:
        canonical_replicas: Zephon's ``canonical_replicas``.
        aggregate_store: Zephon's ``aggregate_store`` (always ``"torch://"``).
    """

    canonical_replicas: int
    aggregate_store: str


def settings(ctx: LaunchContext | None = None) -> ZephonSettings:
    """This launch's Zephon settings, for recipes inside the training image."""
    data = (ctx or context()).data
    try:
        return ZephonSettings(canonical_replicas=data["canonical_replicas"], aggregate_store=data["aggregate_store"])
    except KeyError:
        raise RuntimeError("This job's data contract isn't zephon_modal.zephon.Zephon") from None
