# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""What a training process needs to know about the launch it belongs to.

The runner builds a :class:`LaunchContext` in each container and passes it to
every trainer process as one JSON environment variable. Recipes read it with
:func:`context` rather than with their own environment variables, so the same
recipe resumes unchanged on any placement.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

CONTEXT_ENV = "ZEPHON_MODAL_CONTEXT"


@dataclass(frozen=True)
class LaunchContext:
    """One launch of a run on one placement, as seen from one node.

    Attributes:
        run_id: The run; every launch of it shares ``run_dir``.
        run_dir: Checkpoints, sample hashes and launch records live here.
        steps: Total optimizer steps to reach, not steps to add.
        seed: Data seed. Changing it mid-run changes the data.
        launch_id: Sortable launch identity plus the cluster-wide execution attempt suffix.
        placement: Name of the placement this launch runs on.
        nnodes: Nodes in this launch.
        node_rank: This node's rank.
        nproc_per_node: Trainer processes (GPUs) per node.
        sample_hash_dir: Node-local directory for per-step sample hashes,
            copied into ``run_dir`` when the launch ends.
        data: Settings from the job's :class:`~zephon_modal.data.DataContract`
            (for Zephon, read them with :func:`zephon_modal.zephon.settings`).
    """

    run_id: str
    run_dir: str
    steps: int
    seed: int
    launch_id: str
    placement: str
    nnodes: int
    node_rank: int
    nproc_per_node: int
    sample_hash_dir: str
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def world_size(self) -> int:
        return self.nnodes * self.nproc_per_node

    @property
    def keep_all_checkpoints(self) -> bool:
        """Whether the trainer must not purge old checkpoints.

        On a multi-node launch each node only sees its own uncommitted Volume
        files, so purging old steps from one node leaves the others' shards
        behind as partial checkpoints.
        """
        return self.nnodes > 1

    def to_env(self) -> dict[str, str]:
        return {CONTEXT_ENV: json.dumps(asdict(self))}

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> LaunchContext:
        env = os.environ if env is None else env
        try:
            raw = env[CONTEXT_ENV]
        except KeyError:
            raise RuntimeError(f"{CONTEXT_ENV} is not set; is this running under zephon_modal?") from None
        return cls(**json.loads(raw))


def context() -> LaunchContext:
    """The current launch's context, for use in recipes inside the training image."""
    return LaunchContext.from_env()
