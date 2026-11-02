# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Per-step sample fingerprints, to show a resized run sees identical data.

Zephon promises the same samples in every global batch regardless of DP
degree, but not the same order within a step or the same rank assignment. So
each DP rank logs a hash of every packed sequence it trains on, tagged with
the global step, and the comparison treats each step as a multiset.

Framework-agnostic: the torchtitan loader and the Megatron entry point both
feed rows here as bytes.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable

ROW_HASH_BYTES = 8
SAMPLE_HASH_SUBDIR = "sample-hashes"


def row_hash(row: bytes) -> str:
    """Fingerprint of one packed sequence's token bytes."""
    return hashlib.blake2b(row, digest_size=ROW_HASH_BYTES).hexdigest()


def hash_file_name(launch_id: str, dp_world_size: int, global_rank: int) -> str:
    """One file per rank per launch; the launch ID prefix lets the latest launch win."""
    return f"{launch_id}-dp{dp_world_size}-rank{global_rank:03d}.jsonl"


class StepHashRecorder:
    """Accumulates row hashes and writes one JSONL record per optimizer step.

    The microbatch count is global (every rank advances it by the DP degree
    per local microbatch), so it means the same thing after a DP change.
    """

    def __init__(
        self,
        path: str,
        *,
        dp_world_size: int,
        dp_rank: int,
        microbatches_per_step: int,
        consumed_microbatches: int = 0,
    ) -> None:
        if microbatches_per_step % dp_world_size:
            raise ValueError(f"microbatches_per_step={microbatches_per_step} not divisible by DP={dp_world_size}")
        self._path = path
        self._dp_world_size = dp_world_size
        self._dp_rank = dp_rank
        self._microbatches_per_step = microbatches_per_step
        self._step_hashes: list[str] = []
        self.consumed_microbatches = consumed_microbatches
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    @property
    def consumed_microbatches(self) -> int:
        return self._consumed_microbatches

    @consumed_microbatches.setter
    def consumed_microbatches(self, consumed: int) -> None:
        if consumed % self._microbatches_per_step:
            raise ValueError(f"Checkpoint was taken mid-step: {consumed} microbatches consumed")
        self._consumed_microbatches = consumed
        self._step_hashes = []

    def record(self, rows: Iterable[bytes]) -> None:
        """Record one local microbatch's packed rows."""
        self._step_hashes.extend(row_hash(row) for row in rows)
        self._consumed_microbatches += self._dp_world_size
        if self._consumed_microbatches % self._microbatches_per_step == 0:
            record = {
                "step": self._consumed_microbatches // self._microbatches_per_step,
                "dp": self._dp_world_size,
                "dp_rank": self._dp_rank,
                "row_hashes": self._step_hashes,
            }
            # Local disk, opened per step: appending to an open file on a
            # Modal Volume duplicated its contents on background commits.
            with open(self._path, "a") as hash_file:
                hash_file.write(json.dumps(record) + "\n")
            self._step_hashes = []
