# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Telling finished checkpoints from ones a preemption cut short.

A container can be killed mid-save, and on a multi-node placement each node
commits its own shards to the Volume, so a checkpoint whose metadata made it
can still be missing another node's files. Before a launch resumes, node 0
moves every incomplete checkpoint aside so the trainer falls back to the
latest one it can actually load.
"""

from __future__ import annotations

import os
import pickle
import re
from dataclasses import dataclass, field

DISCARDED_SUBDIR = "discarded-checkpoints"


def dcp_complete(path: str) -> bool:
    """Whether a torch.distributed.checkpoint directory has every file its ``.metadata`` lists, at full length."""
    if not os.path.isfile(os.path.join(path, ".metadata")):
        return False
    from torch.distributed.checkpoint.filesystem import FileSystemReader

    try:
        metadata = FileSystemReader(path).read_metadata()
    except (EOFError, pickle.UnpicklingError):
        # Truncated by a kill. Anything else (e.g. a class that won't import)
        # raises: discarding good checkpoints on a reader problem would be worse.
        return False
    # Every shard's (file, offset, length): a file truncated by a kill is too short.
    needed: dict[str, int] = {}
    for info in (metadata.storage_data or {}).values():
        needed[info.relative_path] = max(needed.get(info.relative_path, 0), info.offset + info.length)
    for relative_path, size in needed.items():
        file_path = os.path.join(path, relative_path)
        if not os.path.isfile(file_path) or os.path.getsize(file_path) < size:
            return False
    return True


@dataclass(frozen=True)
class CheckpointLayout:
    """Where a trainer keeps its DCP checkpoints under the run directory.

    Attributes:
        subdir: Checkpoint directory under ``ctx.run_dir``.
        step_pattern: Regex matching one checkpoint's directory name; group 1 is the step.
        tracker: A file naming the latest checkpoint's step, which the trainer
            loads instead of scanning (Megatron's ``latest_checkpointed_iteration.txt``).
    """

    subdir: str = "checkpoint"
    step_pattern: str = r"step-(\d+)"
    tracker: str | None = None

    def steps(self, run_dir: str) -> dict[int, str]:
        """Every checkpoint directory present, complete or not, by step."""
        checkpoint_dir = os.path.join(run_dir, self.subdir)
        if not os.path.isdir(checkpoint_dir):
            return {}
        return {
            int(match.group(1)): os.path.join(checkpoint_dir, name)
            for name in os.listdir(checkpoint_dir)
            if (match := re.fullmatch(self.step_pattern, name)) and os.path.isdir(os.path.join(checkpoint_dir, name))
        }

    def latest_complete(self, run_dir: str) -> int | None:
        """The latest step a resume can load."""
        return max((step for step, path in self.steps(run_dir).items() if dcp_complete(path)), default=None)


@dataclass(frozen=True)
class CheckpointCleanup:
    """What :func:`discard_incomplete` found."""

    latest: int | None
    discarded: list[int] = field(default_factory=list)


def discard_incomplete(layout: CheckpointLayout, run_dir: str, tag: str) -> CheckpointCleanup:
    """Move incomplete checkpoints to ``<run_dir>/discarded-checkpoints`` and point the tracker at the latest complete one.

    Moved, not deleted, so a wrong verdict loses nothing. ``tag`` (e.g. the
    container ID) keeps names unique across attempts.
    """
    discarded = []
    for step, path in sorted(layout.steps(run_dir).items()):
        if not dcp_complete(path):
            target_dir = os.path.join(run_dir, DISCARDED_SUBDIR)
            os.makedirs(target_dir, exist_ok=True)
            os.rename(path, os.path.join(target_dir, f"{os.path.basename(path)}-{tag}"))
            discarded.append(step)
    latest = layout.latest_complete(run_dir)
    if layout.tracker is not None and discarded:
        tracker = os.path.join(run_dir, layout.subdir, layout.tracker)
        if latest is None:
            if os.path.exists(tracker):
                os.remove(tracker)
        else:
            with open(tracker, "w") as tracker_file:
                tracker_file.write(str(latest))
    return CheckpointCleanup(latest=latest, discarded=discarded)
