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
import shutil
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
        required_torch_files: Additional ``torch.save`` files, relative to each
            checkpoint, that must be readable before a checkpoint can be used
            or count toward retention. Use only files loadable with
            ``weights_only=True`` (tensors and primitive state).
        torch_tracker: Optional pair of filenames: a root-level tracker and
            its checkpoint-local source (e.g. ``latest_train_state.pt`` and
            ``train_state.pt``). The source must be readable before a save is
            complete; startup restores the root tracker from the selected save.
    """

    subdir: str = "checkpoint"
    step_pattern: str = r"step-(\d+)"
    tracker: str | None = None
    required_torch_files: tuple[str, ...] = ()
    torch_tracker: tuple[str, str] | None = None

    def complete(self, path: str) -> bool:
        """Whether model shards and any separately saved state are complete."""
        if not dcp_complete(path):
            return False
        required = self.required_torch_files + ((self.torch_tracker[1],) if self.torch_tracker else ())
        if required:
            import torch

            try:
                for relative_path in required:
                    torch.load(os.path.join(path, relative_path), map_location="cpu", weights_only=True)
            except (FileNotFoundError, EOFError, pickle.UnpicklingError, RuntimeError):
                # The model write can finish before a separate state write
                # starts, or a preemption can truncate that state file.
                return False
        return True

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
        """The latest step a resume can load.

        Checks from the newest checkpoint down and stops at the first complete
        one, so the cost doesn't grow with a long run's checkpoint count: a
        checkpoint that was complete once stays complete.
        """
        return next(iter(self.newest_complete(run_dir, 1)), None)

    def newest_complete(self, run_dir: str, count: int) -> list[int]:
        """Up to ``count`` complete steps, newest first, checking only as far back as needed."""
        found = []
        for step, path in sorted(self.steps(run_dir).items(), reverse=True):
            if len(found) == count:
                break
            if self.complete(path):
                found.append(step)
        return found


@dataclass(frozen=True)
class CheckpointCleanup:
    """What :func:`discard_incomplete` found."""

    latest: int | None
    discarded: list[int] = field(default_factory=list)


def discard_incomplete(layout: CheckpointLayout, run_dir: str, tag: str) -> CheckpointCleanup:
    """Move incomplete checkpoints to ``<run_dir>/discarded-checkpoints`` and point the tracker at the latest complete one.

    Moved, not deleted, so a wrong verdict loses nothing. ``tag`` (e.g. the
    container ID) keeps names unique across attempts. Only checkpoints newer
    than the latest complete one are looked at: an older incomplete one is
    never loaded, and checking every checkpoint would make each launch slower
    the longer the run.
    """
    latest = layout.latest_complete(run_dir)
    discarded = []
    for step, path in sorted(layout.steps(run_dir).items()):
        if latest is None or step > latest:
            target_dir = os.path.join(run_dir, DISCARDED_SUBDIR)
            os.makedirs(target_dir, exist_ok=True)
            os.rename(path, os.path.join(target_dir, f"{os.path.basename(path)}-{tag}"))
            discarded.append(step)
    if layout.torch_tracker is not None:
        root_name, local_name = layout.torch_tracker
        tracker = os.path.join(run_dir, layout.subdir, root_name)
        if latest is None:
            if os.path.exists(tracker):
                os.remove(tracker)
        else:
            # The trainer may prefer this file to its text tracker. Copy the
            # selected checkpoint's exact state, including when the previous
            # root tracker lagged a completed save at the time of preemption.
            source = os.path.join(layout.steps(run_dir)[latest], local_name)
            shutil.copyfile(source, tracker + ".tmp")
            os.replace(tracker + ".tmp", tracker)
    if layout.tracker is not None and (discarded or layout.torch_tracker is not None):
        tracker = os.path.join(run_dir, layout.subdir, layout.tracker)
        if latest is None:
            if os.path.exists(tracker):
                os.remove(tracker)
        else:
            with open(tracker, "w") as tracker_file:
                tracker_file.write(str(latest))
    return CheckpointCleanup(latest=latest, discarded=discarded)


def prune(layout: CheckpointLayout, run_dir: str, keep: int) -> list[int]:
    """Delete checkpoints older than the ``keep`` latest complete ones; returns the steps deleted.

    Trainers should keep every checkpoint: on a multi-node launch each node
    sees only its own uncommitted Volume files, so a trainer pruning there
    leaves other nodes' shards behind as partial checkpoints. The reconciler
    prunes instead, from its own committed view of the Volume. The latest
    complete checkpoint, which any resume loads, is always kept, and so is
    anything newer than it (a save that may still be in progress).
    """
    if keep < 1:
        raise ValueError(f"keep must be at least 1, got {keep}")
    kept = layout.newest_complete(run_dir, keep)
    if len(kept) < keep:
        return []
    deleted = []
    for step, path in sorted(layout.steps(run_dir).items()):
        if step < kept[-1]:
            shutil.rmtree(path, ignore_errors=True)
            deleted.append(step)
    # Set-aside partial saves older than anything kept are no use either.
    discarded_dir = os.path.join(run_dir, DISCARDED_SUBDIR)
    if os.path.isdir(discarded_dir):
        for name in os.listdir(discarded_dir):
            match = re.match(layout.step_pattern, name)
            if match and int(match.group(1)) < kept[-1]:
                shutil.rmtree(os.path.join(discarded_dir, name), ignore_errors=True)
    return deleted
