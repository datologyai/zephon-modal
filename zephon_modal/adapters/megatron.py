# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Megatron-LM, launched through one of its pretraining scripts."""

from __future__ import annotations

from dataclasses import dataclass

from zephon_modal.adapters.base import Adapter, resolve
from zephon_modal.checkpoints import CheckpointLayout
from zephon_modal.context import LaunchContext


@dataclass(frozen=True, kw_only=True)
class Megatron(Adapter):
    """Runs a Megatron pretraining script under torchrun.

    Attributes:
        args: ``"module:function"`` importable in the training image. It's
            called with the :class:`LaunchContext` inside the container and
            returns the script's arguments (it may also write files they
            name, such as a data recipe). Everything but ``--train-iters``
            must stay fixed across resumes.
        script: Pretraining script, relative to ``megatron_dir``.
        entry: Optional module that wraps the script: run as
            ``-m <entry> <script> <args>``. For Zephon runs with per-step
            sample hashes, use ``"zephon_modal.zephon.megatron_entry"`` with
            ``script="pretrain_gpt_zephon.py"``.
        megatron_dir: Megatron checkout in the image.
        checkpoint_on_shutdown: Add ``--exit-signal-handler``, so a stopped
            run finishes its iteration and saves before exiting. Only for
            small checkpoints: the save must fit in ``shutdown_timeout`` and
            leave time to commit it (Qwen3-0.6B with optimizer state, ~8 GB,
            took 37 s on 4 H100s). A save that doesn't finish is set aside on
            the next resume.

    The default ``checkpoints`` layout assumes ``--save`` is
    ``<run_dir>/checkpoint`` with the ``torch_dist`` format.
    """

    args: str
    script: str = "pretrain_gpt.py"
    entry: str | None = None
    megatron_dir: str = "/opt/megatron-lm"
    checkpoint_on_shutdown: bool = False
    checkpoints: CheckpointLayout | None = CheckpointLayout(
        subdir="checkpoint", step_pattern=r"iter_(\d+)", tracker="latest_checkpointed_iteration.txt"
    )

    def workdir(self) -> str:
        return self.megatron_dir

    def command(self, ctx: LaunchContext) -> list[str]:
        script_args = [*resolve(self.args)(ctx), *(["--exit-signal-handler"] if self.checkpoint_on_shutdown else [])]
        if self.entry is None:
            return [self.script, *script_args]
        return ["-m", self.entry, self.script, *script_args]

    def local_ranks_filter(self, nproc_per_node: int) -> str:
        # Megatron logs loss and throughput from the last rank.
        return f"0,{nproc_per_node - 1}"
