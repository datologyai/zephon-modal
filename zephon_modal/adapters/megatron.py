# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Megatron-LM (Megatron-LM-zephon), launched through ``pretrain_gpt_zephon.py``."""

from __future__ import annotations

from dataclasses import dataclass

from zephon_modal.adapters.base import Adapter, resolve
from zephon_modal.context import LaunchContext


@dataclass(frozen=True, kw_only=True)
class Megatron(Adapter):
    """Runs a Megatron pretraining script with per-step sample hashes.

    Attributes:
        args: ``"module:function"`` importable in the training image. It's
            called with the :class:`LaunchContext` inside the container and
            returns the script's arguments (it may also write files they
            name, such as a Zephon TOML data recipe). Everything but
            ``--train-iters`` must stay fixed across resumes, and
            ``--zephon-aggregate-store`` should be ``ctx.aggregate_store``.
        script: Pretraining script, relative to ``megatron_dir``.
        megatron_dir: Megatron-LM-zephon checkout in the image.
    """

    args: str
    script: str = "pretrain_gpt_zephon.py"
    megatron_dir: str = "/opt/megatron-lm"

    def workdir(self) -> str:
        return self.megatron_dir

    def command(self, ctx: LaunchContext) -> list[str]:
        return ["-m", "zephon_modal.adapters.megatron_entry", self.script, *resolve(self.args)(ctx)]

    def local_ranks_filter(self, nproc_per_node: int) -> str:
        # Megatron logs loss and throughput from the last rank.
        return f"0,{nproc_per_node - 1}"
