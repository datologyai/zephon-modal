# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""A Megatron pretraining script, plus per-step sample hashes.

Run under torchrun with the Megatron checkout as the working directory::

    torchrun ... -m zephon_modal.zephon.megatron_entry pretrain_gpt_zephon.py <script args>

which is what ``Megatron(entry="zephon_modal.zephon.megatron_entry",
script="pretrain_gpt_zephon.py", ...)`` runs.

It wraps the Zephon dataset provider so every DP rank logs a hash of each
packed sequence it trains on, in the same format as the torchtitan loader,
so ``zephon_modal compare`` works on Megatron runs unchanged.
"""

from __future__ import annotations

import os
import runpy
import sys
from collections.abc import Sequence
from typing import Any

import torch.distributed as dist

from zephon_modal.context import context
from zephon_modal.verify.hashes import StepHashRecorder, hash_file_name, pack_fields
from zephon_modal.zephon import validate_step_size


class SampleHashingIterator:
    """Iterator over a MegatronZephonDataLoader that records per-step sequence hashes.

    Everything else (save_state, restore_state) is forwarded to the wrapped
    loader, so Megatron's dataloader checkpointing sees the Zephon loader.
    """

    def __init__(self, loader: Any, hashes: StepHashRecorder) -> None:
        self._loader = loader
        self._hashes = hashes

    def __iter__(self) -> SampleHashingIterator:
        return self

    def __next__(self) -> dict:
        batch = next(self._loader)
        names = ("tokens", "labels", "loss_mask", "position_ids")
        if any(name not in batch for name in names):
            raise ValueError("Training batch lacks fields required for complete sample hashing")
        self._hashes.record(
            pack_fields({name + ":" + str(batch[name].dtype): batch[name][index].numpy().tobytes() for name in names})
            for index in range(len(batch["tokens"]))
        )
        return batch

    def __getattr__(self, name: str) -> Any:
        return getattr(self._loader, name)


def _install_sample_hashing() -> None:
    from megatron.core import mpu
    from megatron.core.num_microbatches_calculator import get_num_microbatches
    from megatron.training import get_args
    from megatron.training.datasets import zephon_dataloader

    zephon_provider = zephon_dataloader.zephon_train_valid_test_datasets_provider

    def provider(train_valid_test_num_samples: Sequence[int], vp_stage: int | None = None):
        loader, valid, test = zephon_provider(train_valid_test_num_samples, vp_stage)
        if loader is None:
            return loader, valid, test
        ctx = context()
        dp_world_size = mpu.get_data_parallel_world_size()
        dp_rank = mpu.get_data_parallel_rank()
        microbatches_per_step = get_num_microbatches() * dp_world_size
        validate_step_size(microbatches_per_step, ctx)
        hashes = StepHashRecorder(
            os.path.join(ctx.sample_hash_dir, hash_file_name(ctx.launch_id, dp_world_size, dist.get_rank())),
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
            microbatches_per_step=microbatches_per_step,
            # Megatron restores the iteration from its own checkpoint, so the
            # global count needn't live in dataloader state.
            consumed_microbatches=get_args().iteration * microbatches_per_step,
            schema="megatron-batch-v2",
        )
        return SampleHashingIterator(loader, hashes), valid, test

    provider.is_distributed = zephon_provider.is_distributed  # type: ignore[attr-defined]
    # The script imports the provider by name when it runs, so it picks this up.
    zephon_dataloader.zephon_train_valid_test_datasets_provider = provider


def main() -> None:
    """Install sample hashing, then run the script named by the first argument."""
    script, *script_args = sys.argv[1:]
    sys.argv = [script, *script_args]
    _install_sample_hashing()
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
