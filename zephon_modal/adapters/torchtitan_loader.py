# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""A torchtitan-zephon data loader that logs per-step sample hashes.

Only importable inside a torchtitan image. Use it as a recipe's dataloader
config so ``zephon_modal compare`` can check the run.
"""

from __future__ import annotations

import logging
import os
import pickle
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import torch.distributed as dist
from torchtitan.components.data.types import TokenizedTrainingMicrobatch
from torchtitan_recipes.zephon import ZephonDataLoader

from zephon_modal.context import context
from zephon_modal.verify.hashes import StepHashRecorder, hash_file_name

logger = logging.getLogger(__name__)

CONSUMED_MICROBATCHES_KEY = "zephon_modal_consumed_microbatches"
# Checkpoints written by zephon-modal-demo, before this package existed.
LEGACY_CONSUMED_MICROBATCHES_KEY = "demo_consumed_microbatches"


class SampleHashingZephonDataLoader(ZephonDataLoader):
    """ZephonDataLoader that appends per-step sequence hashes to a JSONL file."""

    @dataclass(kw_only=True, slots=True)
    class Config(ZephonDataLoader.Config):
        """ZephonDataLoader's config plus the step size for hash records."""

        microbatches_per_step: int
        """Global microbatches per optimizer step, used to map microbatches to steps."""

    def __init__(
        self, config: Config, *, dp_world_size: int, dp_rank: int, max_context_length: int, **kwargs: Any
    ) -> None:
        super().__init__(
            config, dp_world_size=dp_world_size, dp_rank=dp_rank, max_context_length=max_context_length, **kwargs
        )
        ctx = context()
        global_rank = dist.get_rank() if dist.is_initialized() else 0
        self._dp_rank = dp_rank
        self._seq_len = max_context_length
        self._hashes = StepHashRecorder(
            os.path.join(ctx.sample_hash_dir, hash_file_name(ctx.launch_id, dp_world_size, global_rank)),
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
            microbatches_per_step=config.microbatches_per_step,
        )

    def __iter__(self) -> Iterator[TokenizedTrainingMicrobatch]:
        # Startup latency diagnosis: how long Zephon takes to hand over its
        # first microbatch, separately from torch.compile and CUDA graph setup.
        iter_start = time.perf_counter()
        first_microbatch = True
        for microbatch in super().__iter__():
            if first_microbatch:
                first_microbatch = False
                logger.warning(
                    "[zephon_modal] dp_rank %d: first Zephon microbatch after %.1fs",
                    self._dp_rank,
                    time.perf_counter() - iter_start,
                )
            rows = microbatch.input.reshape(-1, self._seq_len)
            self._hashes.record(row.numpy().tobytes() for row in rows)
            yield microbatch

    def state_dict(self) -> dict[str, bytes]:
        return super().state_dict() | {CONSUMED_MICROBATCHES_KEY: pickle.dumps(self._hashes.consumed_microbatches)}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        super().load_state_dict(state_dict)
        key = CONSUMED_MICROBATCHES_KEY if CONSUMED_MICROBATCHES_KEY in state_dict else LEGACY_CONSUMED_MICROBATCHES_KEY
        self._hashes.consumed_microbatches = pickle.loads(state_dict[key])
