# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""The torchtitan recipe: Qwen3-0.6B on FineWeb-Edu sample-10BT through Zephon.

Loaded inside the training container via
``torchtitan.train --module recipe --config qwen3_0p6b_zephon`` (see
``job.py``). Per-launch values (run ID and directory, step target, seed,
Zephon settings) come from ``zephon_modal.context()`` and
``zephon_modal.zephon.settings()``, so the same recipe resumes unchanged on
any placement. Only the training image can import it.
"""

import os

from storage import FINEWEB_EDU_DIR
from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import AdamW, LRSchedulersContainer, Optim, OptimizersContainer
from torchtitan.config import TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.qwen3 import build_model_config
from torchtitan.trainer import Trainer
from torchtitan_recipes.zephon import ZephonSource

from zephon_modal import context
from zephon_modal.zephon import settings
from zephon_modal.zephon.torchtitan import SampleHashingZephonDataLoader

SEQ_LEN = 4096
SEQUENCES_PER_MICROBATCH = 2
# 64 microbatches per optimizer step: 4 lane windows, so grad accumulation is
# 16 / 8 / 4 on DP 4 / 8 / 16 and every step consumes whole windows.
SEQUENCES_PER_STEP = 128
MICROBATCHES_PER_STEP = SEQUENCES_PER_STEP // SEQUENCES_PER_MICROBATCH
# Left unset, the adapter takes each op's declared parallelism (fetch 4,
# tokenize 4, pack 1) on every rank; with 8 ranks per node the first batch took
# 125-148s. Benchmarked with 8 ranks on 48 CPUs, tokenize sets the ceiling:
# tokenize=2 capped at ~140k tokens/s per rank whatever fetch and pack were,
# 3 reached ~187k, and 4 added nothing but ~16s of startup. Training needs ~63k.
ZEPHON_PARALLELISM = {"fetch": 2, "tokenize": 3, "pack": 1}

CHECKPOINT_INTERVAL = 10
WARMUP_STEPS = 10
LR = 3e-4


def _fineweb_edu_dataloader() -> SampleHashingZephonDataLoader.Config:
    ctx = context()
    zephon = settings(ctx)
    return SampleHashingZephonDataLoader.Config(
        sources=[ZephonSource(name="fineweb_edu", path=FINEWEB_EDU_DIR, fmt="parquet")],
        text_field="text",
        seed=ctx.seed,
        canonical_replicas=zephon.canonical_replicas,
        # Qwen3's tokenizer has no BOS token.
        tokenize_special_tokens="eos",
        # Token-aware allocation only balances token shares between sources;
        # with a single source it's just a calibration pass at startup.
        token_estimation=False,
        **{f"{op}_parallelism": workers for op, workers in ZEPHON_PARALLELISM.items()},
        # Lane state merges through torchrun's TCPStore ("torch://") on every launch.
        aggregate_store=zephon.aggregate_store,
        run_id=ctx.run_id,
        # The loader rejects a step that isn't whole lane windows.
        microbatches_per_step=MICROBATCHES_PER_STEP,
    )


def qwen3_0p6b_zephon() -> Trainer.Config:
    """Qwen3-0.6B on a FineWeb-Edu sample, checkpointing every 10 steps."""
    parallelism = ParallelismConfig()
    model_config = build_model_config("0.6B", seq_len=SEQ_LEN, parallelism=parallelism)
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(global_vocab_size=decoder_vocab_size(model_config)),
        ),
        hf_assets_path=f"{os.environ['HF_ASSETS_DIR']}/Qwen3-0.6B",
        model=model_config,
        dataloader=_fineweb_edu_dataloader(),
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(optimizers=[AdamW.Config(pattern=r".*", lr=LR)]),
            lr_scheduler=LRSchedulersContainer.Config(warmup_steps=WARMUP_STEPS),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=SEQUENCES_PER_MICROBATCH * SEQ_LEN,
            num_tokens_per_train_step=SEQUENCES_PER_STEP * SEQ_LEN,
            max_context_length=SEQ_LEN,
            steps=context().steps,
        ),
        parallelism=parallelism,
        checkpointer=CheckpointManager.Config(
            interval=CHECKPOINT_INTERVAL,
            # Keep everything: on a cluster each node sees only its own files, so
            # pruning there leaves partial checkpoints. `start --keep-checkpoints` prunes.
            keep_latest_k=0,
            # Resuming needs optimizer and Zephon state from the final step,
            # not a model-only export.
            last_save_model_only=False,
        ),
    )
