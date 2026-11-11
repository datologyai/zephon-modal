# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-0.6B pretraining arguments and Zephon data recipes for Megatron."""

import json
import os
from pathlib import Path

from storage import FINEWEB_EDU_DIR, SMOKE_DATA_DIR

from zephon_modal import LaunchContext
from zephon_modal.zephon import settings

SEQ_LEN = 4096
SEQUENCES_PER_MICROBATCH = 2
SEQUENCES_PER_STEP = 128
CHECKPOINT_INTERVAL = 10
DATA_RECIPE_PATH = "/tmp/zephon-data.toml"

# Qwen3-0.6B architecture, trained from scratch with its Hugging Face tokenizer.
MODEL_ARGS = [
    "--use-mcore-models",
    "--num-layers=28",
    "--hidden-size=1024",
    "--ffn-hidden-size=3072",
    "--num-attention-heads=16",
    "--group-query-attention",
    "--num-query-groups=8",
    "--kv-channels=128",
    "--qk-layernorm",
    "--normalization=RMSNorm",
    "--swiglu",
    "--disable-bias-linear",
    "--position-embedding-type=rope",
    "--rotary-percent=1.0",
    "--rotary-base=1000000",
    "--max-position-embeddings=40960",
    "--make-vocab-size-divisible-by=1187",
    "--attention-dropout=0.0",
    "--hidden-dropout=0.0",
    "--init-method-std=0.02",
    "--bf16",
]


def _data_recipe(data_dir: str, seed: int, *, smoke: bool) -> str:
    recipe = f"""\
text_field = "text"
seed = {seed}
tokenize_special_tokens = "eos"
token_estimation = false
"""
    if smoke:
        recipe += """\
chunk_size = 4
shuffle_buffer_size = 16
fetch_parallelism = 1
tokenize_parallelism = 1
pack_parallelism = 1
runner = "inline"
mtp_mode = false
"""
    return (
        recipe
        + f"""
[[sources]]
name = "text"
path = {json.dumps(data_dir)}
fmt = "parquet"
"""
    )


def _args(ctx: LaunchContext, *, smoke: bool) -> list[str]:
    zephon = settings(ctx)
    Path(DATA_RECIPE_PATH).write_text(_data_recipe(SMOKE_DATA_DIR if smoke else FINEWEB_EDU_DIR, ctx.seed, smoke=smoke))
    return [
        *MODEL_ARGS,
        f"--seq-length={128 if smoke else SEQ_LEN}",
        "--tokenizer-type=HuggingFaceTokenizer",
        f"--tokenizer-model={os.environ['HF_ASSETS_DIR']}/Qwen3-0.6B",
        f"--micro-batch-size={SEQUENCES_PER_MICROBATCH}",
        f"--global-batch-size={2 if smoke else SEQUENCES_PER_STEP}",
        f"--train-iters={ctx.steps}",
        f"--seed={ctx.seed}",
        "--lr=0.0003",
        f"--lr-warmup-iters={1 if smoke else 10}",
        # Extending the run's step target must not change the LR schedule.
        "--lr-decay-style=constant",
        "--lr-decay-iters=1000000",
        "--override-opt-param-scheduler",
        "--weight-decay=0.1",
        "--adam-beta1=0.9",
        "--adam-beta2=0.95",
        "--clip-grad=1.0",
        "--tensor-model-parallel-size=1",
        "--pipeline-model-parallel-size=1",
        "--context-parallel-size=1",
        "--no-create-attention-mask-in-dataloader",
        "--reset-position-ids",
        "--reset-attention-mask",
        "--dataloader-inter-document-masking",
        f"--zephon-data-config={DATA_RECIPE_PATH}",
        f"--zephon-canonical-replicas={zephon.canonical_replicas}",
        f"--zephon-aggregate-store={zephon.aggregate_store}",
        f"--zephon-run-id={ctx.run_id}",
        # Keep data state inside the same iteration directory so completeness
        # checks and retention cover the entire resumable checkpoint.
        f"--dataloader-save={ctx.run_dir}/checkpoint",
        f"--save={ctx.run_dir}/checkpoint",
        f"--load={ctx.run_dir}/checkpoint",
        f"--save-interval={2 if smoke else CHECKPOINT_INTERVAL}",
        "--ckpt-format=torch_dist",
        "--eval-iters=0",
        "--eval-interval=1000000",
        "--log-interval=1",
        "--log-throughput",
        "--distributed-timeout-minutes=30",
    ]


def qwen3_0p6b_zephon_args(ctx: LaunchContext) -> list[str]:
    """Build the FineWeb-Edu training arguments from the current launch."""
    return _args(ctx, smoke=False)


def smoke_args(ctx: LaunchContext) -> list[str]:
    """Build a one-GPU, 128-token version of the same model for a short check."""
    return _args(ctx, smoke=True)
