# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Megatron training image with its CUDA stack supplied by NVIDIA's PyTorch image."""

import os

import modal

MEGATRON_REPO = "https://github.com/datologyai/Megatron-LM-zephon"
MEGATRON_COMMIT = os.environ.get("ZEPHON_MODAL_MEGATRON_COMMIT", "7ce36bd201921ff78f798eb362d0aca97bdd6869")
MEGATRON_DIR = "/opt/megatron-lm"
# Match the checkout's docker/.ngc_version.dev; preserve NGC's coupled
# PyTorch, Transformer Engine, Apex and FlashAttention installations.
BASE_IMAGE = os.environ.get("ZEPHON_MODAL_MEGATRON_BASE_IMAGE", "nvcr.io/nvidia/pytorch:26.09-py3")
ZEPHON_SPEC = "zephon[hf,cloud,parquet]==0.2.1"
HF_ASSETS_DIR = "/assets/hf"

training_image = (
    modal.Image.from_registry(BASE_IMAGE)
    .apt_install("git", "libibverbs1", "ibverbs-providers", "rdma-core")
    .run_commands(
        f"git clone {MEGATRON_REPO} {MEGATRON_DIR}",
        f"git -C {MEGATRON_DIR} checkout --detach {MEGATRON_COMMIT}",
        f"python -m pip install -r {MEGATRON_DIR}/requirements-zephon.txt",
    )
    # NGC's PIP_CONSTRAINT keeps its compatible torch build in place.
    .pip_install(ZEPHON_SPEC, "einops", "tensorstore", "huggingface_hub")
    .run_commands(
        'python -c "from huggingface_hub import snapshot_download; '
        "snapshot_download('Qwen/Qwen3-0.6B', "
        "allow_patterns=['tokenizer.json', 'tokenizer_config.json', 'config.json', 'generation_config.json'], "
        f"local_dir='{HF_ASSETS_DIR}/Qwen3-0.6B')\""
    )
    .env(
        {
            "PYTHONPATH": f"/root:{MEGATRON_DIR}",
            "HF_ASSETS_DIR": HF_ASSETS_DIR,
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        }
    )
    .add_local_python_source("recipe", "storage")
)
