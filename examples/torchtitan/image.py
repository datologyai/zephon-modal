# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""The torchtitan training image: CUDA 13.0, nightly torch, torchtitan-zephon at a pinned commit, Zephon 0.2.1."""

import os

import modal

# Modal's H100 hosts run driver 580.95.05 / CUDA 13.0, and the container's CUDA runtime must not be newer than the
# host driver, so stay on cu130.
CUDA_BASE_IMAGE = "nvidia/cuda:13.0.3-devel-ubuntu24.04"
TORCH_INDEX_URL = "https://download.pytorch.org/whl/nightly/cu130"
PYTHON_VERSION = "3.12"

# libibverbs is required for RDMA between cluster nodes.
RDMA_APT_PACKAGES = ["libibverbs1", "ibverbs-providers", "rdma-core"]

# torchtitan-zephon, cloned at a pinned commit on main and run from source like
# its run_train.sh does. Set ZEPHON_MODAL_TORCHTITAN_COMMIT to try another
# commit (a full SHA); the clone is rebuilt whenever it changes.
TORCHTITAN_REPO = "https://github.com/datologyai/torchtitan-zephon"
TORCHTITAN_COMMIT = os.environ.get("ZEPHON_MODAL_TORCHTITAN_COMMIT", "1fb11e4eaecb852f0ceb1efec46101ef6f05aaeb")
TORCHTITAN_DIR = "/opt/torchtitan"

# Zephon from PyPI. Checkpoint aggregation through torchrun's TCPStore
# (aggregate_store="torch://") is in every release since 0.2.0; torchtitan-zephon's
# requirements ask for >=0.2.0,<0.3. Pinned exactly, and installed after them,
# so every image runs the same Zephon.
ZEPHON_SPEC = "zephon[hf,cloud,parquet]==0.2.1"

HF_ASSETS_DIR = "/assets/hf"
QWEN3_TOKENIZER_REPOS = ["Qwen/Qwen3-0.6B"]
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "config.json", "generation_config.json"]

DOWNLOAD_TOKENIZERS_COMMANDS = [
    f'python -c "from huggingface_hub import snapshot_download; '
    f"snapshot_download('{repo}', allow_patterns={TOKENIZER_FILES!r}, "
    f"local_dir='{HF_ASSETS_DIR}/{repo.split('/')[-1]}')\""
    for repo in QWEN3_TOKENIZER_REPOS
]

training_image = (
    modal.Image.from_registry(CUDA_BASE_IMAGE, add_python=PYTHON_VERSION)
    .apt_install("git", *RDMA_APT_PACKAGES)
    .run_commands(
        f"git clone {TORCHTITAN_REPO} {TORCHTITAN_DIR}",
        f"git -C {TORCHTITAN_DIR} checkout --detach {TORCHTITAN_COMMIT}",
    )
    # The checkout owns its dependencies, including its compatible torch_checkpointing pin.
    .pip_install("uv")
    .run_commands(
        'uv pip install --python "$(command -v python)" --compile-bytecode '
        f"-r {TORCHTITAN_DIR}/requirements.txt -r {TORCHTITAN_DIR}/requirements-zephon.txt"
    )
    .uv_pip_install(ZEPHON_SPEC, "huggingface_hub")
    # Following torchtitan-zephon's CI: the nightly torch goes in last so no
    # requirement can downgrade it.
    .uv_pip_install("torch", pre=True, index_url=TORCH_INDEX_URL, extra_options="--reinstall-package torch")
    .run_commands(*DOWNLOAD_TOKENIZERS_COMMANDS)
    .env(
        {
            "PYTHONPATH": f"/root:{TORCHTITAN_DIR}",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "HF_ASSETS_DIR": HF_ASSETS_DIR,
        }
    )
    # The recipe and the paths it reads; ElasticJob adds zephon_modal itself.
    .add_local_python_source("recipe", "storage")
)
