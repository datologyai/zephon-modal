# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""The example's Modal Volumes and the paths they're mounted at.

Volume names come from the environment, so each team (or person) can use its
own; the defaults are neutral.
"""

import os

import modal

DATA_MOUNT = "/data"

# Staged once and then read-only, so a v1 Volume is fine.
data_volume = modal.Volume.from_name(
    os.environ.get("ZEPHON_MODAL_TORCHTITAN_DATA_VOLUME", "zephon-modal-torchtitan-data"), create_if_missing=True
)

# v2 lets every rank write its own DCP shard concurrently.
runs_volume = modal.Volume.from_name(
    os.environ.get("ZEPHON_MODAL_TORCHTITAN_RUNS_VOLUME", "zephon-modal-torchtitan-runs"),
    create_if_missing=True,
    version=2,
)

# torch.compile spends ~2.5 min autotuning flex attention on a cold cache, which
# every elastic relaunch would otherwise pay. Inductor writes cache entries via
# rename from many ranks at once, so this needs a v2 Volume.
compile_cache_volume = modal.Volume.from_name(
    os.environ.get("ZEPHON_MODAL_TORCHTITAN_COMPILE_CACHE_VOLUME", "zephon-modal-torchtitan-compile-cache"),
    create_if_missing=True,
    version=2,
)

FINEWEB_EDU_REPO = "HuggingFaceFW/fineweb-edu"
FINEWEB_EDU_SAMPLE_PREFIX = "sample/10BT/"
FINEWEB_EDU_DIR = f"{DATA_MOUNT}/fineweb-edu/sample-10BT"
