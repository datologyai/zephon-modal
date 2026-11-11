# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Named storage and dataset paths for the Megatron example."""

import os

import modal

DATA_MOUNT = "/data"
DATA_VOLUME_NAME = os.environ.get("ZEPHON_MODAL_MEGATRON_DATA_VOLUME", "zephon-modal-megatron-data")
RUNS_VOLUME_NAME = os.environ.get("ZEPHON_MODAL_MEGATRON_RUNS_VOLUME", "zephon-modal-megatron-runs")
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
runs_volume = modal.Volume.from_name(RUNS_VOLUME_NAME, create_if_missing=True, version=2)

FINEWEB_EDU_REPO = "HuggingFaceFW/fineweb-edu"
FINEWEB_EDU_SAMPLE_PREFIX = "sample/10BT/"
FINEWEB_EDU_DIR = f"{DATA_MOUNT}/fineweb-edu/sample-10BT"
SMOKE_DATA_DIR = f"{DATA_MOUNT}/smoke"
