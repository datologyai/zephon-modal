# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Elastic Megatron Qwen3-0.6B jobs; see README.md for setup and commands."""

import os

import modal
from image import MEGATRON_DIR, training_image
from storage import DATA_MOUNT, data_volume, runs_volume

from zephon_modal import ElasticJob, Megatron, Placement
from zephon_modal.checkpoints import CheckpointLayout
from zephon_modal.zephon import Zephon

PREFIX = os.environ.get("ZEPHON_MODAL_MEGATRON_PREFIX", "zephon-modal-megatron")
SAMPLE_HASHES = os.environ.get("ZEPHON_MODAL_MEGATRON_SAMPLE_HASHES", "0") == "1"
ALERT_SECRET = os.environ.get("ZEPHON_MODAL_MEGATRON_ALERT_SECRET")
HASHING_ENTRY = "zephon_modal.zephon.megatron_entry" if SAMPLE_HASHES else None


def _job(name: str, args: str, lanes: int, placements: list[Placement]) -> ElasticJob:
    return ElasticJob(
        name=name,
        image=training_image,
        adapter=Megatron(
            args=args,
            script="pretrain_gpt_zephon.py",
            entry=HASHING_ENTRY,
            megatron_dir=MEGATRON_DIR,
            checkpoints=CheckpointLayout(
                step_pattern=r"iter_(\d+)",
                tracker="latest_checkpointed_iteration.txt",
                required_torch_files=("mp_rank_00/train_dataloader_dprank000.pt",),
            ),
        ),
        data=Zephon(canonical_replicas=lanes),
        placements=placements,
        runs_volume=runs_volume,
        volumes={DATA_MOUNT: data_volume},
        alert_secret=modal.Secret.from_name(ALERT_SECRET) if ALERT_SECRET else None,
    )


job = _job(
    PREFIX,
    "recipe:qwen3_0p6b_zephon_args",
    16,
    [Placement("H100:8", nodes=2), Placement("H100:8", fallbacks=("H100:4",))],
)
smoke = _job(f"{PREFIX}-smoke", "recipe:smoke_args", 1, [Placement("H100")])
