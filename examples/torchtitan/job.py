# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-0.6B on torchtitan-zephon: an elastic run across a 2-node H100 cluster and one H100 node.

See README.md in this directory. In short:

    export ZEPHON_MODAL_TORCHTITAN_PREFIX=<you>-tt   # deployed app names are shared across the workspace
    modal run examples/torchtitan/stage_data.py --num-shards 2   # once
    python -m zephon_modal placements examples/torchtitan/job.py
    python -m zephon_modal start examples/torchtitan/job.py --run-id <you>-qwen3 --steps 2000 --keep-checkpoints 3
    python -m zephon_modal status examples/torchtitan/job.py --run-id <you>-qwen3

Environment variables (read when this file is loaded, so set them for every command):

- ``ZEPHON_MODAL_TORCHTITAN_PREFIX``: the Modal app name; its registry is ``<prefix>-runs``.
- ``ZEPHON_MODAL_TORCHTITAN_DATA_VOLUME``, ``ZEPHON_MODAL_TORCHTITAN_RUNS_VOLUME``,
  ``ZEPHON_MODAL_TORCHTITAN_COMPILE_CACHE_VOLUME``: Volume names.
- ``ZEPHON_MODAL_TORCHTITAN_COMMIT``: the torchtitan-zephon commit to build.
- ``ZEPHON_MODAL_TORCHTITAN_ALERT_SECRET``: optional Modal secret providing
  ``ZEPHON_MODAL_ALERT_WEBHOOK``, for the reconciler's alerts.
"""

import os

import modal
from image import TORCHTITAN_DIR, training_image
from storage import DATA_MOUNT, compile_cache_volume, data_volume, runs_volume

from zephon_modal import ElasticJob, Placement, Torchtitan
from zephon_modal.zephon import Zephon

# Zephon lanes: a run can train or resume on any DP degree that divides this,
# and the recipe reads it back from the launch context.
CANONICAL_REPLICAS = 16

# Every DP degree here divides the 16 canonical replicas. H100:8 is quick to get and
# sometimes lands on H200s. The single-node placement is one Modal function asking
# for H100:8 or H100:4: Modal allocates whichever it can, and the launch sizes
# itself to what it got.
PLACEMENTS = [
    Placement("H100:8", nodes=2),
    Placement("H100:8", fallbacks=("H100:4",)),
]

ALERT_SECRET = os.environ.get("ZEPHON_MODAL_TORCHTITAN_ALERT_SECRET")

job = ElasticJob(
    name=os.environ.get("ZEPHON_MODAL_TORCHTITAN_PREFIX", "zephon-modal-torchtitan"),
    image=training_image,
    adapter=Torchtitan(module="recipe", config="qwen3_0p6b_zephon", torchtitan_dir=TORCHTITAN_DIR),
    data=Zephon(canonical_replicas=CANONICAL_REPLICAS),
    placements=PLACEMENTS,
    runs_volume=runs_volume,
    volumes={DATA_MOUNT: data_volume},
    # torch.compile spends ~2.5 min autotuning flex attention on a cold cache,
    # which every elastic relaunch would otherwise pay.
    compile_cache_volume=compile_cache_volume,
    alert_secret=modal.Secret.from_name(ALERT_SECRET) if ALERT_SECRET else None,
)
