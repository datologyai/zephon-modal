# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Stage FineWeb-Edu sample shards into the data Volume and index them.

The Zephon index.json lets every training rank discover shards from one file
instead of reading each Parquet footer at startup.

Each sample-10BT shard is ~2 GB / ~0.7B tokens. Two shards are what the
example's references were trained on; stage more for a longer run (a run's
data can't change under it, so decide before starting).

    modal run examples/torchtitan/stage_data.py --num-shards 2
"""

import os
import shutil

import modal
from storage import DATA_MOUNT, FINEWEB_EDU_DIR, FINEWEB_EDU_REPO, FINEWEB_EDU_SAMPLE_PREFIX, data_volume

STAGING_TIMEOUT_SECONDS = 60 * 60

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("huggingface_hub", "zephon[parquet]>=0.1.0,<0.2")
    .add_local_python_source("storage")
)
app = modal.App(
    f"{os.environ.get('ZEPHON_MODAL_TORCHTITAN_PREFIX', 'zephon-modal-torchtitan')}-stage-data", image=image
)


@app.function(volumes={DATA_MOUNT: data_volume}, timeout=STAGING_TIMEOUT_SECONDS, cpu=8.0)
def stage_fineweb_edu(num_shards: int) -> list[str]:
    """Download the first ``num_shards`` sample-10BT shards (skipping ones already staged) and index them."""
    from huggingface_hub import HfApi, hf_hub_download

    shards = sorted(
        path
        for path in HfApi().list_repo_files(FINEWEB_EDU_REPO, repo_type="dataset")
        if path.startswith(FINEWEB_EDU_SAMPLE_PREFIX) and path.endswith(".parquet")
    )[:num_shards]
    os.makedirs(FINEWEB_EDU_DIR, exist_ok=True)
    staged = []
    for shard in shards:
        target = os.path.join(FINEWEB_EDU_DIR, os.path.basename(shard))
        if not os.path.exists(target):
            downloaded = hf_hub_download(FINEWEB_EDU_REPO, shard, repo_type="dataset", local_dir="/tmp/hf")
            # /tmp and the Volume are different filesystems, so no rename.
            shutil.move(downloaded, target)
        staged.append(f"{target} ({os.path.getsize(target) / 1e9:.2f} GB)")
    from zephon.build_index import build_index

    staged.append(f"index: {build_index('parquet', FINEWEB_EDU_DIR)}")
    data_volume.commit()
    return staged


@app.local_entrypoint()
def main(num_shards: int = 2) -> None:
    """Stage the shards and print what's on the Volume."""
    for line in stage_fineweb_edu.remote(num_shards):
        print(line)
