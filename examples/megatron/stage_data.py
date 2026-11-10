# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Stage FineWeb-Edu shards, or generate a tiny Parquet fixture for the smoke job."""

import os
import shutil
from pathlib import Path

import modal
from storage import (
    DATA_MOUNT,
    DATA_VOLUME_NAME,
    FINEWEB_EDU_DIR,
    FINEWEB_EDU_REPO,
    FINEWEB_EDU_SAMPLE_PREFIX,
    SMOKE_DATA_DIR,
    data_volume,
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("huggingface_hub", "zephon[parquet]==0.2.1", "pyarrow<26")
    .env({"ZEPHON_MODAL_MEGATRON_DATA_VOLUME": DATA_VOLUME_NAME})
    .add_local_python_source("storage")
)
app = modal.App(f"{os.environ.get('ZEPHON_MODAL_MEGATRON_PREFIX', 'zephon-modal-megatron')}-stage-data", image=image)


@app.function(volumes={DATA_MOUNT: data_volume}, timeout=3600, cpu=8.0)
def stage_fineweb_edu(num_shards: int) -> list[str]:
    """Download and index the first requested FineWeb-Edu sample shards."""
    from huggingface_hub import HfApi, hf_hub_download
    from zephon.build_index import build_index

    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    shards = sorted(
        path
        for path in HfApi().list_repo_files(FINEWEB_EDU_REPO, repo_type="dataset")
        if path.startswith(FINEWEB_EDU_SAMPLE_PREFIX) and path.endswith(".parquet")
    )[:num_shards]
    Path(FINEWEB_EDU_DIR).mkdir(parents=True, exist_ok=True)
    staged = []
    for shard in shards:
        target = os.path.join(FINEWEB_EDU_DIR, os.path.basename(shard))
        if not os.path.exists(target):
            downloaded = hf_hub_download(FINEWEB_EDU_REPO, shard, repo_type="dataset", local_dir="/tmp/hf")
            shutil.move(downloaded, target)
        staged.append(target)
    staged.append(f"index: {build_index('parquet', FINEWEB_EDU_DIR)}")
    data_volume.commit()
    return staged


@app.function(volumes={DATA_MOUNT: data_volume}, timeout=120, cpu=2.0)
def stage_smoke_data() -> str:
    """Generate and index a small fixed dataset without downloading FineWeb."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from zephon.build_index import build_index

    directory = Path(SMOKE_DATA_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "text": [
                    " ".join(f"Document {i}, sentence {j}: deterministic training data." for j in range(64))
                    for i in range(64)
                ]
            }
        ),
        directory / "data.parquet",
    )
    index = build_index("parquet", directory)
    data_volume.commit()
    return index


@app.local_entrypoint()
def main(num_shards: int = 2, smoke: bool = False) -> None:
    """Stage the selected dataset and print its paths."""
    if smoke:
        print(stage_smoke_data.remote())
    else:
        for path in stage_fineweb_edu.remote(num_shards):
            print(path)
