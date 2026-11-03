# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Smoke job: zephon_modal's launch, resume and compare path on cheap GPUs.

    python -m zephon_modal placements examples/smoke/job.py
    python -m zephon_modal launch examples/smoke/job.py --run-id <you>-smoke-ref --steps 10 --placement t4x2
    python -m zephon_modal launch examples/smoke/job.py --run-id <you>-smoke --steps 5 --placement t4x2
    python -m zephon_modal launch examples/smoke/job.py --run-id <you>-smoke --steps 10 --placement t4x1
    python -m zephon_modal compare examples/smoke/job.py --reference <you>-smoke-ref --candidate <you>-smoke

Set ZEPHON_MODAL_SMOKE_VOLUME to use your own runs Volume.
"""

import os
from dataclasses import dataclass

import modal

from zephon_modal import Adapter, ElasticJob, LaunchContext, Placement


@dataclass(frozen=True, kw_only=True)
class FakeTrainer(Adapter):
    """Runs examples/smoke/fake_trainer.py."""

    def workdir(self) -> str:
        return "/root"

    def command(self, ctx: LaunchContext) -> list[str]:
        return ["fake_trainer.py"]


image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
    .add_local_file(os.path.join(os.path.dirname(__file__), "fake_trainer.py"), "/root/fake_trainer.py")
    # The adapter class is defined here, so the container must be able to unpickle it.
    .add_local_python_source("job")
)

job = ElasticJob(
    name="zephon-modal-smoke",
    image=image,
    adapter=FakeTrainer(),
    placements=[Placement("T4:2"), Placement("T4")],
    runs_volume=modal.Volume.from_name(
        os.environ.get("ZEPHON_MODAL_SMOKE_VOLUME", "zephon-modal-smoke-runs"), create_if_missing=True, version=2
    ),
    cpu=2.0,
    memory=4096,
    timeout=10 * 60,
)
