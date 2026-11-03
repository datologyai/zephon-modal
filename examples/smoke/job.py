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
from zephon_modal.checkpoints import CheckpointLayout


@dataclass(frozen=True, kw_only=True)
class FakeTrainer(Adapter):
    """Runs examples/smoke/fake_trainer.py.

    Attributes:
        step_seconds: Sleep per step, to leave time to stop a container mid-run.
        checkpoint_on_shutdown: Save at the end of the current step when stopped.
    """

    step_seconds: float = 0.0
    checkpoint_on_shutdown: bool = False
    checkpoints: CheckpointLayout | None = CheckpointLayout()

    def workdir(self) -> str:
        return "/root"

    def command(self, ctx: LaunchContext) -> list[str]:
        return ["fake_trainer.py"]

    def env(self, ctx: LaunchContext) -> dict[str, str]:
        return {
            "FAKE_TRAINER_STEP_SECONDS": str(self.step_seconds),
            "FAKE_TRAINER_CHECKPOINT_ON_SHUTDOWN": "1" if self.checkpoint_on_shutdown else "0",
        }


image = (
    modal.Image.debian_slim(python_version="3.12")
    # numpy for torch.distributed.checkpoint.
    .uv_pip_install("torch", "numpy", index_url="https://download.pytorch.org/whl/cpu")
    .add_local_file(os.path.join(os.path.dirname(__file__), "fake_trainer.py"), "/root/fake_trainer.py")
    # The adapter class is defined here, so the container must be able to unpickle it.
    .add_local_python_source("job")
)

runs_volume = modal.Volume.from_name(
    os.environ.get("ZEPHON_MODAL_SMOKE_VOLUME", "zephon-modal-smoke-runs"), create_if_missing=True, version=2
)


def smoke_job(name: str, adapter: FakeTrainer, timeout: int = 10 * 60) -> ElasticJob:
    """A fake-trainer job on one or two T4s."""
    return ElasticJob(
        name=name,
        image=image,
        adapter=adapter,
        placements=[Placement("T4:2"), Placement("T4")],
        runs_volume=runs_volume,
        cpu=2.0,
        memory=4096,
        timeout=timeout,
    )


job = smoke_job("zephon-modal-smoke", FakeTrainer())
# Slow steps, so a container can be stopped mid-run (`modal container stop`)
# to rehearse a preemption. `preempt` exits without saving; `preempt_save`
# saves on the way out.
preempt = smoke_job("zephon-modal-smoke-preempt", FakeTrainer(step_seconds=3.0), timeout=20 * 60)
preempt_save = smoke_job(
    "zephon-modal-smoke-preempt-save", FakeTrainer(step_seconds=3.0, checkpoint_on_shutdown=True), timeout=20 * 60
)
