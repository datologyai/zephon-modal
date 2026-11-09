# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Smoke job: zephon_modal's start, resume and compare path on cheap GPUs.

    export ZEPHON_MODAL_SMOKE_PREFIX=<you>-smoke   # deployed app names are shared across the workspace
    python -m zephon_modal placements examples/smoke/job.py
    python -m zephon_modal launch examples/smoke/job.py --run-id <you>-ref --steps 40 --placement t4x2
    python -m zephon_modal start examples/smoke/job.py:preempt --run-id <you>-run --steps 40
    python -m zephon_modal status examples/smoke/job.py:preempt --run-id <you>-run
    modal container stop <task>   # mid-run, to rehearse a preemption
    python -m zephon_modal compare examples/smoke/job.py --reference <you>-ref --candidate <you>-run --steps 40
    modal app stop <you>-smoke-preempt   # deployed apps keep their reconciler schedule until stopped

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


def smoke_job(
    name: str, adapter: FakeTrainer, timeout: int = 10 * 60, placements: list[Placement] | None = None
) -> ElasticJob:
    """A fake-trainer job on one or two T4s."""
    return ElasticJob(
        name=name,
        image=image,
        adapter=adapter,
        placements=placements or [Placement("T4:2"), Placement("T4")],
        runs_volume=runs_volume,
        cpu=2.0,
        memory=4096,
        timeout=timeout,
    )


PREFIX = os.environ.get("ZEPHON_MODAL_SMOKE_PREFIX", "zephon-modal-smoke")
job = smoke_job(PREFIX, FakeTrainer())
# Slow steps, so a container can be stopped mid-run (`modal container stop`)
# to rehearse a preemption. `preempt` exits without saving; `preempt_save`
# saves on the way out.
preempt = smoke_job(f"{PREFIX}-preempt", FakeTrainer(step_seconds=3.0), timeout=20 * 60)
preempt_save = smoke_job(
    f"{PREFIX}-preempt-save", FakeTrainer(step_seconds=3.0, checkpoint_on_shutdown=True), timeout=20 * 60
)
# Launches end at Modal's function timeout after two minutes, as a day-long
# run's do at 24 hours; the reconciler relaunches them and they resume.
short_timeout = smoke_job(f"{PREFIX}-timeout", FakeTrainer(step_seconds=3.0), timeout=120)
# One Modal function asking for ["T4:2", "T4"]: Modal allocates whichever it
# can, and the launch sizes itself from the GPUs it got.
fallback = smoke_job(f"{PREFIX}-fallback", FakeTrainer(), placements=[Placement("T4:2", fallbacks=("T4",))])
