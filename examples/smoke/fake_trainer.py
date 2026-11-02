# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""A stand-in trainer for exercising zephon_modal's launch path cheaply.

Behaves like a trainer from the outside: it resumes from the latest
checkpoint under the run directory, trains each rank's share of a step's
global microbatches, records sample hashes, and checkpoints every few steps.
The "data" is a pure function of (step, microbatch, seed), which is what
Zephon guarantees for real data, so `compare` must match across DP degrees.
"""

import os
import re

import torch.distributed as dist

from zephon_modal import context
from zephon_modal.verify.hashes import StepHashRecorder, hash_file_name

MICROBATCHES_PER_STEP = 4
ROWS_PER_MICROBATCH = 2
CHECKPOINT_INTERVAL = 5


def latest_checkpoint(checkpoint_dir: str) -> int:
    """The highest checkpointed step, or 0 for a fresh run."""
    steps = [int(m.group(1)) for name in os.listdir(checkpoint_dir) if (m := re.fullmatch(r"step-(\d+)", name))]
    return max(steps, default=0)


def main() -> None:
    """Train from the latest checkpoint to the launch's step target."""
    ctx = context()
    dist.init_process_group("gloo")
    rank, dp = dist.get_rank(), dist.get_world_size()
    checkpoint_dir = f"{ctx.run_dir}/checkpoint"
    os.makedirs(checkpoint_dir, exist_ok=True)
    start = latest_checkpoint(checkpoint_dir)
    hashes = StepHashRecorder(
        os.path.join(ctx.sample_hash_dir, hash_file_name(ctx.launch_id, dp, rank)),
        dp_world_size=dp,
        dp_rank=rank,
        microbatches_per_step=MICROBATCHES_PER_STEP,
        consumed_microbatches=start * MICROBATCHES_PER_STEP,
    )
    print(f"[fake_trainer] rank {rank}/{dp} on {ctx.placement}: resuming at step {start}, training to {ctx.steps}")
    for step in range(start, ctx.steps):
        for microbatch in range(rank, MICROBATCHES_PER_STEP, dp):
            hashes.record(f"{ctx.seed}-{step}-{microbatch}-{row}".encode() for row in range(ROWS_PER_MICROBATCH))
        dist.barrier()
        if rank == 0 and (step + 1) % CHECKPOINT_INTERVAL == 0:
            os.makedirs(f"{checkpoint_dir}/step-{step + 1}", exist_ok=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
