# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""A stand-in trainer for exercising zephon_modal's launch path cheaply.

Behaves like a trainer from the outside: it resumes from the latest
checkpoint under the run directory, trains each rank's share of a step's
global microbatches, records sample hashes, and saves a DCP checkpoint every
few steps. The "data" is a pure function of (step, microbatch, seed), which
is what Zephon guarantees for real data, so `compare` must match across DP
degrees.

On SIGTERM it stops at the end of the current step without saving.
FAKE_TRAINER_STEP_SECONDS slows steps down so a container can be stopped
mid-run.
"""

import os
import re
import signal
import time

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict_saver import save as dcp_save

from zephon_modal import context
from zephon_modal.verify.hashes import StepHashRecorder, hash_file_name

MICROBATCHES_PER_STEP = 4
ROWS_PER_MICROBATCH = 2
CHECKPOINT_INTERVAL = 5

stop_requested = False


def _on_signal(signum: int, _frame: object) -> None:
    global stop_requested
    print(f"[fake_trainer] rank {os.environ.get('RANK')} got {signal.Signals(signum).name}", flush=True)
    if signum == signal.SIGTERM:
        stop_requested = True


def latest_checkpoint(checkpoint_dir: str) -> int:
    """The highest step with DCP metadata (as torchtitan picks it), or 0 for a fresh run."""
    steps = [
        int(m.group(1))
        for name in os.listdir(checkpoint_dir)
        if (m := re.fullmatch(r"step-(\d+)", name)) and os.path.isfile(f"{checkpoint_dir}/{name}/.metadata")
    ]
    return max(steps, default=0)


def save(checkpoint_dir: str, step: int, rank: int) -> None:
    """Per-rank state, like a sharded optimizer: every rank writes its own file."""
    dcp_save({f"rank{rank}": torch.full((1024,), float(step))}, checkpoint_id=f"{checkpoint_dir}/step-{step}")


def main() -> None:
    """Train from the latest checkpoint to the launch's step target."""
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _on_signal)
    ctx = context()
    step_seconds = float(os.environ.get("FAKE_TRAINER_STEP_SECONDS", "0"))
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
        schema="fake-v1",
        consumed_microbatches=start * MICROBATCHES_PER_STEP,
    )
    print(f"[fake_trainer] rank {rank}/{dp} on {ctx.placement}: resuming at step {start}, training to {ctx.steps}")
    for step in range(start, ctx.steps):
        time.sleep(step_seconds)
        for microbatch in range(rank, MICROBATCHES_PER_STEP, dp):
            hashes.record(f"{ctx.seed}-{step}-{microbatch}-{row}".encode() for row in range(ROWS_PER_MICROBATCH))
        # Every rank must agree to stop at the same step.
        stop = torch.tensor([int(stop_requested)])
        dist.all_reduce(stop, op=dist.ReduceOp.MAX)
        if (step + 1) % CHECKPOINT_INTERVAL == 0:
            save(checkpoint_dir, step + 1, rank)
            if rank == 0:
                print(f"[fake_trainer] saved step {step + 1}", flush=True)
        if stop.item():
            print(f"[fake_trainer] rank {rank} stopping after step {step + 1}", flush=True)
            break
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
