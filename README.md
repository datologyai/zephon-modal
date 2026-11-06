# zephon-modal

Elastic, deterministic training runs on [Modal](https://modal.com) with
[Zephon](https://github.com/datologyai/zephon). You describe one recipe and
the physical shapes it may run on. Any launch of a run can land on any of
those shapes, resume from the latest checkpoint, and still train on exactly
the same samples at every step. `compare` checks that.

**Status: early, private.** `supervise` keeps a run going across placements
until it reaches its step target (see [Supervisor](#supervisor)), and a
launch survives Modal stopping its container (see [Preemption](#preemption)).
`tune` comes next (see [Roadmap](#roadmap)).

## How it fits together

- **Placements** are physical shapes, in order of preference:
  `Placement("H100:8", nodes=2)`, `Placement("H100:8")`, `Placement("H100:4")`.
  Each one becomes its own Modal function (`train-<placement>`), because
  Modal fixes a function's GPU spec and cluster size when it's defined.
  There's no preemption setting: Modal doesn't offer non-preemptible GPU
  Functions, so placements differ in how quickly they're available.
- **Adapters** say how to launch a framework's trainer under torchrun:
  `Torchtitan` and `Megatron`. Adapters are plain data. Recipe-specific code
  is named by import string and runs inside the container.
- **A data contract** says which data-parallel degrees the job's data loader
  allows, and what settings its trainers see. The core doesn't know about
  Zephon: the default contract allows any DP degree, which suits workloads
  that can resume anywhere and don't need identical data across resizes.
- **Zephon** (`zephon_modal.zephon`) is the contract that makes resizes
  deterministic. Zephon divides the data into lanes (`canonical_replicas`),
  and a run can train or resume on any DP degree that divides the lane
  count, so 16 lanes allow DP 1, 2, 4, 8 or 16. `ElasticJob` rejects
  placements that don't fit. Zephon's `aggregate_store` is always `torch://`
  (torchrun's own TCPStore): Volume writes don't reach other containers until
  commit + reload, so a shared `aggregate_dir` can't work across nodes, and
  restore reads the merged lane state from the trainer's checkpoint anyway.
- **Recipes read the launch from `zephon_modal.context()`** (run directory,
  step target, seed, launch ID, node and rank counts) and Zephon's settings
  from `zephon_modal.zephon.settings()`.
- **Sample hashes:** each DP rank logs a hash of every sequence it trains on
  per step. `compare` treats each step as a multiset, so runs on different
  GPU counts match when they trained on the same data. The recorder is
  framework- and loader-agnostic; `zephon_modal.zephon` has ready-made
  wrappers for torchtitan-zephon and Megatron-LM-zephon.

## A job file

```python
import modal
from zephon_modal import ElasticJob, Placement, Torchtitan
from zephon_modal.zephon import Zephon

job = ElasticJob(
    name="qwen3-smoke",
    image=training_image,  # framework + zephon>=0.2.1 + your recipe modules
    adapter=Torchtitan(module="my_recipes", config="qwen3_0p6b_zephon_smoke"),
    data=Zephon(canonical_replicas=16),
    placements=[Placement("H100:8", nodes=2), Placement("H100:8"), Placement("H100:4")],
    runs_volume=modal.Volume.from_name("my-runs", create_if_missing=True, version=2),
    volumes={"/data": modal.Volume.from_name("my-data")},
    compile_cache_volume=modal.Volume.from_name("my-compile-cache", create_if_missing=True, version=2),
)
```

In the torchtitan recipe, use `SampleHashingZephonDataLoader.Config` from
`zephon_modal.zephon.torchtitan` with `microbatches_per_step`; take
`canonical_replicas` and `aggregate_store` from `zephon_modal.zephon.settings()`,
and `run_id`, the step target and `keep_all_checkpoints` from
`zephon_modal.context()`. For Megatron-LM-zephon, pass
`Megatron(args="my_recipes:pretrain_args", script="pretrain_gpt_zephon.py",
entry="zephon_modal.zephon.megatron_entry")`, where `pretrain_args(ctx)`
returns the script's arguments.

The recipe must still consume whole lane windows per optimizer step
(microbatches per step a multiple of `canonical_replicas`). Zephon only warns
about this, so check it in the recipe.

## Commands

```bash
uv sync --group dev
uv run python -m zephon_modal placements path/to/job.py
uv run python -m zephon_modal launch path/to/job.py --run-id <you>-run --steps 20 --placement h100x8
uv run python -m zephon_modal launch path/to/job.py --run-id <you>-run --steps 40 --placement h100x4
uv run python -m zephon_modal compare path/to/job.py --reference <ref-run> --candidate <you>-run

uv run python -m zephon_modal supervise path/to/job.py --run-id <you>-run --steps 1000
uv run python -m zephon_modal status path/to/job.py --run-id <you>-run
uv run python -m zephon_modal cancel path/to/job.py --run-id <you>-run
```

`--steps` is the total to reach, not steps to add. Reusing a run ID resumes
it. Use `--detach` for anything long: without it, Modal stops the app if this
client disconnects.

With Zephon's CLI installed too (on Zephon `main` after v0.2.1), the same
commands are `zephon modal <command> ...`: zephon-modal
registers itself as a `zephon.cli` plugin. zephon-modal doesn't depend on
zephon, so `python -m zephon_modal` keeps working without it.

## Supervisor

`supervise` starts a small, non-preemptible CPU function in the job's app,
detached, and returns. It runs one launch at a time and decides where:

- **The most preferred placement that isn't cooling down** gets the next
  launch.
- **Queued past `--queue-timeout`** (15 min; at first, or while a restart
  waits for containers): the supervisor cancels the launch, cools that
  placement down for `--cooldown` (30 min), and tries the next.
- **Preempted:** Modal restarts the launch on the same placement by itself.
  Past `--max-restarts` (2) restarts, the supervisor cancels it, cools the
  placement down, and moves on.
- **Hit the function's `timeout`:** launch again; the run resumes from its
  checkpoint.
- **Failed:** launch again, up to `--max-failures` (1) in a row, then give up.

A cancelled launch stops its trainer and commits, as on a preemption. The
supervisor never stops a healthy launch to move up; cooled-down placements
come back for the next launch. Its state is `<run_dir>/supervisor.json`, which
`status` prints; `cancel` cancels the supervisor, which cancels its launch.
Each container records its attempt in `launches/` when it starts, and that is
how the supervisor tells a running launch from a queued one: Modal's call
graph left `task_id` empty for running launches.

A supervisor stops after 24 hours (Modal's longest timeout), leaving its
launch running unsupervised. `supervise` refuses to start while that launch
still runs (two launches would write the same run); run it again once the
launch ends, or set the job's `timeout` so launches end sooner.

## Preemption

When Modal preempts a container (or you run `modal container stop`), it
sends SIGINT to the function's process group, kills the container 30
seconds later, and restarts the same input on a new container. zephon-modal
makes that restart a clean resume:

- **The runner, not each rank, decides how the trainer stops.** torchrun runs
  in its own session, so Modal's SIGINT reaches only the runner. It sends
  torchrun the adapter's `shutdown_signal` (SIGTERM), which forwards it to
  every rank, waits up to `shutdown_timeout` (10 s), then kills the trainer
  and commits the Volume.
- **The steps hashed so far survive.** Each node syncs its hash files to the
  run directory every 30 s and again when stopped, so Modal's background
  commits keep them even if the final commit doesn't finish. A restarted
  attempt keeps its launch ID; it restores its node's files and appends, and
  `compare` keeps a file's last record for each step.
- **Incomplete checkpoints are set aside before a resume.** A checkpoint
  counts only if every shard its DCP `.metadata` lists is present at full
  length, which also catches another node's shards that never got
  committed. Node 0 moves the rest to `discarded-checkpoints/` (and points
  Megatron's tracker file at the latest good one) before torchrun starts.
- **Every attempt is recorded** in `launches/<launch_id>/`: `launch.json` for
  what was asked, one `<task_id>.json` per container and attempt for how it
  went (`outcome` running, then completed, failed, stopped or cancelled;
  the checkpoint it resumed from; any it set aside; the latest at the end).

A stopped run loses the steps since its last periodic checkpoint. Saving on
the way out (`Megatron(checkpoint_on_shutdown=True)`, which adds
`--exit-signal-handler`) only helps for small checkpoints: the save and its
commit must fit in Modal's 30 s, and Qwen3-0.6B with optimizer state (~8 GB)
took 37 s to save on 4 H100s.

## Smoke test

`examples/smoke` exercises launch, resume and compare with a fake trainer on
T4s (a few cents). It uses the default data contract, not Zephon: its fake
data is a pure function of (step, microbatch, seed), so it's deterministic on
any DP degree by construction.

```bash
export ZEPHON_MODAL_SMOKE_VOLUME=<you>-zephon-modal-smoke
uv run python -m zephon_modal launch examples/smoke/job.py --run-id <you>-smoke-ref --steps 10 --placement t4x2
uv run python -m zephon_modal launch examples/smoke/job.py --run-id <you>-smoke --steps 5 --placement t4x2
uv run python -m zephon_modal launch examples/smoke/job.py --run-id <you>-smoke --steps 10 --placement t4x1
uv run python -m zephon_modal compare examples/smoke/job.py --reference <you>-smoke-ref --candidate <you>-smoke
```

Expected: the second launch resumes at step 5 on one GPU, and `compare`
reports 10/10 steps identical. Running with `--seed 43` gives 0/10 and exit
status 1.

To rehearse a preemption, use the `preempt` job (3 s per step), and stop its
container mid-run from another terminal:

```bash
uv run python -m zephon_modal launch examples/smoke/job.py:job --run-id <you>-smoke-ref --steps 40 --placement t4x2
uv run python -m zephon_modal launch examples/smoke/job.py:preempt --run-id <you>-preempt --steps 40 --placement t4x2
modal container list    # find the zephon-modal-smoke-preempt container, then after ~30 s:
modal container stop <container-id>
uv run python -m zephon_modal compare examples/smoke/job.py --reference <you>-smoke-ref --candidate <you>-preempt
```

Expected: the launch restarts on a new container, resumes from step 5 or 10,
finishes, and `compare` reports 40/40 identical. `preempt_save` does the same
but saves on the way out, so it resumes from the step it was stopped at.

## Roadmap

1. **Supervisor:** walk the placement list, relaunch on preemption or
   timeout, and keep its own state on the runs Volume so it can be restarted.
2. **`zephon modal tune` / `profile`:** search Zephon worker settings per
   placement on CPU containers, then save a profile that `launch` uses.
