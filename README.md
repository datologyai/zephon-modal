# zephon-modal

Elastic, deterministic training runs on [Modal](https://modal.com) with
[Zephon](https://github.com/datologyai/zephon). You describe one recipe and
the physical shapes it may run on. Any launch of a run can land on any of
those shapes, resume from the latest checkpoint, and still train on exactly
the same samples at every step. `compare` checks that.

**Status: early, private.** Launches are manual (you pick the
placement). The supervisor that moves a run between placements on its own,
checkpoint handling that survives preemption, and `tune` come next (see
[Roadmap](#roadmap)).

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
```

`--steps` is the total to reach, not steps to add. Reusing a run ID resumes
it. Use `--detach` for anything long: without it, Modal stops the app if this
client disconnects.

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

## Roadmap

1. **Checkpoints that survive preemption:** completion markers, a commit on
   SIGTERM, and safe cleanup of old checkpoints on clusters.
2. **Supervisor:** walk the placement list, relaunch on preemption or
   timeout, and keep its own state on the runs Volume so it can be restarted.
3. **`tune` / `profile`:** search Zephon worker settings per placement on CPU
   containers, then save a profile that `launch` uses.
4. **`zephon modal ...`:** once the `zephon` CLI harness ships, register this
   package as its `modal` subcommand.
