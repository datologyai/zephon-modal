# zephon-modal

Elastic, deterministic training runs on [Modal](https://modal.com) with
[Zephon](https://github.com/datologyai/zephon). You describe one recipe and
the physical shapes it may run on. Any launch of a run can land on any of
those shapes, resume from the latest checkpoint, and still train on exactly
the same samples at every step. `compare` checks that.

**Status: stage 0, private experiment.** Launches are manual (you pick the
placement). The supervisor that moves a run between placements on its own,
checkpoint handling that survives preemption, and `tune` come next (see
[Roadmap](#roadmap)).

## How it fits together

- **Lanes, not ranks.** Zephon divides the data into `lanes`
  (`canonical_replicas`). A run can train or resume on any data-parallel
  degree that divides the lane count, so 16 lanes allow DP 1, 2, 4, 8 or 16.
- **Placements** are physical shapes, in order of preference:
  `Placement("H100:8", nodes=2)`, `Placement("H100:4", nonpreemptible=True)`, ...
  Each one becomes its own Modal function (`train-<placement>`), because
  Modal fixes a function's GPU spec and cluster size when it's defined.
  `ElasticJob` rejects placements whose DP degree doesn't divide the lanes.
- **Adapters** say how to launch a framework's trainer under torchrun:
  `Torchtitan` (torchtitan-zephon) and `Megatron` (Megatron-LM-zephon).
  Adapters are plain data. Recipe-specific code is named by import string and
  runs inside the container.
- **Recipes read the launch from `zephon_modal.context()`**: run directory,
  step target, seed, launch ID, and the Zephon `aggregate_store` /
  `aggregate_dir` (multi-node launches use torchrun's TCPStore, single-node
  launches a directory on the runs Volume).
- **Sample hashes:** each DP rank logs a hash of every packed sequence per
  step. `compare` treats each step as a multiset, so runs on different GPU
  counts match when they trained on the same data.

## A job file

```python
import modal
from zephon_modal import ElasticJob, Placement, Torchtitan

job = ElasticJob(
    name="qwen3-smoke",
    image=training_image,  # framework + zephon>=0.2.1 + your recipe modules
    adapter=Torchtitan(module="my_recipes", config="qwen3_0p6b_zephon_smoke"),
    lanes=16,
    placements=[Placement("H100:8", nodes=2), Placement("H100:8"), Placement("H100:4")],
    runs_volume="my-runs",
    volumes={"/data": modal.Volume.from_name("my-data")},
    compile_cache_volume="my-compile-cache",
)
```

In the torchtitan recipe, use `SampleHashingZephonDataLoader.Config` from
`zephon_modal.adapters.torchtitan_loader` with `microbatches_per_step`, and
take `run_id`, `aggregate_store` / `aggregate_dir`, the step target and
`keep_all_checkpoints` from `zephon_modal.context()`. For Megatron, pass
`Megatron(args="my_recipes:pretrain_args")`, where `pretrain_args(ctx)`
returns `pretrain_gpt_zephon.py`'s arguments.

The recipe must still consume whole lane windows per optimizer step
(microbatches per step a multiple of `lanes`). Zephon only warns about this,
so check it in the recipe.

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
T4s (a few cents):

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
