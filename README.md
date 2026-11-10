# zephon-modal

Elastic, deterministic training runs on [Modal](https://modal.com) with
[Zephon](https://github.com/datologyai/zephon). You describe one recipe and
the physical shapes it may run on. A run moves between those shapes as
capacity comes and goes, resumes from its latest checkpoint each time, and
still trains on exactly the same samples at every step. `compare` checks
that.

**Status: early, private.** Meant to carry a multi-day torchtitan run; see
[Before a long run](#before-a-long-run) for what has and hasn't been
exercised.

## How it fits together

- **Placements** are physical shapes, in order of preference:
  `Placement("H100:8", nodes=2)`, `Placement("H100:8", fallbacks=("H100:4",))`.
  Each becomes its own Modal function (`train-<placement>`), because Modal
  fixes a function's GPU spec and cluster size when it's defined. A
  single-node placement can list fallbacks: Modal allocates the first shape it
  can, and the launch sizes itself from the GPUs it got. Fallbacks are for
  single-node placements only: Modal's docs don't say whether a cluster can
  take them, and we haven't tested it, so make each cluster size its own
  placement. Modal has no API for
  available capacity, so fallbacks and moving between placements are how a
  run gets GPUs sooner.
- **Adapters** say how to launch a framework's trainer under torchrun:
  `Torchtitan` and `Megatron`. They're plain data; recipe-specific code is
  named by import string and runs inside the container.
- **A data contract** says which data-parallel degrees the loader allows and
  what settings trainers see. The default allows any DP degree.
  `zephon_modal.zephon.Zephon` is the one that makes resizes deterministic:
  Zephon divides the data into lanes (`canonical_replicas`), and a run can
  train or resume at any DP degree that divides the lane count. Its
  `aggregate_store` is always `torch://`: Volume writes don't reach other
  containers until commit and reload, so a shared directory can't work.
- **Recipes read the launch from `zephon_modal.context()`** (run directory,
  step target, seed, launch ID, node and rank counts) and Zephon's settings
  from `zephon_modal.zephon.settings()`.
- **Sample hashes** are the proof. Each DP rank hashes every sequence it
  trains on (tokens, labels, positions and masks) per step; `compare` treats
  a step as a multiset, so runs at different DP degrees match when they saw
  the same data.
- **The reconciler** keeps started runs going. It's a scheduled function in
  the job's deployed app; see [Reconciler](#reconciler).

## A job file

```python
import modal
from zephon_modal import ElasticJob, Placement, Torchtitan
from zephon_modal.zephon import Zephon

job = ElasticJob(
    name="qwen3",
    image=training_image,  # framework + zephon>=0.2.1 + your recipe modules
    adapter=Torchtitan(module="my_recipes", config="qwen3_0p6b_zephon"),
    data=Zephon(canonical_replicas=16),
    placements=[Placement("H100:8", nodes=2), Placement("H100:8", fallbacks=("H100:4",))],
    runs_volume=modal.Volume.from_name("my-runs", create_if_missing=True, version=2),
    volumes={"/data": modal.Volume.from_name("my-data")},
    compile_cache_volume=modal.Volume.from_name("my-compile-cache", create_if_missing=True, version=2),
    alert_secret=modal.Secret.from_name("my-slack-webhook"),  # provides ZEPHON_MODAL_ALERT_WEBHOOK
)
```

In the torchtitan recipe, use `SampleHashingZephonDataLoader.Config` from
`zephon_modal.zephon.torchtitan`; take `canonical_replicas` and
`aggregate_store` from `zephon_modal.zephon.settings()`, and `run_id` and the
step target from `zephon_modal.context()`. Have the trainer keep every
checkpoint (torchtitan's `keep_latest_k=0`) and pass `--keep-checkpoints` to
`start` instead: on a cluster each node sees only its own uncommitted files,
so a trainer pruning there leaves partial checkpoints behind. For
Megatron-LM-zephon, pass `Megatron(args="my_recipes:pretrain_args",
script="pretrain_gpt_zephon.py", entry="zephon_modal.zephon.megatron_entry")`.
Both wrappers reject a step that isn't a whole number of lane windows.

## Commands

```bash
uv sync --group dev
uv run python -m zephon_modal placements path/to/job.py
uv run python -m zephon_modal start path/to/job.py --run-id <you>-run --steps 100000 --keep-checkpoints 3
uv run python -m zephon_modal status path/to/job.py --run-id <you>-run
uv run python -m zephon_modal stop path/to/job.py --run-id <you>-run
uv run python -m zephon_modal recover path/to/job.py --run-id <you>-run
uv run python -m zephon_modal launch path/to/job.py --run-id <you>-ref --steps 200 --placement h100x8
uv run python -m zephon_modal compare path/to/job.py --reference <you>-ref --candidate <you>-run --steps 200
uv run python -m zephon_modal replays path/to/job.py --run-id <you>-run
```

`start` deploys the job's app and hands the run to the reconciler, which
takes it from there. `--steps` is the total to reach. Starting a finished or stopped
run again resumes it. `launch` trains on one placement right now, without the
reconciler: for references and experiments. A deployed app keeps its
reconciler schedule until you `modal app stop` it.

With Zephon's CLI installed (Zephon `main` after v0.2.1), the same commands
are `zephon modal <command> ...`.

## Reconciler

Nothing long-lived watches a run. The deployed app has a `reconcile`
function that Modal calls every `reconcile_minutes` (5 by default) and
right after `start`, `stop` or `recover`. Each call looks at every active run once, takes at most one
decision, and exits. Modal runs one call at a time (`max_containers=1`), so
there is exactly one decision-maker with no locks, and no process that can
outlive Modal's 24-hour limit or die holding the run.

Runs live in a Modal Dict, `<app>-runs`: the plan (a spec per placement,
and the policy), the state, and pending start/stop/recover requests. The CLI
only writes requests; the reconciler alone writes plans and state, so a
second `start` of an active run is ignored rather than racing the first. The
state is also mirrored to `<run_dir>/reconciler.json`.

Each tick, for each run:

- **No current launch:** launch on the most preferred placement that isn't
  cooling down.
- **Queued past `--queue-timeout`** (15 min), at first or while a restart
  waits for containers: cancel, cool the placement down for `--cooldown`
  (30 min), move on.
- **Preempted:** Modal restarts the launch on the same placement by itself.
  Past `--max-restarts` (2), cancel, cool down, move on.
- **Running without a new checkpoint for `--stall-timeout`** (1 h): treat it
  as hung and cancel; it counts as a failure. So does a launch whose
  heartbeat goes quiet for 5 min.
- **Hit the function `timeout`** (24 h by default): launch again; it resumes.
- **Failed:** cool the placement down and launch again, on the next
  placement if there is one, so a bad node or shape doesn't take every retry.
  After `--max-failures` (3) in a row without a new checkpoint, give up.
- **New checkpoint:** reset the failure count, and with `--keep-checkpoints N`
  delete all but the latest N complete checkpoints, from the reconciler's
  committed view of the Volume.
- **`--max-gpu-hours`:** give up once the run's attempts add up to that.

**One writer at a time.** Before any successor launch, the old one must
*drain*: its call has ended and every container it started has recorded its
final commit. Each container publishes an attempt record to the registry
before touching the run, refreshes it every 30 s, and marks it ended only
after its last Volume commit has returned. These records live in the Dict,
not on the Volume: a Dict write is atomic and immediately visible, while a
reader can catch a Volume commit half-applied (we saw records read back
empty). A launch also checks, before touching the run, that it is still the
run's current launch. If a launch doesn't
confirm it stopped within 10 minutes, the run waits for an operator: check
that its containers are gone, then `recover`. A launch that has already
ended and confirmed is replaced in the same tick.

**Alerts** go to `ZEPHON_MODAL_ALERT_WEBHOOK` (a Slack-compatible incoming
webhook, from `alert_secret`): give-ups, runs needing recovery, no new
checkpoint for 2 hours, and completion.

## Preemption

When Modal preempts a container (or you run `modal container stop`), it
sends SIGINT to the function's process group, kills the container after a
grace period (about 30 seconds in our tests; Modal's docs don't give a
length), and restarts the same input on new containers. On a cluster, stopping
one node stops and restarts them all. zephon-modal makes the restart a clean
resume:

- **The runner, not each rank, stops the trainer.** torchrun runs in its own
  session, so Modal's SIGINT reaches only the runner. It syncs hashes, sends
  torchrun the adapter's `shutdown_signal` (SIGTERM), waits up to
  `shutdown_timeout` (10 s), kills the trainer if needed, and commits.
- **Hashes survive.** Each node syncs changed hash files to the run every
  30 s, so background commits keep them even if the final commit doesn't
  finish. Every attempt hashes into its own time-ordered namespace
  (`<launch_id>.a<time>-<hex>`),
  and files are split every 1000 steps so syncs stay small.
- **Every node resumes from the same checkpoint.** Node 0 sets aside
  incomplete checkpoints (a DCP checkpoint counts only if every shard its
  `.metadata` lists is present at full length) and commits; the other nodes
  reload and confirm they see the same latest checkpoint before any trainer
  starts.
- **Data settings can't change under a run.** `run.json` records the seed and
  the data contract's settings; a resume with different ones is refused.
  Anything else (trainer flags, this library) may change between launches.

A stopped run loses the steps since its last checkpoint. Saving on the way
out doesn't fit: the save and its commit must finish within the grace
period, and Qwen3-0.6B with optimizer state (~8 GB) took 37 s on 4 H100s.

## Verification

Hashing is optional, and chosen by the recipe. In torchtitan, use
`SampleHashingZephonDataLoader` instead of Zephon's plain loader; in Megatron,
pass `entry="zephon_modal.zephon.megatron_entry"`. Without them nothing is
hashed, and the runner's periodic sync only refreshes its heartbeat. Hashing
costs milliseconds of CPU per step and a few kilobytes per step on the
Volume, but it sits in the data path: it rejects batches it can't hash (extra
`model_kwargs`, for now), so turn it on deliberately.

Two checks use the hashes:

```bash
uv run python -m zephon_modal replays path/to/job.py --run-id <run>
uv run python -m zephon_modal compare path/to/job.py --reference <ref> --candidate <run> --steps 200 --rows-per-step 128
```

- **`replays`** checks a run against itself, with no reference run. Every
  resume re-trains the steps since the last checkpoint, often at a different
  DP degree, so those steps are hashed by both attempts and must match. A
  long run collects this evidence at every resize for free. A step an
  attempt was cut off in the middle of is skipped.
- **`compare`** checks a run against a reference run. With `--steps` it's
  strict: every step in the range must be present in both runs, from every
  DP rank, with the expected row count, and match. Without it, it compares
  the steps both runs have.

Under tensor or context parallelism, every rank sharing a DP rank records
the same rows; each DP rank counts once, and only if its copies agree.
Hashes cover the batches the loader handed the trainer, not completed
optimizer updates.

## Smoke test

`examples/smoke` exercises the whole path with a fake trainer on T4s (a few
cents). Set a prefix: deployed app names are shared across the workspace.

```bash
export ZEPHON_MODAL_SMOKE_PREFIX=<you>-smoke ZEPHON_MODAL_SMOKE_VOLUME=<you>-smoke-runs
uv run python -m zephon_modal launch examples/smoke/job.py --run-id <you>-ref --steps 40 --placement t4x1
uv run python -m zephon_modal start examples/smoke/job.py:preempt --run-id <you>-run --steps 40
uv run python -m zephon_modal status examples/smoke/job.py:preempt --run-id <you>-run
modal container stop <task>   # from `status`, once it has a checkpoint
uv run python -m zephon_modal compare examples/smoke/job.py --reference <you>-ref --candidate <you>-run --steps 40
modal app stop <you>-smoke-preempt
```

Expected: Modal restarts the launch, the reconciler records the restart, the
run finishes, and `compare` reports 40/40. `short_timeout` ends every launch
after two minutes, as a long run's launches end at 24 hours.

## Before a long run

Exercised live with the reconciler, on T4s: a stopped container restarted by
Modal and followed to completion; a launch ending at the function timeout,
draining and relaunching; `stop`, then `start` again resuming; checkpoint
pruning; and failed launches retried, then given up on, each after draining.
On H100s (in zephon-modal-demo): torchtitan moving from a 2-node cluster
(DP 16) to one node (DP 8) after a stop, and Megatron restarting in place on
a 2-node cluster. Every one matched a reference run under the strict
comparison, and `replays` matched across the resize.

Not yet exercised: a queue timeout under the reconciler (Modal allocated
every time), moving to another placement after a failure, the alert webhook,
`recover`, a real preemption (all were `modal container stop`), and anything
longer than a day. Rehearse a long run on T4s, then on the real shape, first.
