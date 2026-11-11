# How zephon-modal works

The [README](../README.md) covers what zephon-modal is for and how to use it.
This page covers how it keeps a run going and why it's built that way. Read it
before trusting a long run to it, or before changing the reconciler.

## Concepts in more detail

**Placements.** Each placement becomes its own Modal function
(`train-<placement>`), because Modal fixes a function's GPU spec and cluster
size when it's defined. A single-node placement can list fallbacks
(`Placement("H100:8", fallbacks=("H100:4",))`): Modal allocates the first
shape it can, and the launch sizes itself from the GPUs it got. Fallbacks are
for single-node placements only. Modal's docs don't say whether a cluster can
take them, and we haven't tested it, so make each cluster size its own
placement. Modal has no API for available capacity, so fallbacks and moving
between placements are how a run gets GPUs sooner.

**Adapters** are plain data. Recipe-specific code is named by import string
and runs inside the container, so the local client never imports torch.

**The data contract** says which data-parallel degrees the loader allows and
what settings trainers see. The default allows any DP degree.
`zephon_modal.zephon.Zephon` is the one that makes resizes deterministic:
Zephon divides the data into lanes (`canonical_replicas`), and a run can train
or resume at any DP degree that divides the lane count. Its `aggregate_store`
is always `torch://`: Volume writes don't reach other containers until commit
and reload, so a shared directory can't work.

**Recipes read the launch** from `zephon_modal.context()` (run directory, step
target, seed, launch ID, node and rank counts) and Zephon's settings from
`zephon_modal.zephon.settings()`.

## The reconciler

Nothing long-lived watches a run. The deployed app has a `reconcile` function
that Modal calls every `reconcile_minutes` (5 by default) and right after
`start`, `stop` or `recover`. Each call looks at every active run once, takes
at most one decision, and exits. Modal runs one call at a time
(`max_containers=1`), so there is exactly one decision-maker with no locks,
and no process that can outlive Modal's 24-hour limit or die holding the run.

Runs live in a Modal Dict, `<app>-runs`: the plan (a spec per placement, and
the policy), the state, and pending start/stop/recover requests. The CLI only
writes requests; the reconciler alone writes plans and state, so a second
`start` of an active run is ignored rather than racing the first. The state is
also mirrored to `<run_dir>/reconciler.json`.

### What it decides, each tick

| Situation | What the reconciler does |
|---|---|
| No current launch | Launch on the most preferred placement that isn't cooling down. |
| Queued past `--queue-timeout` (15 min), at first or while a restart waits for containers | Cancel, cool the placement down for `--cooldown` (30 min), move on. |
| Preempted | Nothing: Modal restarts the launch on the same placement by itself. Past `--max-restarts` (2), cancel, cool down, move on. |
| Running with no new checkpoint for `--stall-timeout` (1 h) | Treat it as hung and cancel; it counts as a failure. |
| A node of the running attempt has a quiet heartbeat, or has exited while the others run, for 5 min | Same as a stall: torchrun on the rest could otherwise wait on it until the stall timeout. |
| Hit the function `timeout` (24 h by default) | Launch again; it resumes. |
| Failed | Cool the placement down and launch again, on the next placement if there is one, so a bad node or shape doesn't take every retry. After `--max-failures` (3) in a row without a new checkpoint, give up. |
| Lost by Modal (`InternalFailure`) | The same, but counted against `--max-lost` (10) instead, since it says nothing about the trainer. A `RemoteError` still counts as a failure: a container that crashed or ran out of memory looks like one. |
| Modal can't be asked (connection or service errors) | Use the containers' heartbeats from the registry instead, and never cancel only for not knowing. If it lasts, the 2-hour quiet alert says so. |
| New checkpoint | Reset the failure counts. With `--keep-checkpoints N`, delete all but the latest N complete checkpoints, from the reconciler's committed view of the Volume. |
| Attempts add up to `--max-gpu-hours` | Give up. |

### One writer at a time

Before any successor launch, the old one must *drain*: its call has ended and
every container it started has recorded its final commit. Each container
publishes an attempt record to the registry before touching the run,
refreshes it every 30 s, and marks it ended only after its last Volume commit
has returned.

These records live in the Dict, not on the Volume: a Dict write is atomic and
immediately visible, while a reader can catch a Volume commit half-applied (we
saw records read back empty).

A launch also checks, before touching the run, that it is still the run's
current launch, and checks again with every heartbeat. If the reconciler has
moved on (after a `recover`, or a spawn that failed on our side but not
Modal's), the launch stops its trainer and commits. A failed read of the
registry isn't a reason to stop.

If a launch doesn't confirm it stopped within 10 minutes, the run waits for an
operator: check that its containers are gone, then `recover`. A launch that
has already ended and confirmed is replaced in the same tick.

### Alerts

Alerts go to `ZEPHON_MODAL_ALERT_WEBHOOK`, a Slack-compatible incoming
webhook provided by the job's `alert_secret`. They cover give-ups, runs
needing recovery, no new checkpoint for 2 hours, and completion.

## Preemption

When Modal preempts a container (or you run `modal container stop`), it sends
SIGINT to the function's process group, kills the container after a grace
period (about 30 seconds in our tests; Modal's docs don't give a length), and
restarts the same input on new containers. On a cluster, stopping one node
stops and restarts them all. zephon-modal makes the restart a clean resume:

- **The runner, not each rank, stops the trainer.** torchrun runs in its own
  session, so Modal's SIGINT reaches only the runner. It syncs hashes, sends
  torchrun the adapter's `shutdown_signal` (SIGTERM), waits up to
  `shutdown_timeout` (10 s), kills the trainer if needed, and commits.
- **Hashes survive.** Each node syncs changed hash files to the run every
  30 s, so background commits keep them even if the final commit doesn't
  finish. Every attempt hashes into its own time-ordered namespace
  (`<launch_id>.a<time>-<hex>`), and files are split every 1000 steps so
  syncs stay small.
- **Every node resumes from the same checkpoint.** Node 0 sets aside
  incomplete checkpoints (a DCP checkpoint counts only if every shard its
  `.metadata` lists is present at full length, and any
  `CheckpointLayout.required_torch_files` are readable) and commits; the other nodes
  reload and confirm they see the same latest checkpoint before any trainer
  starts.
- **Data settings can't change under a run.** `run.json` records the seed and
  the data contract's settings; a resume with different ones is refused.
  Anything else (trainer flags, this library) may change between launches.

A stopped run loses the steps since its last checkpoint. Saving on the way out
doesn't fit: the save and its commit must finish within the grace period, and
Qwen3-0.6B with optimizer state (~8 GB) took 37 s on 4 H100s.

**Why trainers shouldn't prune checkpoints.** On a cluster each node sees only
its own uncommitted Volume files, so a trainer pruning there leaves other
nodes' shards behind as partial checkpoints. Keep every checkpoint in the
trainer (torchtitan's `keep_latest_k=0`) and let `--keep-checkpoints` prune
from the reconciler's committed view.

### Rehearsing preemptions with `stop-after`

```bash
uv run python -m zephon_modal stop-after path/to/job.py --run-id <run> --after 40 120 [--timeout SECONDS]
```

`stop-after` polls every 10 s: checkpoints on the runs Volume, and attempt
records in the job's registry (the reconciler only looks every few minutes).
Once the first checkpoint at or past each step in `--after` is on the Volume
with its DCP `.metadata`, it waits 20 s for the rest of the save (other nodes'
shards) and runs `modal container stop` on the current launch's node-0
container.

It does that only while that attempt's record says it's `running`, and only
once the checkpoint is past the one the attempt resumed from, so every
placement trains before it's stopped. On a cluster, stopping node 0 stops
every node. It exits nonzero if the run stops being active before every step
was reached, or at `--timeout` (3 hours by default). Under
`start --max-restarts 0`, every stop becomes a move to the next placement.

These are injected stops, not real preemptions: Modal sends the same SIGINT
and restarts the input the same way, but how often real preemptions happen is
unknown.

## Sample hashes

Hashing is optional, and chosen by the recipe. In torchtitan, use
`SampleHashingZephonDataLoader` instead of Zephon's plain loader; in Megatron,
pass `entry="zephon_modal.zephon.megatron_entry"`. Without them nothing is
hashed, and the runner's periodic sync only refreshes its heartbeat.

Each DP rank hashes every sequence it trains on (tokens, labels, positions and
masks) per step. Hashing costs milliseconds of CPU per step and a few
kilobytes per step on the Volume, but it sits in the data path: it rejects
batches it can't hash (extra `model_kwargs`, for now), so turn it on
deliberately. Both wrappers reject a step that isn't a whole number of lane
windows.

`compare` treats a step as a multiset of hashes, so runs at different DP
degrees match when they saw the same data.

- **`replays`** checks a run against itself, with no reference run. Every
  resume re-trains the steps since the last checkpoint, often at a different
  DP degree, so those steps are hashed by both attempts and must match. A long
  run collects this evidence at every resize for free. A step an attempt was
  cut off in the middle of is skipped.
- **`compare`** checks a run against a reference run. With `--steps` it's
  strict: every step in the range must be present in both runs, from every DP
  rank, with the expected row count (`--rows-per-step`), and match. Without
  it, it compares the steps both runs have.

Under tensor or context parallelism, every rank sharing a DP rank records the
same rows; each DP rank counts once, and only if its copies agree. Hashes
cover the batches the loader handed the trainer, not completed optimizer
updates.
