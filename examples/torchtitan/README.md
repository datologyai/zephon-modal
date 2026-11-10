# torchtitan example: Qwen3-0.6B on H100s

An elastic Qwen3-0.6B run on [torchtitan-zephon](https://github.com/datologyai/torchtitan-zephon),
reading FineWeb-Edu `sample-10BT` through Zephon. It moves between a 2-node
H100:8 cluster (DP 16) and one node with H100:8 or H100:4 (DP 8 or 4), and
trains on the same samples at every step whichever it's on.

- `job.py`: the `ElasticJob`, its placements and `CANONICAL_REPLICAS = 16`.
- `recipe.py`: the torchtitan config (4096-token sequences, 128 per step,
  a checkpoint every 10 steps).
- `image.py`: CUDA 13.0, torchtitan-zephon cloned at a pinned commit, Zephon
  0.2.1, then nightly cu130 torch.
- `storage.py`: the Volumes. `stage_data.py`: stages the data.

## Settings

Every command reads these when it loads `job.py`, so export them in the shell
you run everything from. App and Volume names are shared across a Modal
workspace: prefix them with your name.

| Variable | Default | What |
|---|---|---|
| `ZEPHON_MODAL_TORCHTITAN_PREFIX` | `zephon-modal-torchtitan` | App name; the run registry is `<prefix>-runs` |
| `ZEPHON_MODAL_TORCHTITAN_DATA_VOLUME` | `zephon-modal-torchtitan-data` | Staged data |
| `ZEPHON_MODAL_TORCHTITAN_RUNS_VOLUME` | `zephon-modal-torchtitan-runs` | Checkpoints, hashes, run state |
| `ZEPHON_MODAL_TORCHTITAN_COMPILE_CACHE_VOLUME` | `zephon-modal-torchtitan-compile-cache` | torch.compile caches |
| `ZEPHON_MODAL_TORCHTITAN_COMMIT` | `990bd303f9a5380fb1ca93ed075b03d810c55e13` | torchtitan-zephon commit to build |
| `ZEPHON_MODAL_TORCHTITAN_ALERT_SECRET` | unset | Modal secret providing `ZEPHON_MODAL_ALERT_WEBHOOK` |

```bash
export ZEPHON_MODAL_TORCHTITAN_PREFIX=<you>-tt
export ZEPHON_MODAL_TORCHTITAN_DATA_VOLUME=<you>-tt-data
export ZEPHON_MODAL_TORCHTITAN_RUNS_VOLUME=<you>-tt-runs
export ZEPHON_MODAL_TORCHTITAN_COMPILE_CACHE_VOLUME=<you>-tt-compile-cache
```

Multi-node clusters must be enabled on your Modal workspace for the
`2xh100x8` placement.

The same recipe and image definition support the default pre-sync commit
and the upstream sync in [torchtitan-zephon #12](https://github.com/datologyai/torchtitan-zephon/pull/12).
To try the sync, export its full commit SHA before running any command:

```bash
export ZEPHON_MODAL_TORCHTITAN_COMMIT=c6a084110448d830b6cc0773afe36fca34c4b20f
```

Changing the commit builds a separate cached image. Each checkout's own
requirements select its `torch_checkpointing` dependency: the pre-sync
checkout pins a Git commit, and the sync pins version 0.1.1. The image adds
no checkpointing override. The recipe sets Flex attention explicitly to
preserve its uncapped packed data stream across both versions, and sets
`training.max_context_length` only on versions that still expose it.

## Stage the data, once

```bash
uv run modal run examples/torchtitan/stage_data.py --num-shards 2
```

Two shards (~1.4B tokens, about 2,700 steps at 128 × 4096 tokens per step)
is what the references were trained on. For a longer run, stage enough
before starting it: changing the data under a run changes the samples its
later steps see.

## Run

```bash
uv run python -m zephon_modal placements examples/torchtitan/job.py
uv run python -m zephon_modal start examples/torchtitan/job.py --run-id <you>-qwen3 --steps 2000 --keep-checkpoints 3
uv run python -m zephon_modal status examples/torchtitan/job.py --run-id <you>-qwen3
uv run python -m zephon_modal stop examples/torchtitan/job.py --run-id <you>-qwen3
```

- `placements` shows each placement and the DP degree it gives.
- `start` deploys `<prefix>` and hands the run to its reconciler, which
  launches on the cluster first, moves to one node when it must, and resumes
  from the latest checkpoint each time. `--steps` is the total to reach.
  Starting a stopped or finished run again resumes it.
- `status` shows the reconciler's view: the current launch, its placement,
  and recent events.
- `stop` stops the run; its launch stops the trainer and commits.

**Keep checkpoints with `--keep-checkpoints`, not in the trainer.** The
recipe keeps every checkpoint (`keep_latest_k=0`): on the cluster each node
sees only its own uncommitted Volume files, so a trainer pruning there leaves
other nodes' shards behind as partial checkpoints. The reconciler prunes from
its committed view instead. Each checkpoint, with optimizer state, is about
8 GB.

The deployed app keeps its reconciler schedule until you stop it:

```bash
modal app stop --yes <you>-tt
```

## Check it trained on the right samples

Make a reference with `launch`, which trains on one placement now, without
the reconciler. Any placement works: `compare` treats a step as a multiset of
sample hashes, so runs at different DP degrees match when they saw the same
data.

```bash
uv run python -m zephon_modal launch examples/torchtitan/job.py --run-id <you>-ref-200 --steps 200 --placement h100x8-or-h100x4
uv run python -m zephon_modal compare examples/torchtitan/job.py --reference <you>-ref-200 --candidate <you>-qwen3 --steps 200 --rows-per-step 128
uv run python -m zephon_modal replays examples/torchtitan/job.py --run-id <you>-qwen3
```

With `--steps`, `compare` is strict: every step through 200 must be in both
runs, from every DP rank, with 128 sequences, and match. `replays` needs no
reference: the steps a resume re-trained since the last checkpoint, often at
another DP degree, must match their first training.

## Rehearse preemptions

`stop-after` stops the current launch's container shortly after the first
complete checkpoint at or past each given step, as a preemption would. On the
cluster, stopping node 0 stops both nodes and Modal restarts them together.
With `--max-restarts 0` every stop becomes a move to the next placement:

```bash
uv run python -m zephon_modal start examples/torchtitan/job.py --run-id <you>-rehearsal --steps 200 \
    --max-restarts 0 --cooldown 600 --keep-checkpoints 3
uv run python -m zephon_modal stop-after examples/torchtitan/job.py --run-id <you>-rehearsal --after 40 120
uv run python -m zephon_modal compare examples/torchtitan/job.py --reference <you>-ref-200 --candidate <you>-rehearsal --steps 200 --rows-per-step 128
uv run python -m zephon_modal replays examples/torchtitan/job.py --run-id <you>-rehearsal
```

These are injected stops, not real preemptions; see [docs/design.md](../../docs/design.md#rehearsing-preemptions-with-stop-after).
