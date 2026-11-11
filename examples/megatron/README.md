# Megatron example: Qwen3-0.6B with Zephon

Train Qwen3-0.6B from scratch on [Megatron-LM-zephon](https://github.com/datologyai/Megatron-LM-zephon),
using its Hugging Face tokenizer and FineWeb-Edu `sample-10BT` through Zephon.
Packed documents have separate attention boundaries and reset position IDs,
matching the TorchTitan example's document isolation.
The job can move between two H100:8 nodes and one H100:8 or H100:4 node.
A separate `smoke` job runs the same model on one H100 with short sequences
and a tiny generated dataset.

- `image.py` builds the training environment and fetches Megatron.
- `recipe.py` supplies model, data, optimizer and checkpoint arguments.
- `job.py` defines the `job` and `smoke` jobs, GPU choices and storage mounts.
- `storage.py` names the Volumes; `stage_data.py` prepares either dataset.

## Setup

From the repository root, install the client and authenticate with Modal:

```bash
uv sync --group dev
uv run modal setup
```

Use your own prefix and Volume names; Modal names are shared in a workspace.
Export the same settings for staging, training and verification commands.

```bash
export ZEPHON_MODAL_MEGATRON_PREFIX=<you>-megatron
export ZEPHON_MODAL_MEGATRON_DATA_VOLUME=<you>-megatron-data
export ZEPHON_MODAL_MEGATRON_RUNS_VOLUME=<you>-megatron-runs
```

| Variable | Default | Purpose |
|---|---|---|
| `ZEPHON_MODAL_MEGATRON_PREFIX` | `zephon-modal-megatron` | App name; the smoke app adds `-smoke` |
| `ZEPHON_MODAL_MEGATRON_DATA_VOLUME` | `zephon-modal-megatron-data` | Prepared datasets |
| `ZEPHON_MODAL_MEGATRON_RUNS_VOLUME` | `zephon-modal-megatron-runs` | Checkpoints, data state and optional hashes |
| `ZEPHON_MODAL_MEGATRON_COMMIT` | `7ce36bd201921ff78f798eb362d0aca97bdd6869` | Full Megatron commit SHA to build |
| `ZEPHON_MODAL_MEGATRON_BASE_IMAGE` | `nvcr.io/nvidia/pytorch:26.09-py3` | NVIDIA PyTorch image with Transformer Engine, Apex and FlashAttention |
| `ZEPHON_MODAL_MEGATRON_SAMPLE_HASHES` | `0` | Set to `1` to record sample fingerprints |
| `ZEPHON_MODAL_MEGATRON_ALERT_SECRET` | unset | Modal secret providing `ZEPHON_MODAL_ALERT_WEBHOOK` |

Changing the commit builds an image from that checkout and its Zephon
requirements. The base image supplies a compatible CUDA and PyTorch stack;
select a matching base image when choosing a different framework revision.

The default image uses CUDA 13.4. NVIDIA supports
[CUDA 13.x minor-version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
with R580 or newer host drivers, with limits on PTX and newer driver features.
NGC's startup check may warn when a host driver is older than the image's
bundled driver. Check the allocated hosts and run a short training/resume
check before a longer run; the warning alone does not identify which
compatibility mechanism is active. Keep the image's coupled CUDA, PyTorch
and Transformer Engine stack together.

## Short training and resume check

Generate a small Parquet dataset, then train two steps and resume the same
run through step three:

```bash
uv run modal run examples/megatron/stage_data.py --smoke
uv run python -m zephon_modal placements examples/megatron/job.py:smoke
uv run python -m zephon_modal launch examples/megatron/job.py:smoke --run-id <you>-smoke --steps 2 --detach
uv run python -m zephon_modal launch examples/megatron/job.py:smoke --run-id <you>-smoke --steps 3 --detach
```

Wait for each command to finish before starting the next. `--steps` is the
total target. The second launch restores model, optimizer, training progress
and Zephon state from the step-two checkpoint before training step three.
Use a new run ID to repeat the check from scratch.

The smoke job uses the full Qwen3-0.6B model, 128-token sequences, two
sequences per step, one Zephon lane and a checkpoint every two steps. It
needs no FineWeb download. Its checkpoints still contain full optimizer
state and occupy several GB.

## FineWeb training

Prepare data before starting a run. Each sample-10BT shard is about 2 GB;
two shards contain roughly 1.4B tokens, enough for about 2,700 steps at
128 × 4096 tokens per step. Keep a run's dataset unchanged across resumes.

```bash
uv run modal run examples/megatron/stage_data.py --num-shards 2
uv run python -m zephon_modal placements examples/megatron/job.py
uv run python -m zephon_modal start examples/megatron/job.py --run-id <you>-qwen3 --steps 2000 --keep-checkpoints 3
uv run python -m zephon_modal status examples/megatron/job.py --run-id <you>-qwen3
uv run python -m zephon_modal stop examples/megatron/job.py --run-id <you>-qwen3
```

If you already staged FineWeb-Edu for the TorchTitan example, set
`ZEPHON_MODAL_MEGATRON_DATA_VOLUME` to that data Volume's name and skip
`stage_data.py`. Both examples use `/data/fineweb-edu/sample-10BT` and read
the same Parquet shards and Zephon index without rewriting the data.

The two-node placement requires Modal multi-node support. The job uses 16
Zephon lanes and 64 global microbatches per optimizer step, so each step
consumes whole lane windows at DP 16, 8 or 4. Megatron saves model and
optimizer state in the `torch_dist` format, with Zephon state inside each
`checkpoint/iter_N` directory. A checkpoint is usable only after both the
model shards and its Zephon state are complete. `--keep-checkpoints` prunes
them together and preserves the latest resumable checkpoint during a save.
Startup also restores Megatron's text and training-state trackers from that
checkpoint, so either tracker selects the same complete save.

`start` leaves a scheduled reconciler deployed. After stopping your runs,
remove the deployed app when it is no longer needed:

```bash
uv run modal app stop --yes <you>-megatron
```

## Optional sample verification

Hashing is off by default. To check which data each step receives, enable
it before launching both a reference and a candidate run. It adds CPU work
to fingerprint tokens, labels, positions and masks; it does not change the
batches or enable Zephon's determinism.

For a short reference and resumed candidate on the generated dataset:

```bash
export ZEPHON_MODAL_MEGATRON_SAMPLE_HASHES=1
uv run python -m zephon_modal launch examples/megatron/job.py:smoke --run-id <you>-ref --steps 3 --detach
uv run python -m zephon_modal launch examples/megatron/job.py:smoke --run-id <you>-candidate --steps 2 --detach
uv run python -m zephon_modal launch examples/megatron/job.py:smoke --run-id <you>-candidate --steps 3 --detach
uv run python -m zephon_modal compare examples/megatron/job.py:smoke --reference <you>-ref --candidate <you>-candidate --steps 3 --rows-per-step 2
```

For the FineWeb job, use its job target and `--rows-per-step 128` instead.
`replays` checks steps repeated after an interruption against their earlier
attempts; a clean checkpoint resume may have no repeated steps to compare.
