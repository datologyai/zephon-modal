# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""What runs inside each training container: torchrun around the adapter's trainer.

:func:`train_node` is registered once per placement by :class:`ElasticJob`,
under a different Modal function name each time. It stays a plain
module-level function so the container can import it by name; everything
launch-specific arrives as a :class:`RunSpec`.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import threading
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from zephon_modal.adapters.base import Adapter
from zephon_modal.checkpoints import discard_incomplete
from zephon_modal.context import LaunchContext
from zephon_modal.data import DataContract
from zephon_modal.placement import Placement
from zephon_modal.verify.hashes import SAMPLE_HASH_SUBDIR

RENDEZVOUS_PORT = 29500
# Hash files are written per step on local disk (appending to an open file on
# a Volume duplicated its contents on background commits) and synced to the
# run directory every HASH_SYNC_SECONDS, so Modal's background commits make
# them durable before a stopped container's final commit, which may not finish.
LOCAL_SAMPLE_HASH_DIR = "/tmp/zephon-modal/sample-hashes"
HASH_SYNC_SECONDS = 30.0
CHECKPOINT_SUBDIR = "checkpoint"
LAUNCHES_SUBDIR = "launches"


@dataclass(frozen=True)
class RunSpec:
    """Everything one launch needs, pickled into the Modal call."""

    run_id: str
    steps: int
    seed: int
    launch_id: str
    placement: Placement
    adapter: Adapter
    data: DataContract
    runs_mount: str
    runs_volume_name: str
    compile_cache_mount: str | None = None
    compile_cache_volume_name: str | None = None

    @property
    def run_dir(self) -> str:
        return f"{self.runs_mount}/{self.run_id}"


@dataclass(frozen=True)
class NodeTopology:
    """Where this container sits in the launch."""

    nnodes: int
    node_rank: int
    master_addr: str | None

    def torchrun_args(self) -> list[str]:
        if self.nnodes == 1:
            return ["--rdzv_backend=c10d", "--rdzv_endpoint=localhost:0"]
        return [
            f"--nnodes={self.nnodes}",
            f"--node-rank={self.node_rank}",
            f"--master-addr={self.master_addr}",
            f"--master-port={RENDEZVOUS_PORT}",
        ]


def launch_context(spec: RunSpec, topology: NodeTopology) -> LaunchContext:
    """The context this node's trainer processes see."""
    return LaunchContext(
        run_id=spec.run_id,
        run_dir=spec.run_dir,
        steps=spec.steps,
        seed=spec.seed,
        launch_id=spec.launch_id,
        placement=spec.placement.name,
        nnodes=topology.nnodes,
        node_rank=topology.node_rank,
        nproc_per_node=spec.placement.gpus_per_node,
        sample_hash_dir=LOCAL_SAMPLE_HASH_DIR,
        data=spec.data.context(),
    )


def torchrun_command(adapter: Adapter, ctx: LaunchContext, topology: NodeTopology) -> list[str]:
    """The full torchrun invocation for this node."""
    return [
        "torchrun",
        f"--nproc_per_node={ctx.nproc_per_node}",
        *topology.torchrun_args(),
        f"--local-ranks-filter={adapter.local_ranks_filter(ctx.nproc_per_node)}",
        "--role=rank",
        "--tee=3",
        *adapter.command(ctx),
    ]


def node_topology(placement: Placement) -> NodeTopology:
    """This container's place in the launch, from Modal's cluster info on multi-node placements."""
    if placement.nodes == 1:
        return NodeTopology(nnodes=1, node_rank=0, master_addr=None)
    import modal

    cluster = modal.Cluster.from_context()
    return NodeTopology(
        nnodes=placement.nodes,
        node_rank=cluster.container_rank(),
        master_addr=cluster.container_ips(family="ipv4")[0],
    )


def _compile_cache_env(spec: RunSpec) -> dict[str, str]:
    if spec.compile_cache_mount is None:
        return {}
    import torch

    # Kernels autotuned on one GPU model aren't valid on another (H100 vs H200).
    gpu_name = torch.cuda.get_device_name(0).replace(" ", "_")
    cache_dir = f"{spec.compile_cache_mount}/{gpu_name}"
    return {"TORCHINDUCTOR_CACHE_DIR": f"{cache_dir}/inductor", "TRITON_CACHE_DIR": f"{cache_dir}/triton"}


def task_id() -> str:
    """This container's Modal task ID; a restarted input gets a new one."""
    return os.environ.get("MODAL_TASK_ID", "local")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _launch_dir(spec: RunSpec) -> str:
    return f"{spec.run_dir}/{LAUNCHES_SUBDIR}/{spec.launch_id}"


def _write_atomically(path: str, content: str) -> None:
    """Replace ``path`` in one rename, so a background commit sees the old file or the new one, never half of one."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp-{task_id()}"
    with open(tmp_path, "w") as tmp_file:
        tmp_file.write(content)
    os.replace(tmp_path, path)


def _write_json(path: str, record: Mapping[str, Any]) -> None:
    _write_atomically(path, json.dumps(record, indent=2))


def _record_launch(spec: RunSpec, ctx: LaunchContext) -> None:
    """``launches/<launch_id>/launch.json``: what was asked for. Restarts rewrite the same contents."""
    if ctx.node_rank != 0:
        return
    record = {
        "launch_id": spec.launch_id,
        "placement": asdict(spec.placement) | {"name": spec.placement.name},
        "world_size": ctx.world_size,
        "dp": spec.placement.dp_size(spec.adapter.model_parallel_size),
        "steps": spec.steps,
        "seed": spec.seed,
        "data": {"contract": type(spec.data).__qualname__, **spec.data.context()},
    }
    _write_json(f"{_launch_dir(spec)}/launch.json", record)


def record_attempt(spec: RunSpec, ctx: LaunchContext, record: Mapping[str, Any]) -> None:
    """``launches/<launch_id>/<task_id>.json``: how one container's attempt went.

    Modal restarts a preempted input on a new container with the same
    launch ID, so a launch can have several attempts, each with one file
    per node.
    """
    _write_json(f"{_launch_dir(spec)}/{task_id()}.json", {"task_id": task_id(), "node_rank": ctx.node_rank, **record})


def prepare_resume(spec: RunSpec, ctx: LaunchContext) -> dict[str, Any]:
    """On node 0, set aside checkpoints a stopped container left incomplete; returns what the resume will load.

    Runs before torchrun starts, and the trainers can't load until every
    node has joined the rendezvous, so no node reads a checkpoint while it's moved.
    """
    layout = spec.adapter.checkpoints
    if ctx.node_rank != 0 or layout is None:
        return {}
    cleanup = discard_incomplete(layout, spec.run_dir, tag=task_id())
    if cleanup.discarded:
        print(f"[zephon_modal] set aside incomplete checkpoints {cleanup.discarded}; resuming from {cleanup.latest}")
    return {"resume_from": cleanup.latest, "discarded_checkpoints": cleanup.discarded}


def _restore_sample_hashes(spec: RunSpec, ctx: LaunchContext) -> None:
    """Bring back this node's hash files from an earlier attempt of the same launch.

    A restarted attempt keeps the launch ID and so the file names; without
    this, its copy at exit would replace the earlier attempt's steps. The
    trainer appends to these, and :func:`~zephon_modal.verify.compare.load_step_digests`
    keeps a file's last record for each step.
    """
    run_hash_dir = f"{spec.run_dir}/{SAMPLE_HASH_SUBDIR}"
    if not os.path.isdir(run_hash_dir):
        return
    first_rank = ctx.node_rank * ctx.nproc_per_node
    for name in os.listdir(run_hash_dir):
        match = re.fullmatch(rf"{re.escape(spec.launch_id)}-dp\d+-rank(\d+)\.jsonl", name)
        if match is None or not first_rank <= int(match.group(1)) < first_rank + ctx.nproc_per_node:
            continue
        with open(f"{run_hash_dir}/{name}") as hash_file:
            content = hash_file.read()
        # A kill mid-write can leave half a line.
        content = content[: content.rfind("\n") + 1]
        os.makedirs(LOCAL_SAMPLE_HASH_DIR, exist_ok=True)
        with open(f"{LOCAL_SAMPLE_HASH_DIR}/{name}", "w") as local_file:
            local_file.write(content)


_hash_sync_lock = threading.Lock()


def sync_sample_hashes(spec: RunSpec) -> None:
    """Copy this node's hash files into the run directory, complete lines only."""
    if not os.path.isdir(LOCAL_SAMPLE_HASH_DIR):
        return
    # The periodic sync and the final one share temp file names.
    with _hash_sync_lock:
        for name in os.listdir(LOCAL_SAMPLE_HASH_DIR):
            with open(f"{LOCAL_SAMPLE_HASH_DIR}/{name}") as hash_file:
                content = hash_file.read()
            _write_atomically(f"{spec.run_dir}/{SAMPLE_HASH_SUBDIR}/{name}", content[: content.rfind("\n") + 1])


def _sync_sample_hashes_until(spec: RunSpec, done: threading.Event) -> None:
    while not done.wait(HASH_SYNC_SECONDS):
        sync_sample_hashes(spec)


def start_trainer(
    spec: RunSpec, topology: NodeTopology, extra_env: Mapping[str, str] | None = None
) -> subprocess.Popen:
    """Record the launch and start torchrun for this node.

    Pair with :func:`finish_launch` once the trainer exits. ``extra_env``
    reaches every trainer process (e.g. profiling switches).
    """
    ctx = launch_context(spec, topology)
    os.makedirs(spec.run_dir, exist_ok=True)
    _record_launch(spec, ctx)
    _restore_sample_hashes(spec, ctx)
    # Unbuffered: torchrun sends worker output through files, and buffered
    # output reaches the logs in late chunks, or not at all if the trainer is killed.
    env = (
        os.environ
        | {"PYTHONUNBUFFERED": "1"}
        | spec.adapter.env(ctx)
        | _compile_cache_env(spec)
        | dict(extra_env or {})
        | ctx.to_env()
    )
    command = torchrun_command(spec.adapter, ctx, topology)
    print(f"[zephon_modal] {spec.run_id} launch {spec.launch_id} on {spec.placement.name}: {' '.join(command)}")
    # Its own session: Modal's SIGINT goes to the whole process group, and the
    # runner, not every rank at once, decides how the trainer stops.
    return subprocess.Popen(command, cwd=spec.adapter.workdir(), env=env, start_new_session=True)


def stop_trainer(trainer: subprocess.Popen, adapter: Adapter) -> int | None:
    """Send torchrun the adapter's shutdown signal; kill the trainer if it outlasts ``shutdown_timeout``.

    Returns torchrun's exit code, or ``None`` if it had to be killed.
    """
    started = time.monotonic()
    print(f"[zephon_modal] container stopping: sending {adapter.shutdown_signal.name} to the trainer")
    trainer.send_signal(adapter.shutdown_signal)
    try:
        returncode = trainer.wait(timeout=adapter.shutdown_timeout)
    except subprocess.TimeoutExpired:
        print(f"[zephon_modal] trainer still running after {adapter.shutdown_timeout:.0f}s; killing it")
        os.killpg(trainer.pid, signal.SIGKILL)
        trainer.wait()
        return None
    print(f"[zephon_modal] trainer exited with {returncode} after {time.monotonic() - started:.1f}s")
    return returncode


def finish_launch(spec: RunSpec) -> list[str]:
    """Move sample hashes into the run and commit Volumes; returns the checkpoint steps present.

    Call it whether or not training succeeded, so the next container sees
    whatever checkpoints exist.
    """
    import modal

    sync_sample_hashes(spec)
    modal.Volume.from_name(spec.runs_volume_name).commit()
    if spec.compile_cache_volume_name is not None:
        modal.Volume.from_name(spec.compile_cache_volume_name).commit()
    checkpoint_dir = f"{spec.run_dir}/{CHECKPOINT_SUBDIR}"
    return sorted(os.listdir(checkpoint_dir)) if os.path.isdir(checkpoint_dir) else []


def train_node(spec: RunSpec) -> list[str]:
    """Run one node of a launch; returns the checkpoint steps present afterwards.

    If Modal stops the container (it raises ``KeyboardInterrupt`` here), the
    trainer is stopped, the Volume committed, and the interrupt re-raised so
    Modal restarts the input.
    """
    topology = node_topology(spec.placement)
    ctx = launch_context(spec, topology)
    attempt: dict[str, Any] = {"started": _now(), "placement": spec.placement.name, **prepare_resume(spec, ctx)}
    trainer = start_trainer(spec, topology)
    syncing_done = threading.Event()
    threading.Thread(target=_sync_sample_hashes_until, args=(spec, syncing_done), daemon=True).start()
    try:
        returncode = trainer.wait()
        attempt |= {"outcome": "completed" if returncode == 0 else "failed", "returncode": returncode}
    except KeyboardInterrupt:
        # One stop is enough; ignore any repeat while we clean up.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        # Every step hashed so far, before anything slow: the final commit may not finish.
        sync_sample_hashes(spec)
        attempt |= {"outcome": "stopped", "returncode": stop_trainer(trainer, spec.adapter)}
        raise
    finally:
        syncing_done.set()
        attempt["ended"] = _now()
        if spec.adapter.checkpoints is not None and ctx.node_rank == 0:
            attempt["latest_checkpoint"] = spec.adapter.checkpoints.latest_complete(spec.run_dir)
        record_attempt(spec, ctx, attempt)
        checkpoints = finish_launch(spec)
    if returncode:
        raise subprocess.CalledProcessError(returncode, trainer.args)
    return checkpoints


def compare(runs_mount: str, runs_volume_name: str, reference: str, candidate: str) -> list[dict]:
    """Compare two runs' sample hashes on the runs Volume."""
    import modal

    from zephon_modal.verify.compare import compare_runs

    modal.Volume.from_name(runs_volume_name).reload()
    return compare_runs(f"{runs_mount}/{reference}", f"{runs_mount}/{candidate}")
