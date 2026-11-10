# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""What runs inside each training container: torchrun around the adapter's trainer.

:func:`train_node` is registered once per placement by :class:`ElasticJob`,
under a different Modal function name each time. It stays a plain
module-level function so the container can import it by name; everything
launch-specific arrives as a :class:`RunSpec`.

A launch is one Modal call. Modal restarts a preempted call on new
containers, so a launch has one or more *attempts*; every node of a cluster
belongs to the same attempt. When the reconciler runs the launch, each
container publishes an attempt record to the job's registry (a Modal Dict),
which the reconciler decides from: written when it starts, refreshed every
:data:`SYNC_SECONDS`, and marked ``ended`` only after its last Volume commit
has returned.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import signal
import subprocess
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from zephon_modal.adapters.base import Adapter
from zephon_modal.bootstrap import coordinate
from zephon_modal.checkpoints import discard_incomplete
from zephon_modal.context import LaunchContext
from zephon_modal.data import DataContract
from zephon_modal.placement import Placement
from zephon_modal.verify.hashes import SAMPLE_HASH_SUBDIR

RENDEZVOUS_PORT = 29500
# Hash files are written per step on local disk (appending to an open file on
# a Volume duplicated its contents on background commits) and copied to the
# run directory every SYNC_SECONDS, so Modal's background commits make them
# durable before a stopped container's final commit, which may not finish.
LOCAL_SAMPLE_HASH_DIR = "/tmp/zephon-modal/sample-hashes"
SYNC_SECONDS = 30.0
CHECKPOINT_SUBDIR = "checkpoint"
MANIFEST_FILE = "run.json"


@dataclass(frozen=True)
class RunSpec:
    """Everything one launch needs, pickled into the Modal call.

    Attributes:
        registry_name: The job's run registry, when the reconciler started
            this launch. Node 0 checks that the launch is still the run's
            current one before touching the run directory.
        attempt_tag: Set by the launch itself: a fresh, time-ordered suffix
            per attempt, shared by every node, so restarts never mix their
            sample hashes and the latest attempt's sort last.
    """

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
    registry_name: str | None = None
    attempt_tag: str = ""

    @property
    def run_dir(self) -> str:
        return f"{self.runs_mount}/{self.run_id}"


@dataclass(frozen=True)
class NodeTopology:
    """Where this container sits in the launch."""

    nnodes: int
    node_rank: int
    master_addr: str | None
    # GPUs this node got, when the placement has fallbacks (otherwise its own count).
    gpus_per_node: int | None = None

    def torchrun_args(self) -> list[str]:
        if self.nnodes == 1:
            return ["--rdzv_backend=c10d", "--rdzv_endpoint=localhost:0"]
        return [
            f"--nnodes={self.nnodes}",
            f"--node-rank={self.node_rank}",
            f"--master-addr={self.master_addr}",
            f"--master-port={RENDEZVOUS_PORT}",
        ]


def allocated_shape(spec: RunSpec, topology: NodeTopology) -> Placement:
    """The shape this launch got: the placement itself, or the fallback Modal allocated."""
    if topology.gpus_per_node is None:
        return spec.placement
    return spec.placement.allocated(topology.gpus_per_node)


def launch_context(spec: RunSpec, topology: NodeTopology) -> LaunchContext:
    """The context this node's trainer processes see."""
    shape = allocated_shape(spec, topology)
    return LaunchContext(
        run_id=spec.run_id,
        run_dir=spec.run_dir,
        steps=spec.steps,
        seed=spec.seed,
        launch_id=spec.launch_id + spec.attempt_tag,
        placement=shape.name,
        nnodes=topology.nnodes,
        node_rank=topology.node_rank,
        nproc_per_node=shape.gpus_per_node,
        sample_hash_dir=local_hash_dir(spec),
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


def _visible_gpus() -> int:
    """GPUs in this container; Modal doesn't say which fallback it allocated."""
    listing = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
    return sum(line.startswith("GPU ") for line in listing.splitlines())


def node_topology(placement: Placement) -> NodeTopology:
    """This container's place in the launch, from Modal's cluster info on multi-node placements."""
    if placement.nodes == 1:
        gpus = _visible_gpus() if placement.fallbacks else None
        return NodeTopology(nnodes=1, node_rank=0, master_addr=None, gpus_per_node=gpus)
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


def new_launch_id() -> str:
    """Sortable, so the latest launch's record wins for a re-trained step.

    Made before the launch and passed in its spec, so every node of a cluster shares it.
    """
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + "-" + uuid.uuid4().hex[:8]


def task_id() -> str:
    """This container's Modal task ID; a restarted input gets a new one."""
    return os.environ.get("MODAL_TASK_ID", "local")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# The heartbeat thread and the main thread both sync hash files.
_write_lock = threading.Lock()


def _write_atomically(path: str, content: str) -> None:
    """Replace ``path`` in one rename, so this container never sees half a file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp-{task_id()}"
    with open(tmp_path, "w") as tmp_file:
        tmp_file.write(content)
    os.replace(tmp_path, path)


def _registry(spec: RunSpec) -> Any:
    from zephon_modal.registry import Registry

    return Registry.from_name(spec.registry_name) if spec.registry_name else None


def publish_attempt(spec: RunSpec, ctx: LaunchContext, record: Mapping[str, Any], attempts: int = 1) -> None:
    """Publish this container's attempt record to the job's registry, when the reconciler runs this launch.

    Retries up to ``attempts`` times, backing off, then raises.
    """
    registry = _registry(spec)
    if registry is None:
        return
    full = {"task_id": task_id(), "node_rank": ctx.node_rank, **record}
    for attempt in range(attempts - 1):
        if _try_publish(registry, spec, full):
            return
        time.sleep(2**attempt)
    registry.record(spec.run_id, spec.launch_id, task_id(), full)


def _try_publish(registry: Any, spec: RunSpec, record: dict[str, Any]) -> bool:
    try:
        registry.record(spec.run_id, spec.launch_id, task_id(), record)
    except Exception as error:
        print(f"[zephon_modal] couldn't publish the attempt record, retrying: {error!r}", flush=True)
        return False
    return True


def local_hash_dir(spec: RunSpec) -> str:
    """An attempt-specific local directory; warm containers may execute other runs."""
    run_key = hashlib.sha256(spec.run_dir.encode()).hexdigest()[:16]
    return f"{LOCAL_SAMPLE_HASH_DIR}/{run_key}/{spec.launch_id}{spec.attempt_tag}"


def check_manifest(spec: RunSpec) -> None:
    """Write ``run.json`` on a run's first launch; refuse a resume whose data settings differ.

    Only what decides the samples: the seed and the data contract's settings
    (for Zephon, the canonical replica count). Everything else, from the
    trainer's flags to this library, may change between launches.
    """
    manifest = json.loads(
        json.dumps({"seed": spec.seed, "data": {"contract": type(spec.data).__qualname__, **spec.data.context()}})
    )
    path = f"{spec.run_dir}/{MANIFEST_FILE}"
    if os.path.exists(path):
        with open(path) as manifest_file:
            existing = json.load(manifest_file)
        if existing != manifest:
            raise ValueError(f"{spec.run_id} was started with {existing}, not {manifest}; use a new run ID")
        return
    _write_atomically(path, json.dumps(manifest, indent=2))


def superseded(spec: RunSpec) -> bool:
    """Whether the reconciler has moved on from this launch (never, without a reconciler).

    Raises whatever reading the registry raises: a failed read isn't evidence.
    """
    registry = _registry(spec)
    if registry is None:
        return False
    current = getattr(registry.state(spec.run_id), "current", None) or {}
    return current.get("launch_id") != spec.launch_id


def _check_current(spec: RunSpec) -> None:
    """Refuse to run a launch the reconciler has moved on from."""
    if superseded(spec):
        raise RuntimeError(f"Launch {spec.launch_id} is no longer {spec.run_id}'s current launch")


def prepare_resume(spec: RunSpec) -> dict[str, Any]:
    """On node 0, set aside incomplete checkpoints; returns what the resume will load."""
    layout = spec.adapter.checkpoints
    if layout is None:
        return {}
    cleanup = discard_incomplete(layout, spec.run_dir, tag=task_id())
    if cleanup.discarded:
        print(f"[zephon_modal] set aside incomplete checkpoints {cleanup.discarded}; resuming from {cleanup.latest}")
    return {"resume_from": cleanup.latest, "discarded_checkpoints": cleanup.discarded}


def bootstrap_attempt(spec: RunSpec, topology: NodeTopology) -> tuple[RunSpec, dict[str, Any]]:
    """Give this attempt its tag and make every node resume from the same checkpoint.

    Node 0 checks the launch is current, checks the manifest, sets aside
    incomplete checkpoints and commits; then every other node reloads its
    view of the Volume and confirms it sees the same latest checkpoint
    before any trainer starts.
    """
    import modal

    volume = modal.Volume.from_name(spec.runs_volume_name)

    def prepare() -> dict[str, Any]:
        volume.reload()
        _check_current(spec)
        check_manifest(spec)
        # Time-ordered, so a restart's hashes sort after the attempt it replaces.
        tag = f".a{datetime.now(timezone.utc):%Y%m%dT%H%M%S%f}-{uuid.uuid4().hex[:6]}"
        state = {"attempt_tag": tag, **prepare_resume(spec)}
        volume.commit()
        return state

    def refresh(state: dict[str, Any]) -> None:
        volume.reload()
        layout = spec.adapter.checkpoints
        if layout is not None and layout.latest_complete(spec.run_dir) != state.get("resume_from"):
            raise RuntimeError("Cluster nodes disagree on the checkpoint to resume from")

    state = coordinate(
        key=spec.launch_id,
        nodes=topology.nnodes,
        rank=topology.node_rank,
        master_addr=topology.master_addr,
        prepare=prepare,
        refresh=refresh,
    )
    return dataclasses.replace(spec, attempt_tag=state["attempt_tag"]), state


_synced: dict[str, tuple[int, float]] = {}


def sync_sample_hashes(spec: RunSpec) -> None:
    """Copy this node's changed hash files into the run directory, complete lines only."""
    local_dir = local_hash_dir(spec)
    if not os.path.isdir(local_dir):
        return
    with _write_lock:
        for name in sorted(os.listdir(local_dir)):
            source = f"{local_dir}/{name}"
            stat = os.stat(source)
            if _synced.get(source) == (stat.st_size, stat.st_mtime):
                continue
            with open(source) as source_file:
                content = source_file.read()
            _write_atomically(f"{spec.run_dir}/{SAMPLE_HASH_SUBDIR}/{name}", content[: content.rfind("\n") + 1])
            _synced[source] = (stat.st_size, stat.st_mtime)


def _heartbeat(
    spec: RunSpec,
    ctx: LaunchContext,
    attempt: dict[str, Any],
    done: threading.Event,
    supersede: Callable[[], None] | None = None,
) -> None:
    """Sync hashes, refresh the attempt record and check the launch is still current, until the trainer stops.

    The launch checks it's current before it starts, but the reconciler can
    move on while it runs: after an operator's ``recover``, or a spawn that
    failed on our side but not Modal's. Then ``supersede`` stops the trainer,
    so two launches overlap for at most a beat and the trainer's shutdown.

    Errors are logged, never fatal: a dead heartbeat would get a healthy
    launch cancelled as unresponsive, and a failed read isn't a reason to stop.
    """
    while not done.wait(SYNC_SECONDS):
        _beat(spec, ctx, attempt, supersede)


def _beat(
    spec: RunSpec, ctx: LaunchContext, attempt: dict[str, Any], supersede: Callable[[], None] | None = None
) -> None:
    try:
        sync_sample_hashes(spec)
        publish_attempt(spec, ctx, dict(attempt) | {"heartbeat": _now()})
        if supersede is not None and superseded(spec):
            print(f"[zephon_modal] launch {spec.launch_id} is no longer {spec.run_id}'s current launch", flush=True)
            supersede()
    except Exception as error:
        print(f"[zephon_modal] heartbeat failed, retrying in {SYNC_SECONDS:.0f}s: {error!r}", flush=True)


def start_trainer(spec: RunSpec, topology: NodeTopology) -> subprocess.Popen:
    """Start torchrun for this node."""
    ctx = launch_context(spec, topology)
    os.makedirs(spec.run_dir, exist_ok=True)
    # Unbuffered: torchrun sends worker output through files, and buffered
    # output reaches the logs in late chunks, or not at all if the trainer is killed.
    env = os.environ | {"PYTHONUNBUFFERED": "1"} | spec.adapter.env(ctx) | _compile_cache_env(spec) | ctx.to_env()
    command = torchrun_command(spec.adapter, ctx, topology)
    print(f"[zephon_modal] {spec.run_id} launch {ctx.launch_id} on {ctx.placement}: {' '.join(command)}")
    # Its own session: Modal's SIGINT goes to the whole process group, and the
    # runner, not every rank at once, decides how the trainer stops.
    return subprocess.Popen(command, cwd=spec.adapter.workdir(), env=env, start_new_session=True)


def stop_trainer(trainer: subprocess.Popen, adapter: Adapter) -> int | None:
    """Send torchrun the adapter's shutdown signal; kill the trainer if it outlasts ``shutdown_timeout``.

    Returns torchrun's exit code, or ``None`` if it had to be killed.
    """
    started = time.monotonic()
    print(f"[zephon_modal] stopping the trainer with {adapter.shutdown_signal.name}")
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
    """Commit the Volumes; returns the checkpoint directories present."""
    import modal

    modal.Volume.from_name(spec.runs_volume_name).commit()
    if spec.compile_cache_volume_name is not None:
        modal.Volume.from_name(spec.compile_cache_volume_name).commit()
    checkpoint_dir = f"{spec.run_dir}/{CHECKPOINT_SUBDIR}"
    return sorted(os.listdir(checkpoint_dir)) if os.path.isdir(checkpoint_dir) else []


def train_node(spec: RunSpec) -> list[str]:
    """Run one node of a launch; returns the checkpoint directories present afterwards.

    If Modal stops the container (``KeyboardInterrupt`` here) or the call is
    cancelled (``InputCancellation``), the trainer is stopped and the Volume
    committed before the exception propagates: Modal restarts a stopped
    input, and a cancelled one ends. Modal gives a grace period for all of it
    (about 30 seconds in our tests; its docs don't give a length), so hashes
    are synced first and the trainer gets ``shutdown_timeout``.
    """
    from modal.exception import InputCancellation

    topology = node_topology(spec.placement)
    ctx = launch_context(spec, topology)
    attempt: dict[str, Any] = {
        "started": _now(),
        "placement": spec.placement.name,
        "allocated": ctx.placement,
        "outcome": "starting",
    }
    trainer: subprocess.Popen | None = None
    done, was_superseded = threading.Event(), threading.Event()
    heartbeat: threading.Thread | None = None
    # The heartbeat may stop the trainer too: one stop at a time, and whoever comes second waits for it.
    stopping = threading.Lock()

    def stop_trainer_once() -> None:
        with stopping:
            if trainer is not None and trainer.poll() is None:
                attempt["returncode"] = stop_trainer(trainer, spec.adapter)

    def supersede() -> None:
        was_superseded.set()
        stop_trainer_once()

    try:
        # Before anything touches the run, so the reconciler waits for this container.
        publish_attempt(spec, ctx, attempt)
        spec, resume = bootstrap_attempt(spec, topology)
        ctx = launch_context(spec, topology)
        attempt |= resume | {"outcome": "running"}
        trainer = start_trainer(spec, topology)
        publish_attempt(spec, ctx, attempt)
        heartbeat = threading.Thread(target=_heartbeat, args=(spec, ctx, attempt, done, supersede), daemon=True)
        heartbeat.start()
        returncode = trainer.wait()
        if was_superseded.is_set():
            attempt["outcome"] = "superseded"
            raise RuntimeError(f"Launch {spec.launch_id} is no longer {spec.run_id}'s current launch")
        attempt |= {"outcome": "completed" if returncode == 0 else "failed", "returncode": returncode}
        if returncode:
            raise subprocess.CalledProcessError(returncode, trainer.args)
    except (KeyboardInterrupt, InputCancellation) as stop:
        # One stop is enough; ignore repeats while cleaning up (Modal cancels with SIGUSR1).
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGUSR1, signal.SIG_IGN)
        attempt["outcome"] = "cancelled" if isinstance(stop, InputCancellation) else "stopped"
        # Every step hashed so far, before anything slow: the final commit may not finish.
        sync_sample_hashes(spec)
        raise
    except BaseException:
        if attempt["outcome"] in ("starting", "running"):
            attempt["outcome"] = "failed"
        raise
    finally:
        stop_trainer_once()
        done.set()
        if heartbeat is not None:
            heartbeat.join(timeout=2)
        sync_sample_hashes(spec)
        if spec.adapter.checkpoints is not None and ctx.node_rank == 0:
            try:
                attempt["latest_checkpoint"] = spec.adapter.checkpoints.latest_complete(spec.run_dir)
            except Exception as error:
                # Only a note for the record; the final commit below matters more.
                print(f"[zephon_modal] couldn't read the latest checkpoint: {error!r}", flush=True)
        checkpoints = finish_launch(spec)
        # Only now: "ended" tells the reconciler everything this container wrote is committed.
        publish_attempt(spec, ctx, attempt | {"ended": _now()}, attempts=4)
    return checkpoints


def replays(runs_mount: str, runs_volume_name: str, run_id: str) -> list[dict]:
    """Check one run's replayed steps against each other, on the runs Volume."""
    import modal

    from zephon_modal.verify.compare import check_replays

    modal.Volume.from_name(runs_volume_name).reload()
    return check_replays(f"{runs_mount}/{run_id}")


def compare(
    runs_mount: str,
    runs_volume_name: str,
    reference: str,
    candidate: str,
    steps: int | None = None,
    start_step: int = 1,
    rows_per_step: int | None = None,
) -> list[dict]:
    """Compare two runs' sample hashes on the runs Volume."""
    import modal

    from zephon_modal.verify.compare import compare_runs

    modal.Volume.from_name(runs_volume_name).reload()
    return compare_runs(
        f"{runs_mount}/{reference}",
        f"{runs_mount}/{candidate}",
        steps=steps,
        start_step=start_step,
        rows_per_step=rows_per_step,
    )
