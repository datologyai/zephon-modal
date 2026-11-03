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
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import asdict, dataclass

from zephon_modal.adapters.base import Adapter
from zephon_modal.context import LaunchContext
from zephon_modal.data import DataContract
from zephon_modal.placement import Placement
from zephon_modal.verify.hashes import SAMPLE_HASH_SUBDIR

RENDEZVOUS_PORT = 29500
# Hash files are written per step; they live on local disk and are copied to
# the run directory when the launch ends.
LOCAL_SAMPLE_HASH_DIR = "/tmp/zephon-modal/sample-hashes"
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


def _record_launch(spec: RunSpec, ctx: LaunchContext) -> None:
    """One file per launch (never appended to; see LOCAL_SAMPLE_HASH_DIR)."""
    if ctx.node_rank != 0:
        return
    launches_dir = f"{spec.run_dir}/{LAUNCHES_SUBDIR}"
    os.makedirs(launches_dir, exist_ok=True)
    record = {
        "launch_id": spec.launch_id,
        "placement": asdict(spec.placement) | {"name": spec.placement.name},
        "world_size": ctx.world_size,
        "dp": spec.placement.dp_size(spec.adapter.model_parallel_size),
        "steps": spec.steps,
        "seed": spec.seed,
        "data": {"contract": type(spec.data).__qualname__, **spec.data.context()},
    }
    with open(f"{launches_dir}/{spec.launch_id}.json", "w") as launch_file:
        json.dump(record, launch_file, indent=2)


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
    env = os.environ | spec.adapter.env(ctx) | _compile_cache_env(spec) | dict(extra_env or {}) | ctx.to_env()
    command = torchrun_command(spec.adapter, ctx, topology)
    print(f"[zephon_modal] {spec.run_id} launch {spec.launch_id} on {spec.placement.name}: {' '.join(command)}")
    return subprocess.Popen(command, cwd=spec.adapter.workdir(), env=env)


def finish_launch(spec: RunSpec) -> list[str]:
    """Move sample hashes into the run and commit Volumes; returns the checkpoint steps present.

    Call it whether or not training succeeded, so the next container sees
    whatever checkpoints exist.
    """
    import modal

    if os.path.isdir(LOCAL_SAMPLE_HASH_DIR):
        shutil.copytree(LOCAL_SAMPLE_HASH_DIR, f"{spec.run_dir}/{SAMPLE_HASH_SUBDIR}", dirs_exist_ok=True)
    modal.Volume.from_name(spec.runs_volume_name).commit()
    if spec.compile_cache_volume_name is not None:
        modal.Volume.from_name(spec.compile_cache_volume_name).commit()
    checkpoint_dir = f"{spec.run_dir}/{CHECKPOINT_SUBDIR}"
    return sorted(os.listdir(checkpoint_dir)) if os.path.isdir(checkpoint_dir) else []


def train_node(spec: RunSpec) -> list[str]:
    """Run one node of a launch; returns the checkpoint steps present afterwards."""
    trainer = start_trainer(spec, node_topology(spec.placement))
    try:
        if trainer.wait():
            raise subprocess.CalledProcessError(trainer.returncode, trainer.args)
    finally:
        checkpoints = finish_launch(spec)
    return checkpoints


def compare(runs_mount: str, runs_volume_name: str, reference: str, candidate: str) -> list[dict]:
    """Compare two runs' sample hashes on the runs Volume."""
    import modal

    from zephon_modal.verify.compare import compare_runs

    modal.Volume.from_name(runs_volume_name).reload()
    return compare_runs(f"{runs_mount}/{reference}", f"{runs_mount}/{candidate}")
