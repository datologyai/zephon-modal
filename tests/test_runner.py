import dataclasses
import os
import subprocess
import sys
import time

import pytest

from zephon_modal import runner
from zephon_modal.adapters import Megatron, Torchtitan
from zephon_modal.context import LaunchContext
from zephon_modal.data import DataContract
from zephon_modal.placement import Placement
from zephon_modal.runner import NodeTopology, RunSpec, launch_context, stop_trainer, torchrun_command
from zephon_modal.verify.hashes import SAMPLE_HASH_SUBDIR, hash_file_name
from zephon_modal.zephon import Zephon, settings


def _spec(placement: Placement, adapter=None, data: DataContract | None = None) -> RunSpec:
    return RunSpec(
        run_id="r1",
        steps=20,
        seed=42,
        launch_id="20261102T000000",
        placement=placement,
        adapter=adapter or Torchtitan(module="recipes", config="smoke"),
        data=Zephon(canonical_replicas=16) if data is None else data,
        runs_mount="/runs",
        runs_volume_name="runs",
    )


def test_single_node_context_carries_zephon_settings():
    ctx = launch_context(_spec(Placement("H100:4")), NodeTopology(1, 0, None))
    assert ctx.world_size == 4
    assert not ctx.keep_all_checkpoints
    zephon = settings(ctx)
    assert (zephon.canonical_replicas, zephon.aggregate_store) == (16, "torch://")


def test_multi_node_context_uses_tcpstore():
    ctx = launch_context(_spec(Placement("H100:8", nodes=2)), NodeTopology(2, 1, "10.0.0.1"))
    assert (ctx.world_size, ctx.node_rank) == (16, 1)
    assert settings(ctx).aggregate_store == "torch://"
    assert ctx.keep_all_checkpoints


def test_generic_contract_adds_no_settings():
    ctx = launch_context(_spec(Placement("H100:4"), data=DataContract()), NodeTopology(1, 0, None))
    assert ctx.data == {}
    with pytest.raises(RuntimeError, match="isn't zephon_modal.zephon.Zephon"):
        settings(ctx)


def test_context_round_trips_through_env():
    ctx = launch_context(_spec(Placement("H100:8")), NodeTopology(1, 0, None))
    assert LaunchContext.from_env(ctx.to_env()) == ctx


def test_torchtitan_command():
    spec = _spec(Placement("H100:8", nodes=2))
    topology = NodeTopology(2, 0, "10.0.0.1")
    command = torchrun_command(spec.adapter, launch_context(spec, topology), topology)
    assert command[:6] == [
        "torchrun",
        "--nproc_per_node=8",
        "--nnodes=2",
        "--node-rank=0",
        "--master-addr=10.0.0.1",
        "--master-port=29500",
    ]
    assert command[-5:] == ["-m", "torchtitan.train", "--module=recipes", "--config=smoke", "--output-dir=/runs/r1"]


def megatron_args(ctx: LaunchContext) -> list[str]:
    return [f"--train-iters={ctx.steps}", f"--zephon-aggregate-store={settings(ctx).aggregate_store}"]


def test_megatron_runs_the_script_directly_without_an_entry():
    spec = _spec(Placement("H100:4"), Megatron(args=f"{__name__}:megatron_args"))
    topology = NodeTopology(1, 0, None)
    command = torchrun_command(spec.adapter, launch_context(spec, topology), topology)
    assert command[-3:] == ["pretrain_gpt.py", "--train-iters=20", "--zephon-aggregate-store=torch://"]


def test_megatron_command_resolves_args_in_container():
    adapter = Megatron(
        args=f"{__name__}:megatron_args", script="pretrain_gpt_zephon.py", entry="zephon_modal.zephon.megatron_entry"
    )
    spec = _spec(Placement("H100:4"), adapter)
    topology = NodeTopology(1, 0, None)
    command = torchrun_command(spec.adapter, launch_context(spec, topology), topology)
    assert "--local-ranks-filter=0,3" in command
    assert command[-5:] == [
        "-m",
        "zephon_modal.zephon.megatron_entry",
        "pretrain_gpt_zephon.py",
        "--train-iters=20",
        "--zephon-aggregate-store=torch://",
    ]


def test_megatron_checkpoint_on_shutdown_adds_exit_signal_handler():
    adapter = Megatron(args=f"{__name__}:megatron_args", checkpoint_on_shutdown=True)
    ctx = launch_context(_spec(Placement("H100:8"), adapter=adapter), NodeTopology(1, 0, None))
    assert adapter.command(ctx)[-1] == "--exit-signal-handler"


def _start(code: str) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", code], start_new_session=True)


def test_stop_trainer_lets_a_trainer_exit():
    trainer = _start(
        "import signal, sys, time\n"
        "signal.signal(signal.SIGTERM, lambda *_: sys.exit(3))\n"
        "print('ready', flush=True)\n"
        "time.sleep(60)\n"
    )
    time.sleep(1)
    assert stop_trainer(trainer, Torchtitan(module="m", config="c", shutdown_timeout=10)) == 3


def test_stop_trainer_kills_a_trainer_that_outlasts_the_timeout():
    trainer = _start("import signal, time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\ntime.sleep(60)\n")
    time.sleep(1)
    started = time.monotonic()
    assert stop_trainer(trainer, Torchtitan(module="m", config="c", shutdown_timeout=0.5)) is None
    assert time.monotonic() - started < 5


def test_restart_restores_this_nodes_hash_files(tmp_path, monkeypatch):
    local = tmp_path / "local"
    monkeypatch.setattr(runner, "LOCAL_SAMPLE_HASH_DIR", str(local))
    spec = dataclasses.replace(_spec(Placement("H100:8", nodes=2)), runs_mount=str(tmp_path))
    run_hashes = tmp_path / "r1" / SAMPLE_HASH_SUBDIR
    run_hashes.mkdir(parents=True)
    for rank in (7, 8, 15):
        (run_hashes / hash_file_name(spec.launch_id, 16, rank)).write_text('{"step": 1}\n{"step": 2')
    (run_hashes / hash_file_name("20261101T000000", 16, 8)).write_text('{"step": 1}\n')

    ctx = launch_context(spec, NodeTopology(2, 1, "10.0.0.1"))
    runner._restore_sample_hashes(spec, ctx)

    # Node 1 holds global ranks 8-15, and only this launch's files; the half-written line is dropped.
    assert sorted(os.listdir(local)) == [hash_file_name(spec.launch_id, 16, r) for r in (8, 15)]
    assert (local / hash_file_name(spec.launch_id, 16, 8)).read_text() == '{"step": 1}\n'


def test_sync_copies_complete_lines_only(tmp_path, monkeypatch):
    local = tmp_path / "local"
    local.mkdir()
    monkeypatch.setattr(runner, "LOCAL_SAMPLE_HASH_DIR", str(local))
    spec = dataclasses.replace(_spec(Placement("H100:4")), runs_mount=str(tmp_path))
    name = hash_file_name(spec.launch_id, 4, 0)
    (local / name).write_text('{"step": 1}\n{"step": 2')
    runner.sync_sample_hashes(spec)
    synced = tmp_path / "r1" / SAMPLE_HASH_SUBDIR
    assert os.listdir(synced) == [name]
    assert (synced / name).read_text() == '{"step": 1}\n'
