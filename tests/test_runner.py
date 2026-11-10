import dataclasses
import json
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


def test_each_attempt_hashes_into_its_own_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "LOCAL_SAMPLE_HASH_DIR", str(tmp_path))
    spec = _spec(Placement("H100:8"))
    first = launch_context(dataclasses.replace(spec, attempt_tag=".a0001"), NodeTopology(1, 0, None))
    second = launch_context(dataclasses.replace(spec, attempt_tag=".a0002"), NodeTopology(1, 0, None))
    assert first.launch_id == spec.launch_id + ".a0001"
    assert first.sample_hash_dir != second.sample_hash_dir


def _synced_spec(tmp_path, monkeypatch) -> RunSpec:
    monkeypatch.setattr(runner, "LOCAL_SAMPLE_HASH_DIR", str(tmp_path / "local"))
    monkeypatch.setattr(runner, "_synced", {})
    spec = dataclasses.replace(_spec(Placement("H100:4")), runs_mount=str(tmp_path), attempt_tag=".a0001")
    os.makedirs(runner.local_hash_dir(spec))
    return spec


def test_sync_copies_complete_lines_only(tmp_path, monkeypatch):
    spec = _synced_spec(tmp_path, monkeypatch)
    name = hash_file_name(spec.launch_id, 4, 0)
    with open(f"{runner.local_hash_dir(spec)}/{name}", "w") as local_file:
        local_file.write('{"step": 1}\n{"step": 2')
    runner.sync_sample_hashes(spec)
    synced = tmp_path / "r1" / SAMPLE_HASH_SUBDIR
    assert os.listdir(synced) == [name]
    assert (synced / name).read_text() == '{"step": 1}\n'


def test_sync_skips_files_that_havent_changed(tmp_path, monkeypatch):
    spec = _synced_spec(tmp_path, monkeypatch)
    name = hash_file_name(spec.launch_id, 4, 0)
    local = f"{runner.local_hash_dir(spec)}/{name}"
    with open(local, "w") as local_file:
        local_file.write('{"step": 1}\n')
    runner.sync_sample_hashes(spec)
    synced = tmp_path / "r1" / SAMPLE_HASH_SUBDIR / name
    synced.write_text("sentinel")
    runner.sync_sample_hashes(spec)
    assert synced.read_text() == "sentinel"
    with open(local, "a") as local_file:
        local_file.write('{"step": 2}\n')
    runner.sync_sample_hashes(spec)
    assert synced.read_text() == '{"step": 1}\n{"step": 2}\n'


def test_manifest_allows_changes_that_dont_affect_the_data(tmp_path):
    spec = dataclasses.replace(_spec(Placement("H100:4")), runs_mount=str(tmp_path))
    runner.check_manifest(spec)
    runner.check_manifest(dataclasses.replace(spec, steps=500, adapter=Torchtitan(module="other", config="big")))
    with pytest.raises(ValueError, match="use a new run ID"):
        runner.check_manifest(dataclasses.replace(spec, seed=7))
    with pytest.raises(ValueError, match="use a new run ID"):
        runner.check_manifest(dataclasses.replace(spec, data=Zephon(canonical_replicas=32)))


def test_launch_attempts_are_node_zero_records_oldest_first(tmp_path):
    launch_dir = tmp_path / "r1" / "launches" / "L1"
    launch_dir.mkdir(parents=True)
    records = {
        "ta-2.json": {"task_id": "ta-2", "node_rank": 0, "started": "2026-11-09T01:00:00+00:00"},
        "ta-1.json": {"task_id": "ta-1", "node_rank": 0, "started": "2026-11-09T00:00:00+00:00", "ended": "x"},
        "ta-9.json": {"task_id": "ta-9", "node_rank": 1, "started": "2026-11-09T01:00:00+00:00"},
        "launch.json": {"launch_id": "L1"},
    }
    for name, record in records.items():
        (launch_dir / name).write_text(json.dumps(record))
    run_dir = str(tmp_path / "r1")
    assert [r["task_id"] for r in runner.launch_attempts(run_dir, "L1")] == ["ta-1", "ta-2"]
    assert len(runner.launch_records(run_dir, "L1")) == 3
    assert len(runner.all_records(run_dir)) == 3


def test_heartbeat_survives_a_failed_write(tmp_path, monkeypatch, capsys):
    import threading

    spec = dataclasses.replace(_spec(Placement("H100:4")), runs_mount=str(tmp_path))
    ctx = launch_context(spec, NodeTopology(1, 0, None))
    monkeypatch.setattr(runner, "SYNC_SECONDS", 0.01)
    beats = []

    def flaky_record(spec, ctx, record):
        beats.append(record)
        if len(beats) == 1:
            raise OSError("Volume hiccup")

    monkeypatch.setattr(runner, "record_attempt", flaky_record)
    done = threading.Event()
    thread = threading.Thread(target=runner._heartbeat, args=(spec, ctx, {"outcome": "running"}, done))
    thread.start()
    while len(beats) < 3:
        time.sleep(0.01)
    done.set()
    thread.join(timeout=2)
    assert "heartbeat failed" in capsys.readouterr().out
    assert all("heartbeat" in beat for beat in beats)


def test_volume_history_skips_a_record_caught_mid_commit(tmp_path):
    launch_dir = tmp_path / "r1" / "launches" / "L1"
    launch_dir.mkdir(parents=True)
    (launch_dir / "ta-1.json").write_text(json.dumps({"task_id": "ta-1", "node_rank": 0, "started": "x"}))
    (launch_dir / "ta-2.json").write_text("")
    assert len(runner.launch_records(str(tmp_path / "r1"), "L1")) == 1
    assert len(runner.all_records(str(tmp_path / "r1"))) == 1


class _Store(dict):
    def put(self, key, value, skip_if_exists=False):
        self[key] = value


def test_records_are_published_to_the_registry_and_the_final_one_only_on_request(tmp_path, monkeypatch):
    from zephon_modal.registry import Registry

    registry = Registry(_Store())
    monkeypatch.setattr(runner, "_registry", lambda spec: registry)
    monkeypatch.setenv("MODAL_TASK_ID", "ta-1")
    spec = dataclasses.replace(_spec(Placement("H100:4")), runs_mount=str(tmp_path), registry_name="job-runs")
    ctx = launch_context(spec, NodeTopology(1, 0, None))
    runner.record_attempt(spec, ctx, {"started": "2026-11-10T00:00:00+00:00", "outcome": "running"})
    runner.record_attempt(spec, ctx, {"started": "2026-11-10T00:00:00+00:00", "ended": "t"}, publish=False)
    (record,) = registry.launch_records("r1", spec.launch_id)
    assert record["outcome"] == "running" and "ended" not in record
    # The Volume's history copy already has the final record.
    assert "ended" in runner.launch_records(str(tmp_path / "r1"), spec.launch_id)[0]
    runner.publish_attempt(
        spec, {"task_id": "ta-1", "node_rank": 0, "started": "2026-11-10T00:00:00+00:00", "ended": "t"}
    )
    assert "ended" in registry.launch_records("r1", spec.launch_id)[0]


def test_publishing_retries_then_raises(monkeypatch):
    calls = []

    class Flaky:
        def record(self, *args):
            calls.append(args)
            raise ConnectionError("Dict unavailable")

    monkeypatch.setattr(runner, "_registry", lambda spec: Flaky())
    monkeypatch.setattr(runner.time, "sleep", lambda seconds: None)
    with pytest.raises(ConnectionError):
        runner.publish_attempt(_spec(Placement("H100:4")), {"started": "x"}, attempts=3)
    assert len(calls) == 3
