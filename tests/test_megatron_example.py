# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import toml

from zephon_modal.context import LaunchContext
from zephon_modal.zephon import Zephon

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/megatron"


@pytest.fixture
def example(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(EXAMPLE))
    monkeypatch.setenv("ZEPHON_MODAL_MEGATRON_PREFIX", "test-megatron")
    monkeypatch.setenv("ZEPHON_MODAL_MEGATRON_DATA_VOLUME", "test-megatron-data")
    monkeypatch.setenv("ZEPHON_MODAL_MEGATRON_RUNS_VOLUME", "test-megatron-runs")
    monkeypatch.delenv("ZEPHON_MODAL_MEGATRON_SAMPLE_HASHES", raising=False)
    monkeypatch.delenv("ZEPHON_MODAL_MEGATRON_ALERT_SECRET", raising=False)
    monkeypatch.setenv("HF_ASSETS_DIR", "/assets/hf")

    def load(name):
        spec = importlib.util.spec_from_file_location(name, EXAMPLE / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    load("storage")
    recipe = load("recipe")
    monkeypatch.setattr(recipe, "DATA_RECIPE_PATH", str(tmp_path / "data.toml"))
    return load, recipe


def _context(tmp_path, lanes):
    return LaunchContext(
        run_id="example-test",
        run_dir=str(tmp_path / "run"),
        steps=37,
        seed=123,
        launch_id="first",
        placement="test",
        nnodes=1,
        node_rank=0,
        nproc_per_node=1,
        sample_hash_dir=str(tmp_path / "hashes"),
        data=Zephon(canonical_replicas=lanes).context(),
    )


@pytest.mark.parametrize(("smoke", "lanes", "seq_len", "global_batch"), [(False, 16, 4096, 128), (True, 1, 128, 2)])
def test_recipe_keeps_data_and_checkpoint_settings_across_resume(
    example, tmp_path, smoke, lanes, seq_len, global_batch
):
    _, recipe = example
    ctx = _context(tmp_path, lanes)
    build = recipe.smoke_args if smoke else recipe.qwen3_0p6b_zephon_args
    args = build(ctx)
    flags = dict(arg.split("=", 1) for arg in args if "=" in arg)
    data = toml.loads(Path(recipe.DATA_RECIPE_PATH).read_text())
    assert flags["--seq-length"] == str(seq_len)
    assert flags["--global-batch-size"] == str(global_batch)
    assert global_batch // int(flags["--micro-batch-size"]) % lanes == 0
    assert flags["--zephon-canonical-replicas"] == str(lanes)
    assert flags["--zephon-aggregate-store"] == "torch://"
    assert flags["--zephon-run-id"] == ctx.run_id
    assert flags["--seed"] == str(ctx.seed)
    assert flags["--save"] == flags["--load"] == f"{ctx.run_dir}/checkpoint"
    assert flags["--dataloader-save"] == f"{ctx.run_dir}/dataloader"
    assert flags["--ckpt-format"] == "torch_dist"
    assert not {"--no-save-optim", "--no-load-optim", "--no-load-rng"}.intersection(args)
    assert data["seed"] == ctx.seed
    assert data["tokenize_special_tokens"] == "eos"
    assert data["token_estimation"] is False
    assert data["sources"] == [
        {"name": "text", "path": recipe.SMOKE_DATA_DIR if smoke else recipe.FINEWEB_EDU_DIR, "fmt": "parquet"}
    ]
    if smoke:
        assert data["chunk_size"] <= 64
        assert data["runner"] == "inline"
        assert data["mtp_mode"] is False

    resumed = build(replace(ctx, steps=77, launch_id="restored"))
    assert [arg for arg in args if not arg.startswith("--train-iters=")] == [
        arg for arg in resumed if not arg.startswith("--train-iters=")
    ]
    assert "--train-iters=77" in resumed
    assert toml.loads(Path(recipe.DATA_RECIPE_PATH).read_text()) == data


@pytest.mark.parametrize("hashing", [False, True], ids=["default-no-hashes", "opt-in-hashes"])
def test_jobs_load_and_launch_without_local_framework_checkouts(example, monkeypatch, tmp_path, hashing):
    load, _ = example
    if hashing:
        monkeypatch.setenv("ZEPHON_MODAL_MEGATRON_SAMPLE_HASHES", "1")
    load("image")
    module = load("job")
    assert module.job.name == "test-megatron"
    assert module.smoke.name == "test-megatron-smoke"
    assert [p.name for p in module.job.placements] == ["2xh100x8", "h100x8-or-h100x4"]
    assert [p.name for p in module.smoke.placements] == ["h100x1"]
    for job, lanes in ((module.job, 16), (module.smoke, 1)):
        assert job.data.canonical_replicas == lanes
        assert job.runs_volume.name == "test-megatron-runs"
        assert job.volumes["/data"].name == "test-megatron-data"
        command = job.adapter.command(_context(tmp_path, lanes))
        if hashing:
            assert command[:3] == ["-m", "zephon_modal.zephon.megatron_entry", "pretrain_gpt_zephon.py"]
        else:
            assert job.adapter.entry is None
            assert command[0] == "pretrain_gpt_zephon.py"
        assert "--train-iters=37" in command
