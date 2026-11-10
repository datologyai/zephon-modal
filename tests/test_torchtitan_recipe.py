# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from zephon_modal.context import LaunchContext
from zephon_modal.zephon import Zephon


@dataclass(kw_only=True, slots=True)
class CurrentTrainingConfig:
    num_tokens_per_microbatch_per_dp_rank: int
    num_tokens_per_train_step: int
    steps: int


@dataclass(kw_only=True, slots=True)
class LegacyTrainingConfig(CurrentTrainingConfig):
    max_context_length: int = 2048


@pytest.mark.parametrize(
    ("training_config", "default_attention"),
    [(LegacyTrainingConfig, "flex"), (CurrentTrainingConfig, "varlen")],
    ids=["pre-sync", "upstream-sync"],
)
def test_example_preserves_training_and_data_settings(monkeypatch, tmp_path, training_config, default_attention):
    # Strict dataclasses model the removed constructor argument and prevent adding
    # it back as a dynamic attribute on the current API. Frameworks stay out of the client env.
    configurable = SimpleNamespace(Config=SimpleNamespace)

    def build_model_config(flavor, *, seq_len, parallelism, attn_backend=default_attention):
        assert flavor == "0.6B"
        return SimpleNamespace(max_context_length=seq_len, attn_backend=attn_backend)

    modules = {
        "storage": {"FINEWEB_EDU_DIR": "/data/fineweb"},
        "torchtitan.components.checkpointer": {"CheckpointManager": configurable},
        "torchtitan.components.loss": {"ChunkedLossWrapper": configurable, "CrossEntropyLoss": configurable},
        "torchtitan.components.optim": dict.fromkeys(
            ("AdamW", "LRSchedulersContainer", "Optim", "OptimizersContainer"), configurable
        ),
        "torchtitan.config": {"TrainingConfig": training_config},
        "torchtitan.config.parallelism": {"ParallelismConfig": SimpleNamespace},
        "torchtitan.models.common.config_utils": {"decoder_vocab_size": lambda model: 151936},
        "torchtitan.models.qwen3": {"build_model_config": build_model_config},
        "torchtitan.trainer": {"Trainer": configurable},
        "torchtitan_recipes.zephon": {"ZephonSource": SimpleNamespace},
        "zephon_modal.zephon.torchtitan": {"SampleHashingZephonDataLoader": configurable},
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)

    ctx = LaunchContext(
        run_id="recipe-test",
        run_dir=str(tmp_path),
        steps=37,
        seed=123,
        launch_id="launch-test",
        placement="h100x4",
        nnodes=1,
        node_rank=0,
        nproc_per_node=4,
        sample_hash_dir=str(tmp_path / "hashes"),
        data=Zephon(canonical_replicas=16).context(),
    )
    for key, value in ctx.to_env().items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("HF_ASSETS_DIR", "/assets/hf")
    path = Path(__file__).resolve().parents[1] / "examples/torchtitan/recipe.py"
    spec = importlib.util.spec_from_file_location("torchtitan_example_recipe", path)
    recipe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recipe)

    config = recipe.qwen3_0p6b_zephon()

    assert config.model.max_context_length == 4096
    assert config.model.attn_backend == "flex"
    assert config.training.num_tokens_per_microbatch_per_dp_rank == 2 * 4096
    assert config.training.num_tokens_per_train_step == 128 * 4096
    assert config.training.steps == 37
    if training_config is LegacyTrainingConfig:
        assert config.training.max_context_length == config.model.max_context_length
    else:
        assert not hasattr(config.training, "max_context_length")
    assert getattr(config.dataloader, "max_num_documents", None) is None
    assert config.dataloader.seed == 123
    assert config.dataloader.canonical_replicas == 16
    assert config.dataloader.microbatches_per_step == 64
    assert config.dataloader.aggregate_store == "torch://"
    assert config.checkpointer.keep_latest_k == 0
    assert config.checkpointer.last_save_model_only is False
