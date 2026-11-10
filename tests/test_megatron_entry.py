# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from zephon_modal.zephon import megatron_entry


@pytest.mark.parametrize(
    ("has_state_api", "state_iteration", "legacy_iteration", "expected_iteration"),
    [
        pytest.param(True, 0, None, 0, id="modern-fresh"),
        pytest.param(True, 7, None, 7, id="modern-resume"),
        pytest.param(True, 0, 99, 0, id="prefer-modern-zero"),
        pytest.param(True, 7, 99, 7, id="prefer-modern-resume"),
        pytest.param(False, None, 7, 7, id="legacy-only"),
        pytest.param(True, None, 7, 7, id="state-not-initialized"),
    ],
)
def test_hashing_uses_latest_available_iteration(
    monkeypatch, tmp_path, has_state_api, state_iteration, legacy_iteration, expected_iteration
):
    batch = {
        "tokens": torch.tensor([[11, 12]]),
        "labels": torch.tensor([[12, 13]]),
        "loss_mask": torch.tensor([[1.0, 1.0]]),
        "position_ids": torch.tensor([[0, 1]]),
    }

    def provider(_samples, _vp_stage):
        return iter([batch]), None, None

    provider.is_distributed = True
    modules = {
        name: ModuleType(name)
        for name in (
            "megatron",
            "megatron.core",
            "megatron.core.num_microbatches_calculator",
            "megatron.training",
            "megatron.training.datasets",
        )
    }
    modules["megatron.core"].mpu = SimpleNamespace(
        get_data_parallel_world_size=lambda: 2, get_data_parallel_rank=lambda: 0
    )
    modules["megatron.core.num_microbatches_calculator"].get_num_microbatches = lambda: 1
    training = modules["megatron.training"]
    modules["megatron"].training = training
    if has_state_api:
        state = SimpleNamespace(iteration=state_iteration) if state_iteration is not None else None
        training.get_train_state = lambda: state
    if legacy_iteration is not None:
        training.get_args = lambda: SimpleNamespace(iteration=legacy_iteration)
    dataloader = SimpleNamespace(zephon_train_valid_test_datasets_provider=provider)
    modules["megatron.training.datasets"].zephon_dataloader = dataloader
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(megatron_entry.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(
        megatron_entry,
        "context",
        lambda: SimpleNamespace(
            sample_hash_dir=str(tmp_path),
            launch_id="resume",
            data={"canonical_replicas": 2, "aggregate_store": "torch://"},
        ),
    )

    megatron_entry._install_sample_hashing()
    loader, valid, test = dataloader.zephon_train_valid_test_datasets_provider([], None)
    assert next(loader) is batch
    assert valid is test is None
    (hash_file,) = tmp_path.glob("*.jsonl")
    record = json.loads(hash_file.read_text())
    assert record["step"] == expected_iteration + 1
    assert record["schema"] == "megatron-batch-v2"
    assert len(record["row_hashes"]) == 1
