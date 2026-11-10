# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from zephon_modal.zephon import megatron_entry


@pytest.mark.parametrize("iteration", [0, 7])
def test_hashing_resumes_from_train_state_without_cli_iteration(monkeypatch, tmp_path, iteration):
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
    modules["megatron.training"].get_train_state = lambda: SimpleNamespace(iteration=iteration)
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
    assert record["step"] == iteration + 1
    assert record["schema"] == "megatron-batch-v2"
    assert len(record["row_hashes"]) == 1
