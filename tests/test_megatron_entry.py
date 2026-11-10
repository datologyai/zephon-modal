# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

import json
import pickle
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from zephon_modal.verify.hashes import StepHashRecorder
from zephon_modal.zephon import megatron_entry


def _batch(index):
    tokens = torch.arange(6, dtype=torch.int64).reshape(2, 3) + 10 * (index + 1)
    return {
        "tokens": tokens,
        "labels": tokens + 1,
        "loss_mask": torch.ones_like(tokens, dtype=torch.float32),
        "position_ids": torch.tensor([[0, 1, 2], [0, 1, 2]]),
    }


def _records(tmp_path, launch):
    return [
        json.loads(line)
        for path in sorted(tmp_path.glob(f"{launch}-*.jsonl"))
        for line in path.read_text().splitlines()
    ]


class CheckpointableLoader:
    # Stub only the framework loader boundary; hashing and serialization are real.
    is_save_state_rank_independent = True
    is_save_state_collective = True

    def __init__(self, batches):
        self.batches = batches
        self.cursor = 0
        self.saved_states = []
        self.restored_states = []

    def __iter__(self):
        return self

    def __next__(self):
        if self.cursor == len(self.batches):
            raise StopIteration
        batch = self.batches[self.cursor]
        self.cursor += 1
        return batch

    def save_state(self):
        state = {"zephon": pickle.dumps(self.cursor)}
        self.saved_states.append(state)
        return state

    def restore_state(self, state):
        self.restored_states.append(state)
        self.cursor = pickle.loads(state["zephon"])


@pytest.fixture
def hashing_iterator(tmp_path):
    def build(loader, *, launch="fresh", dp=1, rank=0, iteration=0):
        hashes = StepHashRecorder(
            str(tmp_path / f"{launch}-rank{rank}.jsonl"),
            dp_world_size=dp,
            dp_rank=rank,
            microbatches_per_step=4,
            consumed_microbatches=iteration * 4,
            schema="megatron-batch-v2",
        )
        return megatron_entry.SampleHashingIterator(loader, hashes)

    return build


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


@pytest.mark.parametrize(("initial_dp", "restored_dp"), [(1, 1), (2, 2), (2, 1), (1, 2)])
def test_checkpoint_round_trip_delegates_loader_state_and_continues_hash_steps(
    hashing_iterator, tmp_path, initial_dp, restored_dp
):
    batches = [_batch(index) for index in range(16)]
    base_loader = CheckpointableLoader(batches)
    loader = hashing_iterator(base_loader, dp=initial_dp)
    consumed_batches = 8 // initial_dp
    for index in range(consumed_batches):
        assert next(loader) is batches[index]
    assert [record["step"] for record in _records(tmp_path, "fresh")] == [1, 2]

    state = loader.save_state()
    assert len(base_loader.saved_states) == 1
    assert state is base_loader.saved_states[0]
    assert set(state) == {"zephon"}
    assert pickle.loads(state["zephon"]) == consumed_batches
    # Use Megatron's checkpoint envelope without adding wrapper state to it.
    checkpoint = tmp_path / "train_dataloader_dprank000.pt"
    torch.save({"dataloader_state_dict": state}, checkpoint)
    restored_state = torch.load(checkpoint, weights_only=True)["dataloader_state_dict"]

    restored_base = CheckpointableLoader(batches)
    # Megatron supplies the restored iteration; no hash counter is stored in the loader.
    restored = hashing_iterator(restored_base, launch="restored", dp=restored_dp, iteration=2)
    assert restored.is_save_state_rank_independent is True
    assert restored.is_save_state_collective is True
    restored.restore_state(restored_state)
    assert len(restored_base.restored_states) == 1
    assert restored_base.restored_states[0] is restored_state
    assert restored_base.cursor == consumed_batches
    assert _records(tmp_path, "restored") == []

    # The stub checks delegation and hash numbering, not Zephon's data resharding.
    for index in range(4 // restored_dp):
        assert _records(tmp_path, "restored") == []
        assert next(restored) is batches[consumed_batches + index]
    (record,) = _records(tmp_path, "restored")
    assert (record["step"], record["dp"]) == (3, restored_dp)
    assert record["schema"] == "megatron-batch-v2"
    assert len(record["row_hashes"]) == 2 * (4 // restored_dp)


@pytest.mark.parametrize(
    ("rank_independent", "collective"), [(True, True), (True, False), (False, True), (False, False)]
)
def test_wrapper_preserves_checkpoint_flags_on_nonzero_data_rank(hashing_iterator, rank_independent, collective):
    base_loader = CheckpointableLoader([])
    base_loader.is_save_state_rank_independent = rank_independent
    base_loader.is_save_state_collective = collective
    loader = hashing_iterator(base_loader, dp=2, rank=1)

    assert getattr(loader, "is_save_state_rank_independent", False) is rank_independent
    assert getattr(loader, "is_save_state_collective", False) is collective


def test_wrapper_does_not_add_checkpoint_capabilities_to_plain_iterators(hashing_iterator):
    loader = hashing_iterator(iter([]))

    assert not hasattr(loader, "save_state")
    assert not hasattr(loader, "restore_state")
    assert getattr(loader, "is_save_state_rank_independent", False) is False
    assert getattr(loader, "is_save_state_collective", False) is False
