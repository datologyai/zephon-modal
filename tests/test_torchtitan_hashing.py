# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import pickle
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import ModuleType

import pytest
import torch
import torch.distributed.checkpoint as dcp

from zephon_modal.context import LaunchContext
from zephon_modal.zephon import Zephon


@dataclass(kw_only=True)
class Microbatch:
    input: torch.Tensor
    labels: torch.Tensor
    positions: torch.Tensor
    padding_mask: torch.Tensor
    loss_token_counts: torch.Tensor
    routing_token_counts: torch.Tensor
    model_kwargs: dict = field(default_factory=dict)


def _batch(index=0):
    tokens = torch.arange(6, dtype=torch.int64).reshape(1, 6) + 10 * (index + 1)
    return Microbatch(
        input=tokens,
        labels=tokens + 1,
        positions=torch.tensor([[0, 1, 0, 0, 1, 2]]),
        padding_mask=torch.tensor([[False, False, True, False, False, False]]),
        loss_token_counts=torch.tensor([5]),
        routing_token_counts=torch.tensor([5]),
    )


def _records(tmp_path, launch):
    return [
        json.loads(line)
        for path in sorted((tmp_path / "hashes").glob(f"{launch}-*.jsonl"))
        for line in path.read_text().splitlines()
    ]


@pytest.fixture
def hashing_loader(monkeypatch, tmp_path):
    # Stub only the framework boundary. The actual wrapper, tensors, hash
    # recorder and DCP serialization run unchanged in the CPU-only client env.
    class ZephonDataLoader:
        @dataclass(kw_only=True, slots=True)
        class Config:
            pass

        def __init__(self, config, *, batches, **kwargs):
            self._batches = batches
            self._cursor = 0

        def __iter__(self):
            while self._cursor < len(self._batches):
                batch = self._batches[self._cursor]
                self._cursor += 1
                yield batch

        def state_dict(self):
            return {"base_loader": pickle.dumps(self._cursor)}

        def load_state_dict(self, state):
            self._cursor = pickle.loads(state["base_loader"])

    for name, attributes in {
        "torchtitan.components.data.types": {"TokenizedTrainingMicrobatch": Microbatch},
        "torchtitan_recipes.zephon": {"ZephonDataLoader": ZephonDataLoader},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)

    path = Path(__file__).resolve().parents[1] / "zephon_modal/zephon/torchtitan.py"
    spec = importlib.util.spec_from_file_location("torchtitan_hashing_under_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.dist, "is_initialized", lambda: False)

    def build(
        batches,
        *,
        launch="fresh",
        dp=1,
        rank=0,
        global_rank=None,
        microbatches_per_step=4,
        canonical_replicas=2,
    ):
        ctx = LaunchContext(
            run_id="hashing-test",
            run_dir=str(tmp_path),
            steps=10,
            seed=42,
            launch_id=launch,
            placement="cpu",
            nnodes=1,
            node_rank=0,
            nproc_per_node=max(dp, (global_rank or 0) + 1),
            sample_hash_dir=str(tmp_path / "hashes"),
            data=Zephon(canonical_replicas=canonical_replicas).context(),
        )
        for key, value in ctx.to_env().items():
            monkeypatch.setenv(key, value)
        monkeypatch.setattr(module.dist, "is_initialized", lambda: global_rank is not None)
        monkeypatch.setattr(module.dist, "get_rank", lambda: global_rank)
        return module.SampleHashingZephonDataLoader(
            module.SampleHashingZephonDataLoader.Config(microbatches_per_step=microbatches_per_step),
            dp_world_size=dp,
            dp_rank=rank,
            max_context_length=3,
            batches=batches,
        )

    return build


@pytest.mark.parametrize(("dp", "rank", "global_rank"), [(1, 0, None), (2, 1, 5)])
def test_fresh_hashes_start_at_step_one_after_a_whole_global_step(hashing_loader, tmp_path, dp, rank, global_rank):
    batches = [_batch(i) for i in range(4 // dp)]
    loader = hashing_loader(batches, dp=dp, rank=rank, global_rank=global_rank)
    iterator = iter(loader)
    for batch in batches[:-1]:
        assert next(iterator) is batch
        assert _records(tmp_path, "fresh") == []
    assert next(iterator) is batches[-1]

    (record,) = _records(tmp_path, "fresh")
    assert (record["step"], record["dp"], record["dp_rank"]) == (1, dp, rank)
    assert record["schema"] == "torchtitan-batch-v2"
    assert len(record["row_hashes"]) == 2 * len(batches)
    assert pickle.loads(loader.state_dict()["zephon_modal_consumed_microbatches"]) == 4
    (path,) = (tmp_path / "hashes").glob("*.jsonl")
    assert path.name == f"fresh-dp{dp}-rank{global_rank or 0:03d}-s00000.jsonl"


@pytest.mark.parametrize(("initial_dp", "restored_dp"), [(1, 1), (2, 2), (2, 1), (1, 2)])
def test_dcp_restores_base_loader_and_global_hash_progress(hashing_loader, tmp_path, initial_dp, restored_dp):
    batches = [_batch(i) for i in range(16)]
    loader = hashing_loader(batches, dp=initial_dp)
    iterator = iter(loader)
    consumed_batches = 8 // initial_dp
    for _ in range(consumed_batches):
        next(iterator)
    assert [record["step"] for record in _records(tmp_path, "fresh")] == [1, 2]
    state = loader.state_dict()
    assert set(state) == {"base_loader", "zephon_modal_consumed_microbatches"}
    assert pickle.loads(state["base_loader"]) == consumed_batches
    assert pickle.loads(state["zephon_modal_consumed_microbatches"]) == 8

    checkpoint = tmp_path / "checkpoint"
    dcp.save({"dataloader": loader}, checkpoint_id=str(checkpoint), no_dist=True)
    restored = hashing_loader(batches, launch="restored", dp=restored_dp)
    dcp.load({"dataloader": restored}, checkpoint_id=str(checkpoint), no_dist=True)
    assert restored.state_dict() == state

    iterator = iter(restored)
    # The base stub's cursor checks delegation, not Zephon's DP resharding.
    for index in range(4 // restored_dp):
        assert _records(tmp_path, "restored") == []
        assert next(iterator) is batches[consumed_batches + index]
    (record,) = _records(tmp_path, "restored")
    assert record["step"] == 3
    assert record["dp"] == restored_dp
    assert len(record["row_hashes"]) == 2 * (4 // restored_dp)
    assert pickle.loads(restored.state_dict()["zephon_modal_consumed_microbatches"]) == 12


@pytest.mark.parametrize("name", ["input", "labels", "positions", "padding_mask"])
def test_hashes_cover_each_field_and_each_sequence(hashing_loader, tmp_path, name):
    original = _batch()
    changed_field = getattr(original, name).clone()
    changed_field[0, 0] = not changed_field[0, 0] if name == "padding_mask" else changed_field[0, 0] + 1
    changed = replace(original, **{name: changed_field})
    loader = hashing_loader([original, changed], microbatches_per_step=1, canonical_replicas=1)
    yielded = list(loader)
    assert yielded[0] is original
    assert yielded[1] is changed

    first, second = _records(tmp_path, "fresh")
    assert [first["step"], second["step"]] == [1, 2]
    assert len(first["row_hashes"]) == len(second["row_hashes"]) == 2
    assert first["row_hashes"][0] != second["row_hashes"][0]
    assert first["row_hashes"][1] == second["row_hashes"][1]


def test_rejects_model_inputs_that_would_not_be_hashed(hashing_loader, tmp_path):
    loader = hashing_loader([replace(_batch(), model_kwargs={"extra": torch.ones(1)})])
    with pytest.raises(ValueError, match="extra model_kwargs"):
        next(iter(loader))
    assert _records(tmp_path, "fresh") == []


def test_rejects_checkpoint_progress_in_the_middle_of_a_step(hashing_loader):
    loader = hashing_loader([_batch()])
    state = loader.state_dict() | {"zephon_modal_consumed_microbatches": pickle.dumps(3)}
    with pytest.raises(ValueError, match="Checkpoint was taken mid-step"):
        loader.load_state_dict(state)
