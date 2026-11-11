import os
import shutil

import pytest
import torch
from torch.distributed.checkpoint.state_dict_saver import save as dcp_save

from zephon_modal import checkpoints
from zephon_modal.checkpoints import DISCARDED_SUBDIR, CheckpointLayout, dcp_complete, discard_incomplete, prune

MEGATRON = CheckpointLayout(step_pattern=r"iter_(\d+)", tracker="latest_checkpointed_iteration.txt")


def _save(path) -> None:
    dcp_save({"weight": torch.arange(4096.0)}, checkpoint_id=str(path), no_dist=True)


@pytest.fixture
def run_dir(tmp_path):
    for step in (10, 20):
        _save(tmp_path / "checkpoint" / f"step-{step}")
    return tmp_path


def test_finished_checkpoint_is_complete(run_dir):
    assert dcp_complete(str(run_dir / "checkpoint" / "step-20"))
    assert CheckpointLayout().latest_complete(str(run_dir)) == 20


@pytest.mark.parametrize(
    "damage",
    [
        pytest.param(lambda path: os.remove(path / ".metadata"), id="no metadata"),
        pytest.param(lambda path: os.remove(path / "__0_0.distcp"), id="missing shard"),
        pytest.param(lambda path: os.truncate(path / "__0_0.distcp", 100), id="truncated shard"),
        pytest.param(lambda path: os.truncate(path / ".metadata", 10), id="truncated metadata"),
    ],
)
def test_damaged_checkpoint_is_incomplete(run_dir, damage):
    damage(run_dir / "checkpoint" / "step-20")
    assert not dcp_complete(str(run_dir / "checkpoint" / "step-20"))
    assert CheckpointLayout().latest_complete(str(run_dir)) == 10


def test_discard_moves_incomplete_checkpoints_aside(run_dir):
    os.remove(run_dir / "checkpoint" / "step-20" / "__0_0.distcp")
    cleanup = discard_incomplete(CheckpointLayout(), str(run_dir), tag="ta-1")
    assert (cleanup.latest, cleanup.discarded) == (10, [20])
    assert sorted(os.listdir(run_dir / "checkpoint")) == ["step-10"]
    assert os.listdir(run_dir / DISCARDED_SUBDIR) == ["step-20-ta-1"]


def test_discard_points_megatron_tracker_at_latest_complete(tmp_path):
    for step in (10, 20):
        _save(tmp_path / "checkpoint" / f"iter_{step:07d}")
    tracker = tmp_path / "checkpoint" / "latest_checkpointed_iteration.txt"
    tracker.write_text("20")
    os.truncate(tmp_path / "checkpoint" / "iter_0000020" / "__0_0.distcp", 0)
    assert discard_incomplete(MEGATRON, str(tmp_path), tag="ta-1").latest == 10
    assert tracker.read_text() == "10"

    os.remove(tmp_path / "checkpoint" / "iter_0000010" / ".metadata")
    assert discard_incomplete(MEGATRON, str(tmp_path), tag="ta-2").latest is None
    assert not tracker.exists()


def test_nothing_to_discard_leaves_tracker_alone(tmp_path):
    _save(tmp_path / "checkpoint" / "iter_0000010")
    tracker = tmp_path / "checkpoint" / "latest_checkpointed_iteration.txt"
    tracker.write_text("10")
    assert discard_incomplete(MEGATRON, str(tmp_path), tag="ta-1").discarded == []
    assert tracker.read_text() == "10"


@pytest.mark.parametrize("state_write", ["missing", "empty", "truncated"])
def test_megatron_partial_data_save_cannot_replace_the_last_resumable_checkpoint(tmp_path, state_write):
    layout = CheckpointLayout(
        step_pattern=r"iter_(\d+)",
        tracker="latest_checkpointed_iteration.txt",
        required_torch_files=("mp_rank_00/train_dataloader_dprank000.pt",),
    )
    paths = {step: tmp_path / "checkpoint" / f"iter_{step:07d}" for step in (10, 20)}
    state_files = {}
    for step, path in paths.items():
        _save(path)
        state_files[step] = path / layout.required_torch_files[0]
        state_files[step].parent.mkdir()
        torch.save({"dataloader_state_dict": {"zephon": b"checkpoint"}}, state_files[step])
    tracker = tmp_path / "checkpoint" / layout.tracker
    tracker.write_text("20")
    if state_write == "missing":
        state_files[20].unlink()
    else:
        os.truncate(state_files[20], 0 if state_write == "empty" else 100)

    # Model DCP is complete, but its Zephon state has not safely landed yet.
    assert dcp_complete(str(paths[20]))
    assert layout.latest_complete(str(tmp_path)) == 10
    assert prune(layout, str(tmp_path), keep=1) == []
    assert state_files[10].is_file()
    cleanup = discard_incomplete(layout, str(tmp_path), tag="preempted")
    assert (cleanup.latest, cleanup.discarded) == (10, [20])
    assert tracker.read_text() == "10"

    # A subsequent complete save can replace the previous model and data state
    # together; no separate dataloader directory accumulates indefinitely.
    _save(paths[20])
    state_files[20].parent.mkdir()
    torch.save({"dataloader_state_dict": {"zephon": b"new checkpoint"}}, state_files[20])
    assert layout.latest_complete(str(tmp_path)) == 20
    assert prune(layout, str(tmp_path), keep=1) == [10]
    assert not paths[10].exists()
    assert state_files[20].is_file()
    # Retain the failed save at the current step for diagnosis; it becomes
    # eligible for cleanup once a newer complete checkpoint is kept.
    assert (tmp_path / DISCARDED_SUBDIR / "iter_0000020-preempted").exists()


def test_prune_keeps_the_latest_complete_checkpoints_and_anything_newer(run_dir):
    for step in (30, 40):
        _save(run_dir / "checkpoint" / f"step-{step}")
    # A save still in progress, newer than every complete checkpoint.
    (run_dir / "checkpoint" / "step-50").mkdir()
    assert prune(CheckpointLayout(), str(run_dir), keep=2) == [10, 20]
    assert sorted(os.listdir(run_dir / "checkpoint")) == ["step-30", "step-40", "step-50"]
    assert prune(CheckpointLayout(), str(run_dir), keep=2) == []


def test_prune_never_deletes_the_only_complete_checkpoint(run_dir):
    os.remove(run_dir / "checkpoint" / "step-20" / ".metadata")
    assert prune(CheckpointLayout(), str(run_dir), keep=1) == []
    with pytest.raises(ValueError):
        prune(CheckpointLayout(), str(run_dir), keep=0)


def test_scans_stop_at_the_newest_complete_checkpoint(tmp_path, monkeypatch):
    # A long run: 300 checkpoints, all complete. Finding the latest, setting
    # aside, and pruning must look at only as many as they need.
    for step in range(10, 3010, 10):
        (tmp_path / "checkpoint" / f"step-{step}").mkdir(parents=True)
    checked = []
    monkeypatch.setattr(checkpoints, "dcp_complete", lambda path: checked.append(path) or True)
    layout = CheckpointLayout()
    assert layout.latest_complete(str(tmp_path)) == 3000
    assert len(checked) == 1
    checked.clear()
    assert discard_incomplete(layout, str(tmp_path), tag="ta-1").discarded == []
    assert len(checked) == 1
    checked.clear()
    assert len(prune(layout, str(tmp_path), keep=3)) == 297
    assert len(checked) == 3


def test_older_incomplete_checkpoints_are_left_alone(run_dir):
    os.remove(run_dir / "checkpoint" / "step-10" / ".metadata")
    assert discard_incomplete(CheckpointLayout(), str(run_dir), tag="ta-1").discarded == []


def test_prune_tolerates_already_deleted_checkpoints_and_clears_old_discards(run_dir):
    _save(run_dir / "checkpoint" / "step-30")
    (run_dir / DISCARDED_SUBDIR / "step-5-ta-0").mkdir(parents=True)
    (run_dir / DISCARDED_SUBDIR / "step-35-ta-9").mkdir(parents=True)
    listed = CheckpointLayout().steps(str(run_dir))

    class StaleListing(CheckpointLayout):
        """Lists step-10 although the trainer has just deleted it."""

        def steps(self, run_dir: str) -> dict[int, str]:
            return listed

    shutil.rmtree(listed[10])
    assert prune(StaleListing(), str(run_dir), keep=1) == [10, 20]
    assert sorted(os.listdir(run_dir / DISCARDED_SUBDIR)) == ["step-35-ta-9"]
