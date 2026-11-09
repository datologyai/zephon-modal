import json
import os

import pytest

from zephon_modal.verify import hashes
from zephon_modal.verify.compare import (
    INCOMPLETE,
    MATCH,
    MISMATCH,
    REFERENCE_ONLY,
    check_replays,
    compare_runs,
    format_replays,
    format_report,
    identical,
)
from zephon_modal.verify.hashes import SAMPLE_HASH_SUBDIR, StepHashRecorder, hash_file_name, segment_path

# 8 global microbatches per step, 2 rows each: 16 rows per step.
MICROBATCHES_PER_STEP = 8


def _rows(step: int, microbatch: int) -> list[bytes]:
    return [f"s{step}-m{microbatch}-r{r}".encode() for r in range(2)]


def _file(run_dir, launch_id: str, dp: int, rank: int, step: int = 1):
    """The hash file holding ``step`` for one rank."""
    return run_dir / SAMPLE_HASH_SUBDIR / os.path.basename(segment_path(hash_file_name(launch_id, dp, rank), step))


def _train(run_dir, launch_id: str, dp: int, steps: range, microbatch_order=lambda mbs: mbs) -> None:
    """Simulate `dp` ranks splitting each step's global microbatches between them."""
    hash_dir = os.path.join(run_dir, SAMPLE_HASH_SUBDIR)
    recorders = [
        StepHashRecorder(
            os.path.join(hash_dir, hash_file_name(launch_id, dp, rank)),
            dp_world_size=dp,
            dp_rank=rank,
            microbatches_per_step=MICROBATCHES_PER_STEP,
            consumed_microbatches=steps.start * MICROBATCHES_PER_STEP,
        )
        for rank in range(dp)
    ]
    for step in steps:
        microbatches = microbatch_order(list(range(MICROBATCHES_PER_STEP)))
        for index, microbatch in enumerate(microbatches):
            recorders[index % dp].record(_rows(step, microbatch))


def test_same_samples_on_different_dp_match(tmp_path):
    _train(tmp_path / "ref", "20261102T000000", dp=8, steps=range(4))
    _train(tmp_path / "cand", "20261102T000000", dp=4, steps=range(2))
    # Resume at step 2 on 2 ranks, in a different order within each step.
    _train(tmp_path / "cand", "20261102T010000", dp=2, steps=range(2, 4), microbatch_order=lambda mbs: mbs[::-1])
    rows = compare_runs(str(tmp_path / "ref"), str(tmp_path / "cand"))
    assert [row["step"] for row in rows] == [1, 2, 3, 4]
    assert identical(rows)
    assert [row["candidate_dp"] for row in rows] == [4, 4, 2, 2]


def test_latest_launch_wins_for_retrained_steps(tmp_path):
    _train(tmp_path / "ref", "20261102T000000", dp=2, steps=range(2))
    # A first launch trained step 2 with the wrong data, then crashed.
    _train(tmp_path / "cand", "20261102T000000", dp=2, steps=range(2))
    path = _file(tmp_path / "cand", "20261102T000000", 2, 0)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[-1]["row_hashes"][0] = "bad"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    assert not identical(compare_runs(str(tmp_path / "ref"), str(tmp_path / "cand")))
    # The relaunch retrains step 2 from the step-1 checkpoint.
    _train(tmp_path / "cand", "20261102T010000", dp=2, steps=range(1, 2))
    assert identical(compare_runs(str(tmp_path / "ref"), str(tmp_path / "cand")))


def test_mid_step_restore_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="mid-step"):
        StepHashRecorder(
            str(tmp_path / "h.jsonl"), dp_world_size=2, dp_rank=0, microbatches_per_step=8, consumed_microbatches=3
        )


def test_shorter_candidate_checks_shared_steps(tmp_path):
    _train(tmp_path / "ref", "20261102T000000", dp=4, steps=range(4))
    _train(tmp_path / "cand", "20261102T000000", dp=2, steps=range(2))
    rows = compare_runs(str(tmp_path / "ref"), str(tmp_path / "cand"))
    assert [row["status"] for row in rows] == [MATCH, MATCH, REFERENCE_ONLY, REFERENCE_ONLY]
    assert identical(rows)
    assert format_report(rows).endswith("2/2 shared steps identical (2 only in reference)")


def test_disjoint_runs_are_not_identical(tmp_path):
    _train(tmp_path / "ref", "20261102T000000", dp=2, steps=range(2))
    _train(tmp_path / "cand", "20261102T000000", dp=2, steps=range(2, 4))
    assert not identical(compare_runs(str(tmp_path / "ref"), str(tmp_path / "cand")))


def test_restarted_attempt_appends_and_its_records_win(tmp_path):
    _train(tmp_path / "ref", "20261102T000000", dp=2, steps=range(8))
    # Attempt 1 trains steps 0-6, is stopped, and its checkpoint is at step 5.
    # Attempt 2 of the same launch appends steps 5-8 to the same files; the
    # stale step-6 record before it must not double-count.
    _train(tmp_path / "cand", "20261102T010000", dp=2, steps=range(6))
    _train(tmp_path / "cand", "20261102T010000", dp=2, steps=range(5, 6), microbatch_order=lambda mbs: mbs[:4] * 2)
    _train(tmp_path / "cand", "20261102T010000", dp=2, steps=range(5, 8))
    rows = compare_runs(str(tmp_path / "ref"), str(tmp_path / "cand"))
    assert [row["status"] for row in rows] == [MATCH] * 8
    assert {row["rows"] for row in rows} == {16}


def test_strict_comparison_rejects_missing_steps_including_both_runs(tmp_path):
    _train(tmp_path / "ref", "20261102T000000", dp=2, steps=range(2))
    _train(tmp_path / "cand", "20261102T000000", dp=2, steps=range(1))
    rows = compare_runs(str(tmp_path / "ref"), str(tmp_path / "cand"), steps=3)
    assert [row["status"] for row in rows] == [MATCH, REFERENCE_ONLY, INCOMPLETE]
    assert not identical(rows, strict=True)


def test_same_missing_rank_in_both_runs_is_incomplete_not_a_match(tmp_path):
    for run in ("ref", "cand"):
        _train(tmp_path / run, "20261102T000000", dp=2, steps=range(2))
        _file(tmp_path / run, "20261102T000000", 2, 1).unlink()
    rows = compare_runs(str(tmp_path / "ref"), str(tmp_path / "cand"), steps=2)
    assert [row["status"] for row in rows] == [INCOMPLETE, INCOMPLETE]
    assert not identical(rows)


def test_strict_comparison_checks_global_sequence_count(tmp_path):
    for run, dp in (("ref", 4), ("cand", 2)):
        _train(tmp_path / run, "20261102T000000", dp=dp, steps=range(2))
    assert identical(
        compare_runs(str(tmp_path / "ref"), str(tmp_path / "cand"), steps=2, rows_per_step=16), strict=True
    )
    assert not identical(
        compare_runs(str(tmp_path / "ref"), str(tmp_path / "cand"), steps=2, rows_per_step=32), strict=True
    )


def test_incomplete_new_attempt_does_not_borrow_an_older_ranks_records(tmp_path):
    _train(tmp_path / "ref", "20261102T000000", dp=2, steps=range(2))
    _train(tmp_path / "cand", "20261102T010000.a000001-x", dp=2, steps=range(2))
    _train(tmp_path / "cand", "20261102T010000.a000002-y", dp=2, steps=range(1, 2))
    _file(tmp_path / "cand", "20261102T010000.a000002-y", 2, 1).unlink()
    rows = compare_runs(str(tmp_path / "ref"), str(tmp_path / "cand"), steps=2)
    assert [row["status"] for row in rows] == [MATCH, INCOMPLETE]


def test_strict_comparison_of_empty_directories_is_not_success(tmp_path):
    rows = compare_runs(str(tmp_path / "ref"), str(tmp_path / "cand"), steps=2)
    assert not identical(rows, strict=True)


def test_hash_files_split_into_segments_and_still_compare(tmp_path, monkeypatch):
    monkeypatch.setattr(hashes, "SEGMENT_STEPS", 2)
    _train(tmp_path / "ref", "20261102T000000", dp=4, steps=range(5))
    _train(tmp_path / "cand", "20261102T000000", dp=2, steps=range(5))
    assert sorted(os.listdir(tmp_path / "cand" / SAMPLE_HASH_SUBDIR))[:3] == [
        "20261102T000000-dp2-rank000-s00000.jsonl",
        "20261102T000000-dp2-rank000-s00001.jsonl",
        "20261102T000000-dp2-rank000-s00002.jsonl",
    ]
    assert identical(
        compare_runs(str(tmp_path / "ref"), str(tmp_path / "cand"), steps=5, rows_per_step=16), strict=True
    )


def _copy_as_model_parallel_peers(run_dir, launch_id: str, dp: int, peers: int) -> None:
    """Give each DP rank ``peers`` global ranks recording the same rows, as tensor parallelism does."""
    hash_dir = run_dir / SAMPLE_HASH_SUBDIR
    originals = {dp_rank: _file(run_dir, launch_id, dp, dp_rank).read_text() for dp_rank in range(dp)}
    for dp_rank in range(dp):
        _file(run_dir, launch_id, dp, dp_rank).unlink()
    for dp_rank, rows in originals.items():
        for peer in range(peers):
            global_rank = dp_rank * peers + peer
            (hash_dir / os.path.basename(segment_path(hash_file_name(launch_id, dp, global_rank), 1))).write_text(rows)


def test_model_parallel_peers_count_once(tmp_path):
    _train(tmp_path / "ref", "20261102T000000", dp=4, steps=range(3))
    _train(tmp_path / "cand", "20261102T000000", dp=2, steps=range(3))
    _copy_as_model_parallel_peers(tmp_path / "cand", "20261102T000000", dp=2, peers=2)
    rows = compare_runs(str(tmp_path / "ref"), str(tmp_path / "cand"), steps=3, rows_per_step=16)
    assert identical(rows, strict=True)


def test_model_parallel_peers_that_disagree_are_incomplete(tmp_path):
    _train(tmp_path / "ref", "20261102T000000", dp=2, steps=range(1))
    _train(tmp_path / "cand", "20261102T000000", dp=2, steps=range(1))
    _copy_as_model_parallel_peers(tmp_path / "cand", "20261102T000000", dp=2, peers=2)
    peer = (
        tmp_path
        / "cand"
        / SAMPLE_HASH_SUBDIR
        / os.path.basename(segment_path(hash_file_name("20261102T000000", 2, 1), 1))
    )
    record = json.loads(peer.read_text())
    record["row_hashes"][0] = "different"
    peer.write_text(json.dumps(record) + "\n")
    rows = compare_runs(str(tmp_path / "ref"), str(tmp_path / "cand"), steps=1)
    assert [row["status"] for row in rows] == [INCOMPLETE]


def test_replayed_steps_check_a_run_against_itself(tmp_path):
    run = tmp_path / "run"
    # Attempt 1 at DP4 trains steps 1-6 and is stopped; its checkpoint is at step 4.
    _train(run, "20261109T000000.a0001", dp=4, steps=range(6))
    # Attempt 2 at DP2 resumes from step 4 and re-trains 5 and 6.
    _train(run, "20261109T000000.a0002", dp=2, steps=range(4, 8))
    rows = check_replays(str(run))
    assert [(row["step"], row["dps"], row["status"]) for row in rows] == [(5, [4, 2], MATCH), (6, [4, 2], MATCH)]
    assert format_replays(rows).endswith("2/2 replayed steps identical")


def test_a_replay_that_differs_is_a_mismatch(tmp_path):
    run = tmp_path / "run"
    _train(run, "20261109T000000.a0001", dp=2, steps=range(3))
    _train(run, "20261109T000000.a0002", dp=2, steps=range(2, 3), microbatch_order=lambda mbs: mbs[:4] * 2)
    assert [row["status"] for row in check_replays(str(run))] == [MISMATCH]


def test_a_partial_step_from_a_stopped_attempt_is_skipped(tmp_path):
    run = tmp_path / "run"
    _train(run, "20261109T000000.a0001", dp=2, steps=range(3))
    _file(run, "20261109T000000.a0001", 2, 1).unlink()  # one rank's records never synced
    _train(run, "20261109T000000.a0002", dp=2, steps=range(2, 3))
    assert check_replays(str(run)) == []
    assert format_replays([]) == "No replayed steps yet"
