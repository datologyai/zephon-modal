import json
import os

import pytest

from zephon_modal.verify.compare import MATCH, REFERENCE_ONLY, compare_runs, format_report, identical
from zephon_modal.verify.hashes import SAMPLE_HASH_SUBDIR, StepHashRecorder, hash_file_name

# 8 global microbatches per step, 2 rows each: 16 rows per step.
MICROBATCHES_PER_STEP = 8


def _rows(step: int, microbatch: int) -> list[bytes]:
    return [f"s{step}-m{microbatch}-r{r}".encode() for r in range(2)]


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
    path = tmp_path / "cand" / SAMPLE_HASH_SUBDIR / hash_file_name("20261102T000000", 2, 0)
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
