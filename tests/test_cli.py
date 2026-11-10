import argparse
import subprocess
import sys
from types import SimpleNamespace

from zephon_modal import cli
from zephon_modal.verify.compare import MATCH, MISMATCH


def test_register_adds_subcommands_with_handlers():
    parser = argparse.ArgumentParser(prog="zephon modal")
    cli.register(parser)
    args = parser.parse_args(["launch", "job.py", "--run-id", "r", "--steps", "10"])
    assert args.func is cli._launch
    assert (args.run_id, args.steps, args.seed, args.detach) == ("r", 10, 42, False)


def test_cli_imports_without_modal():
    # `zephon --help` imports every plugin; Modal's ~0.5 s import waits until a command runs.
    code = "import sys, zephon_modal.cli; print('modal' in sys.modules)"
    assert subprocess.check_output([sys.executable, "-c", code], text=True).strip() == "False"


def test_compare_exit_code_says_whether_runs_match(monkeypatch, capsys):
    rows = [{"step": 1, "reference_dp": 8, "candidate_dp": 4, "rows": 16, "status": MATCH}]
    job = SimpleNamespace(compare=lambda reference, candidate: rows)
    monkeypatch.setattr(cli, "load_job", lambda target: job)
    assert cli.main(["compare", "job.py", "--reference", "a", "--candidate", "b"]) == 0
    rows[0]["status"] = MISMATCH
    assert cli.main(["compare", "job.py", "--reference", "a", "--candidate", "b"]) == 1


def test_start_builds_a_policy(monkeypatch):
    started = {}
    job = SimpleNamespace(start=lambda run_id, steps, **kwargs: started.update(run_id=run_id, steps=steps, **kwargs))
    monkeypatch.setattr(cli, "load_job", lambda target: job)
    argv = ["start", "job.py", "--run-id", "r", "--steps", "100", "--keep-checkpoints", "3", "--no-deploy"]
    assert cli.main(argv) == 0
    assert (started["run_id"], started["steps"], started["deploy"]) == ("r", 100, False)
    assert started["policy"].keep_checkpoints == 3


def test_format_status_shows_current_launch_and_events():
    record = {
        "run_id": "r",
        "status": "running",
        "steps": 100,
        "checkpoint": 20,
        "current": {"placement": "t4x2", "launch_id": "L1", "call_id": "fc-1"},
        "cooling_until": {"t4x1": "2026-11-09T00:00:00+00:00"},
        "events": [{"time": "2026-11-09T00:00:00+00:00", "event": "launched", "placement": "t4x2"}],
    }
    text = cli.format_status(record)
    assert "r: running, target step 100, checkpoint 20" in text
    assert "t4x2 launch L1 (call fc-1)" in text and "launched" in text


def test_replays_exit_code_says_whether_replays_match(monkeypatch, capsys):
    rows = [{"step": 5, "attempts": ["a1", "a2"], "dps": [4, 2], "rows": 16, "status": MATCH}]
    job = SimpleNamespace(replays=lambda run_id: rows)
    monkeypatch.setattr(cli, "load_job", lambda target: job)
    assert cli.main(["replays", "job.py", "--run-id", "r"]) == 0
    assert "1/1 replayed steps identical" in capsys.readouterr().out
    rows[0]["status"] = MISMATCH
    assert cli.main(["replays", "job.py", "--run-id", "r"]) == 1


def _attempt(**fields):
    return {"task_id": "ta-1", "node_rank": 0, "started": "2026-11-09T00:00:00+00:00", "outcome": "running"} | fields


def test_should_stop_once_a_running_attempt_checkpoints_at_or_past_a_pending_step():
    attempt = _attempt(resume_from=None)
    assert not cli.should_stop([40, 120], 30, attempt)
    assert cli.should_stop([40, 120], 40, attempt)
    assert cli.should_stop([40, 120], 50, attempt)
    assert not cli.should_stop([], 50, attempt)
    assert not cli.should_stop([40], None, attempt)


def test_should_stop_waits_until_the_attempt_trains_past_where_it_resumed():
    # Restarted at 40 after the first stop: checkpoint 40 is the one it resumed from.
    assert not cli.should_stop([40], 40, _attempt(resume_from=40))
    assert cli.should_stop([40], 50, _attempt(resume_from=40))
    # Its first record predates knowing where it resumed.
    assert not cli.should_stop([40], 50, _attempt(outcome="starting"))
    assert not cli.should_stop([40], 50, None)


def test_should_stop_leaves_ended_attempts_alone():
    for outcome in ("completed", "failed", "stopped", "cancelled", "superseded"):
        assert not cli.should_stop([40], 50, _attempt(outcome=outcome, resume_from=0))


def test_running_attempt_is_the_latest_node_0_record_without_an_end():
    first = _attempt(task_id="ta-1", started="2026-11-09T00:00:00+00:00", ended="2026-11-09T00:10:00+00:00")
    restarted = _attempt(task_id="ta-2", started="2026-11-09T00:11:00+00:00")
    node_1 = _attempt(task_id="ta-3", node_rank=1, started="2026-11-09T00:12:00+00:00")
    assert cli.running_attempt([first, restarted, node_1]) == restarted
    assert cli.running_attempt([first, node_1]) is None
    assert cli.running_attempt([]) is None


def test_stop_after_parses_steps_and_timeout():
    parser = argparse.ArgumentParser(prog="zephon modal")
    cli.register(parser)
    args = parser.parse_args(["stop-after", "job.py", "--run-id", "r", "--after", "40", "120"])
    assert args.func is cli._stop_after
    assert (args.run_id, args.after, args.timeout) == ("r", [40, 120], 3 * 60 * 60)
