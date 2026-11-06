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
