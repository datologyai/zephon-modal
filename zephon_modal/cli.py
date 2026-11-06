# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""``zephon modal`` (or ``python -m zephon_modal``): launch, inspect and compare runs of a job file.

A job file is a Python module that defines an :class:`ElasticJob` (``job``
by default; pick another with ``path/to/file.py:name``).

:func:`register` is the ``zephon.cli`` entry point. ``zephon --help`` imports
every plugin to list it, so this module imports nothing heavy (Modal included)
until a command runs.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from zephon_modal.supervisor import Policy

if TYPE_CHECKING:
    from zephon_modal.job import ElasticJob

DEFAULT_JOB_ATTRIBUTE = "job"


def load_job(target: str) -> ElasticJob:
    """Load ``path/to/job.py[:attribute]``."""
    from zephon_modal.job import ElasticJob

    path_str, _, attribute = target.partition(":")
    path = Path(path_str).resolve()
    if not path.is_file():
        raise SystemExit(f"No job file at {path}")
    # So the job file can import its neighbours (recipes, images), as `modal run` allows.
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    job = getattr(module, attribute or DEFAULT_JOB_ATTRIBUTE, None)
    if not isinstance(job, ElasticJob):
        raise SystemExit(f"{target} doesn't name an ElasticJob")
    return job


def _placements(args: argparse.Namespace) -> int:
    job = load_job(args.job)
    for index, placement in enumerate(job.placements):
        dp = placement.dp_size(job.adapter.model_parallel_size)
        note = job.data.describe_dp_size(dp)
        print(
            f"{index}  {placement.name:<16} {placement.nodes} x {placement.gpu:<8} "
            f"world {placement.world_size:>3}  dp {dp:>3}" + (f"  {note}" if note else "")
        )
    return 0


def _launch(args: argparse.Namespace) -> int:
    job = load_job(args.job)
    checkpoints = job.launch(args.run_id, args.steps, placement=args.placement, seed=args.seed, detach=args.detach)
    print(f"{args.run_id} on {job.placement(args.placement).name}: checkpoints {checkpoints}")
    return 0


def _compare(args: argparse.Namespace) -> int:
    from zephon_modal.verify.compare import format_report, identical

    job = load_job(args.job)
    rows = job.compare(args.reference, args.candidate)
    print(format_report(rows))
    return 0 if identical(rows) else 1


def _supervise(args: argparse.Namespace) -> int:
    job = load_job(args.job)
    policy = Policy(
        queue_timeout=args.queue_timeout,
        max_restarts=args.max_restarts,
        max_failures=args.max_failures,
        cooldown=args.cooldown,
    )
    try:
        call_id = job.supervise(args.run_id, args.steps, seed=args.seed, policy=policy)
    except RuntimeError as error:
        print(f"zephon modal supervise: error: {error}", file=sys.stderr)
        return 1
    print(f"Supervising {args.run_id} to step {args.steps} (call {call_id}). Follow it with:")
    print(f"  zephon modal status {args.job} --run-id {args.run_id}")
    return 0


def format_status(record: dict) -> str:
    """A supervisor record as text."""
    state = record["status"] + ("" if record.get("alive") or record["status"] != "running" else " (supervisor gone)")
    lines = [f"{record['run_id']}: {state}, target step {record['steps']}"]
    if record.get("current"):
        current = record["current"]
        lines.append(f"current: {current['placement']} launch {current['launch_id']} since {current['started']}")
    cooling = record.get("cooling_until") or {}
    if cooling:
        lines.append("cooling down: " + ", ".join(f"{name} until {until}" for name, until in cooling.items()))
    for event in record.get("events", []):
        detail = ", ".join(f"{key}={value}" for key, value in event.items() if key not in ("time", "event"))
        lines.append(f"  {event['time']}  {event['event']:<18} {detail}")
    return "\n".join(lines)


def _status(args: argparse.Namespace) -> int:
    record = load_job(args.job).status(args.run_id)
    if record is None:
        print(f"{args.run_id} has never been supervised")
        return 1
    print(format_status(record))
    return 0


def _cancel(args: argparse.Namespace) -> int:
    if load_job(args.job).cancel(args.run_id):
        print(f"Cancelled {args.run_id}'s supervisor; its launch stops and commits")
        return 0
    print(f"{args.run_id} has no running supervisor")
    return 1


def register(parser: argparse.ArgumentParser) -> None:
    """Add the subcommands to ``parser``: the ``zephon.cli`` entry point behind ``zephon modal``.

    Each subcommand sets ``func`` to a handler that returns an exit code.
    """
    parser.description = "Launch, inspect and compare elastic training runs on Modal."
    commands = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)

    placements = commands.add_parser("placements", help="List a job's placements and their DP degrees")
    placements.add_argument("job", help="path/to/job.py[:attribute]")
    placements.set_defaults(func=_placements)

    launch = commands.add_parser("launch", help="Train a run on one placement, resuming it if it exists")
    launch.add_argument("job", help="path/to/job.py[:attribute]")
    launch.add_argument("--run-id", required=True, help="Reusing an ID resumes that run")
    launch.add_argument("--steps", type=int, required=True, help="Total steps to reach, not steps to add")
    launch.add_argument("--placement", help="Placement name (see `placements`); defaults to the first")
    launch.add_argument("--seed", type=int, default=42, help="Zephon seed; must stay fixed for a run")
    launch.add_argument(
        "--detach", action="store_true", help="Keep running if this client disconnects (use for long runs)"
    )
    launch.set_defaults(func=_launch)

    supervise = commands.add_parser(
        "supervise", help="Train a run to its step target, moving between placements as needed (detached)"
    )
    supervise.add_argument("job", help="path/to/job.py[:attribute]")
    supervise.add_argument("--run-id", required=True, help="Reusing an ID resumes that run")
    supervise.add_argument("--steps", type=int, required=True, help="Total steps to reach, not steps to add")
    supervise.add_argument("--seed", type=int, default=42, help="Zephon seed; must stay fixed for a run")
    supervise.add_argument(
        "--queue-timeout",
        type=float,
        default=Policy.queue_timeout,
        help="Seconds to wait for containers before trying the next placement",
    )
    supervise.add_argument(
        "--max-restarts", type=int, default=Policy.max_restarts, help="Preemptions on one launch before moving on"
    )
    supervise.add_argument(
        "--max-failures", type=int, default=Policy.max_failures, help="Failed launches in a row before giving up"
    )
    supervise.add_argument(
        "--cooldown", type=float, default=Policy.cooldown, help="Seconds a placement is skipped after giving up on it"
    )
    supervise.set_defaults(func=_supervise)

    status = commands.add_parser("status", help="Show a supervised run's placements and events")
    status.add_argument("job", help="path/to/job.py[:attribute]")
    status.add_argument("--run-id", required=True)
    status.set_defaults(func=_status)

    cancel = commands.add_parser("cancel", help="Stop a supervised run (its launch commits on the way out)")
    cancel.add_argument("job", help="path/to/job.py[:attribute]")
    cancel.add_argument("--run-id", required=True)
    cancel.set_defaults(func=_cancel)

    compare = commands.add_parser("compare", help="Check two runs trained on the same samples at every step")
    compare.add_argument("job", help="path/to/job.py[:attribute]")
    compare.add_argument("--reference", required=True)
    compare.add_argument("--candidate", required=True)
    compare.set_defaults(func=_compare)


def main(argv: list[str] | None = None) -> int:
    """``python -m zephon_modal``: run one subcommand and return its exit code."""
    parser = argparse.ArgumentParser(prog="python -m zephon_modal")
    register(parser)
    args = parser.parse_args(argv)
    return args.func(args)
