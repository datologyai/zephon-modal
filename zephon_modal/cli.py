# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""``zephon modal`` (or ``python -m zephon_modal``): start, follow and compare runs of a job file.

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
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from zephon_modal.reconcile import Policy

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
        print(f"{index}  {placement.name}")
        # A placement with fallbacks is one Modal function; Modal picks the first shape it can allocate.
        for rank, shape in enumerate(placement.shapes):
            dp = shape.dp_size(job.adapter.model_parallel_size)
            note = job.data.describe_dp_size(dp)
            print(
                f"   {'or ' if rank else '   '}{shape.nodes} x {shape.gpu:<8} "
                f"world {shape.world_size:>3}  dp {dp:>3}" + (f"  {note}" if note else "")
            )
    return 0


def _deploy(args: argparse.Namespace) -> int:
    load_job(args.job).deploy()
    return 0


def _start(args: argparse.Namespace) -> int:
    job = load_job(args.job)
    policy = Policy(
        queue_timeout=args.queue_timeout,
        max_restarts=args.max_restarts,
        cooldown=args.cooldown,
        stall_timeout=args.stall_timeout,
        max_failures=args.max_failures,
        max_lost=args.max_lost,
        keep_checkpoints=args.keep_checkpoints,
        max_gpu_hours=args.max_gpu_hours,
    )
    try:
        job.start(args.run_id, args.steps, seed=args.seed, policy=policy, deploy=not args.no_deploy)
    except RuntimeError as error:
        print(f"zephon modal start: error: {error}", file=sys.stderr)
        return 1
    print(f"Started {args.run_id}, to step {args.steps}. Follow it with:")
    print(f"  zephon modal status {args.job} --run-id {args.run_id}")
    return 0


def format_status(record: dict[str, Any]) -> str:
    """A reconciler record as text."""
    lines = [
        f"{record['run_id']}: {record['status']}, target step {record['steps']}, checkpoint {record['checkpoint']}"
    ]
    if record.get("current"):
        current = record["current"]
        lines.append(f"current: {current['placement']} launch {current['launch_id']} (call {current['call_id']})")
    cooling = record.get("cooling_until") or {}
    if cooling:
        lines.append("cooling down: " + ", ".join(f"{name} until {until}" for name, until in cooling.items()))
    for event in record.get("events", [])[-30:]:
        detail = ", ".join(f"{key}={value}" for key, value in event.items() if key not in ("time", "event"))
        lines.append(f"  {event['time']}  {event['event']:<20} {detail}")
    return "\n".join(lines)


def _status(args: argparse.Namespace) -> int:
    record = load_job(args.job).status(args.run_id)
    if record is None:
        print(f"{args.run_id} has never been started")
        return 1
    print(format_status(record))
    return 0


def _stop(args: argparse.Namespace) -> int:
    try:
        load_job(args.job).stop(args.run_id)
    except RuntimeError as error:
        print(f"zephon modal stop: error: {error}", file=sys.stderr)
        return 1
    print(f"Stopping {args.run_id}: its launch stops its trainer and commits; check `status`")
    return 0


def _recover(args: argparse.Namespace) -> int:
    job = load_job(args.job)
    record = job.status(args.run_id) or {}
    current = record.get("current") or {}
    if not args.yes:
        print(
            f"{args.run_id}'s launch {current.get('launch_id')} (call {current.get('call_id')}) never confirmed it stopped."
        )
        print("Recover only once none of its containers is running (Modal dashboard, or `modal container list`).")
        if input("Continue? [y/N] ").strip().lower() != "y":
            return 1
    try:
        job.recover(args.run_id)
    except RuntimeError as error:
        print(f"zephon modal recover: error: {error}", file=sys.stderr)
        return 1
    print(f"{args.run_id} will continue on the next reconcile")
    return 0


def _launch(args: argparse.Namespace) -> int:
    job = load_job(args.job)
    try:
        checkpoints = job.launch(args.run_id, args.steps, placement=args.placement, seed=args.seed, detach=args.detach)
    except RuntimeError as error:
        print(f"zephon modal launch: error: {error}", file=sys.stderr)
        return 1
    print(f"{args.run_id} on {job.placement(args.placement).name}: checkpoints {checkpoints}")
    return 0


def _replays(args: argparse.Namespace) -> int:
    from zephon_modal.verify.compare import MISMATCH, format_replays

    rows = load_job(args.job).replays(args.run_id)
    print(format_replays(rows))
    return 1 if any(row["status"] == MISMATCH for row in rows) else 0


def _compare(args: argparse.Namespace) -> int:
    from zephon_modal.verify.compare import format_report, identical

    job = load_job(args.job)
    options = {}
    if args.steps is not None:
        options = {"steps": args.steps, "start_step": args.start_step, "rows_per_step": args.rows_per_step}
    elif args.rows_per_step is not None or args.start_step != 1:
        raise SystemExit("--start-step and --rows-per-step require --steps")
    rows = job.compare(args.reference, args.candidate, **options)
    print(format_report(rows))
    return 0 if identical(rows, strict=args.steps is not None) else 1


STOP_AFTER_POLL_SECONDS = 10
# After a checkpoint appears, give its save (every node's shards) time to finish first.
STOP_AFTER_SETTLE_SECONDS = 20
# What `ElasticJob.status` reports for a run that may still train.
ACTIVE_STATUSES = ("start requested", "running", "draining", "needs recovery")


def running_attempt(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """The latest node-0 attempt record among one launch's ``records`` whose container hasn't recorded an end."""
    running = [record for record in records if record.get("node_rank") == 0 and "ended" not in record]
    return max(running, key=lambda record: record["started"]) if running else None


def should_stop(pending: Sequence[int], checkpoint: int | None, attempt: Mapping[str, Any] | None) -> bool:
    """Whether ``stop-after`` should stop ``attempt``'s container now.

    Args:
        pending: Checkpoint steps not yet stopped after.
        checkpoint: The run's latest complete checkpoint, if any.
        attempt: The current launch's running node-0 attempt record (see :func:`running_attempt`).

    Returns:
        True once the attempt is training (its first record predates knowing
        where it resumed), has saved a checkpoint past the one it resumed
        from, so every placement trains before it's stopped, and that
        checkpoint is at or past the earliest pending step.
    """
    if not pending or checkpoint is None or attempt is None or attempt.get("outcome") != "running":
        return False
    return checkpoint > (attempt.get("resume_from") or 0) and checkpoint >= min(pending)


def latest_complete_checkpoint(job: ElasticJob, run_id: str) -> int | None:
    """The run's newest checkpoint on the runs Volume (committed view) whose DCP ``.metadata`` has been written."""
    import re

    import modal

    layout = job.adapter.checkpoints
    assert layout is not None
    try:
        entries = job.runs_volume.listdir(f"{run_id}/{layout.subdir}")
    except (FileNotFoundError, modal.exception.NotFoundError):
        return None
    steps = {
        int(match.group(1)): entry.path
        for entry in entries
        if (match := re.fullmatch(layout.step_pattern, entry.path.rsplit("/", 1)[-1]))
    }
    for step in sorted(steps, reverse=True):
        try:
            names = {entry.path.rsplit("/", 1)[-1] for entry in job.runs_volume.listdir(steps[step])}
        except (FileNotFoundError, modal.exception.NotFoundError):
            continue
        # DCP writes .metadata after the shards; the settle wait covers other nodes' commits.
        if ".metadata" in names:
            return step
    return None


def _stop_after(args: argparse.Namespace) -> int:
    import subprocess
    import time

    import modal

    job = load_job(args.job)
    if job.adapter.checkpoints is None:
        raise SystemExit(f"{args.job}'s adapter doesn't say where its checkpoints are")
    registry = job.registry()
    pending = sorted(set(args.after))
    deadline = time.monotonic() + args.timeout
    while pending and time.monotonic() < deadline:
        try:
            record = job.status(args.run_id) or {}
            if record.get("status") not in ACTIVE_STATUSES:
                print(f"{args.run_id} is {record.get('status') or 'not started'}; {pending} not reached", flush=True)
                return 1
            current = record.get("current") or {}
            checkpoint = latest_complete_checkpoint(job, args.run_id)
            attempt = running_attempt(registry.launch_records(args.run_id, current["launch_id"])) if current else None
        except (modal.exception.Error, OSError) as error:
            # Modal couldn't be asked this time; a long rehearsal shouldn't end over it.
            print(f"couldn't read {args.run_id}, retrying: {error!r}", flush=True)
            time.sleep(STOP_AFTER_POLL_SECONDS)
            continue
        if attempt is not None and should_stop(pending, checkpoint, attempt):
            assert checkpoint is not None
            task = attempt["task_id"]
            time.sleep(STOP_AFTER_SETTLE_SECONDS)
            print(f"checkpoint {checkpoint} on {current['placement']}: stopping {task}", flush=True)
            subprocess.run([sys.executable, "-m", "modal", "container", "stop", "--yes", task], check=False)
            pending = [step for step in pending if step > checkpoint]
            continue
        time.sleep(STOP_AFTER_POLL_SECONDS)
    if pending:
        print(f"Timed out with {pending} not reached", flush=True)
        return 1
    return 0


def register(parser: argparse.ArgumentParser) -> None:
    """Add the subcommands to ``parser``: the ``zephon.cli`` entry point behind ``zephon modal``.

    Each subcommand sets ``func`` to a handler that returns an exit code.
    """
    parser.description = "Start, follow and compare elastic training runs on Modal."
    commands = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)

    def command(name: str, help: str, func: Any, run_id: bool = True) -> argparse.ArgumentParser:
        sub = commands.add_parser(name, help=help)
        sub.add_argument("job", help="path/to/job.py[:attribute]")
        if run_id:
            sub.add_argument("--run-id", required=True)
        sub.set_defaults(func=func)
        return sub

    command("placements", "List a job's placements and their DP degrees", _placements, run_id=False)
    command("deploy", "Deploy the job's app (its reconciler and training functions)", _deploy, run_id=False)

    start = command("start", "Deploy, then train a run to its step target across placements", _start)
    start.add_argument("--steps", type=int, required=True, help="Total steps to reach, not steps to add")
    start.add_argument("--seed", type=int, default=42, help="Zephon seed; must stay fixed for a run")
    start.add_argument("--no-deploy", action="store_true", help="Use the app as last deployed")
    defaults = Policy()
    start.add_argument(
        "--queue-timeout",
        type=float,
        default=defaults.queue_timeout,
        help="Seconds to wait for containers before moving on",
    )
    start.add_argument(
        "--max-restarts", type=int, default=defaults.max_restarts, help="Preemptions of one launch before moving on"
    )
    start.add_argument(
        "--cooldown", type=float, default=defaults.cooldown, help="Seconds a placement is skipped after moving off it"
    )
    start.add_argument(
        "--stall-timeout",
        type=float,
        default=defaults.stall_timeout,
        help="Seconds running without a new checkpoint before relaunching",
    )
    start.add_argument(
        "--max-failures",
        type=int,
        default=defaults.max_failures,
        help="Failures in a row without progress before giving up",
    )
    start.add_argument(
        "--max-lost",
        type=int,
        default=defaults.max_lost,
        help="Launches in a row Modal loses to internal failures, without progress, before giving up",
    )
    start.add_argument("--keep-checkpoints", type=int, help="Delete all but this many latest complete checkpoints")
    start.add_argument("--max-gpu-hours", type=float, help="Give up after this many GPU hours in total")

    command("status", "Show a run's state and recent events", _status)
    command("stop", "Stop a run (its launch commits on the way out)", _stop)
    recover = command("recover", "Continue a run whose last launch never confirmed it stopped", _recover)
    recover.add_argument("--yes", action="store_true", help="Skip the confirmation")

    launch = command(
        "launch", "Train a run on one placement now, without the reconciler (references, experiments)", _launch
    )
    launch.add_argument("--steps", type=int, required=True, help="Total steps to reach, not steps to add")
    launch.add_argument("--placement", help="Placement name (see `placements`); defaults to the first")
    launch.add_argument("--seed", type=int, default=42, help="Zephon seed; must stay fixed for a run")
    launch.add_argument("--detach", action="store_true", help="Keep running if this client disconnects")

    command(
        "replays",
        "Check a run against itself: steps re-trained after a resume must match their first training",
        _replays,
    )

    stop_after = command(
        "stop-after",
        "Rehearse preemptions: stop the current launch's container after given checkpoints",
        _stop_after,
    )
    stop_after.add_argument(
        "--after", type=int, nargs="+", required=True, help="Checkpoint steps to stop after, once each"
    )
    stop_after.add_argument(
        "--timeout", type=float, default=3 * 60 * 60, help="Give up after this many seconds (default: 3 hours)"
    )

    compare = commands.add_parser("compare", help="Check two runs trained on the same samples at every step")
    compare.add_argument("job", help="path/to/job.py[:attribute]")
    compare.add_argument("--reference", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--steps", type=int, help="Require complete matching evidence through this step")
    compare.add_argument("--start-step", type=int, default=1, help="First required step (default: 1)")
    compare.add_argument("--rows-per-step", type=int, help="Required global sequence count per step")
    compare.set_defaults(func=_compare)


def main(argv: list[str] | None = None) -> int:
    """``python -m zephon_modal``: run one subcommand and return its exit code."""
    parser = argparse.ArgumentParser(prog="python -m zephon_modal")
    register(parser)
    args = parser.parse_args(argv)
    return args.func(args)
