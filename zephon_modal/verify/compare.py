# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Compare per-step sample digests between runs.

Two runs agree at a step when their DP ranks trained on exactly the same
multiset of packed sequences, regardless of GPU count, rank or order.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from zephon_modal.verify.hashes import SAMPLE_HASH_SUBDIR

MATCH = "match"
MISMATCH = "MISMATCH"
REFERENCE_ONLY = "only in reference"
CANDIDATE_ONLY = "only in candidate"
STATUSES = (MATCH, MISMATCH, REFERENCE_ONLY, CANDIDATE_ONLY)


def load_step_digests(sample_hash_dir: str) -> dict[int, dict[str, Any]]:
    """Fold one run's hash files into ``{step: {digest, rows, dp}}``.

    A step re-trained after a resume appears in several launches; launch IDs
    sort chronologically, so the latest launch's record is the one that counts.
    """
    per_step: dict[int, dict[str, Any]] = {}
    for name in sorted(os.listdir(sample_hash_dir)):
        launch_id = name.split("-dp")[0]
        with open(os.path.join(sample_hash_dir, name)) as hash_file:
            for line in hash_file:
                record = json.loads(line)
                entry = per_step.get(record["step"])
                if entry is None or entry["launch_id"] < launch_id:
                    entry = per_step[record["step"]] = {"launch_id": launch_id, "dp": record["dp"], "rows": []}
                if entry["launch_id"] == launch_id:
                    entry["rows"].extend(record["row_hashes"])
    return {
        step: {
            "digest": hashlib.sha256("".join(sorted(entry["rows"])).encode()).hexdigest()[:16],
            "rows": len(entry["rows"]),
            "dp": entry["dp"],
        }
        for step, entry in sorted(per_step.items())
    }


def compare_runs(reference_dir: str, candidate_dir: str) -> list[dict[str, Any]]:
    """Compare two run directories step by step."""
    ref = load_step_digests(os.path.join(reference_dir, SAMPLE_HASH_SUBDIR))
    cand = load_step_digests(os.path.join(candidate_dir, SAMPLE_HASH_SUBDIR))
    return [
        {
            "step": step,
            "reference_dp": ref.get(step, {}).get("dp"),
            "candidate_dp": cand.get(step, {}).get("dp"),
            "rows": cand.get(step, ref.get(step, {})).get("rows"),
            "status": _status(ref.get(step), cand.get(step)),
        }
        for step in sorted(ref.keys() | cand.keys())
    ]


def _status(ref: dict[str, Any] | None, cand: dict[str, Any] | None) -> str:
    if ref is None:
        return CANDIDATE_ONLY
    if cand is None:
        return REFERENCE_ONLY
    return MATCH if ref["digest"] == cand["digest"] else MISMATCH


def identical(rows: list[dict[str, Any]]) -> bool:
    """Whether the runs share at least one step and agree on every step they share.

    Steps only one run trained (a shorter candidate, or one that went further
    than the reference) can't be checked, so they don't count either way.
    """
    statuses = [row["status"] for row in rows]
    return MATCH in statuses and MISMATCH not in statuses


def format_report(rows: list[dict[str, Any]]) -> str:
    """One line per step, then a summary."""
    lines = [
        f"step {row['step']:>4}  dp {row['reference_dp']} vs {row['candidate_dp']}  rows {row['rows']}  {row['status']}"
        for row in rows
    ]
    counts = {status: sum(row["status"] == status for row in rows) for status in STATUSES}
    shared = counts[MATCH] + counts[MISMATCH]
    unchecked = [f"{counts[status]} {status}" for status in (REFERENCE_ONLY, CANDIDATE_ONLY) if counts[status]]
    lines.append(
        f"{counts[MATCH]}/{shared} shared steps identical" + (f" ({', '.join(unchecked)})" if unchecked else "")
    )
    return "\n".join(lines)
