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
INCOMPLETE = "incomplete evidence"
STATUSES = (MATCH, MISMATCH, REFERENCE_ONLY, CANDIDATE_ONLY, INCOMPLETE)


def _read(sample_hash_dir: str) -> dict[int, dict[str, dict[str, Any]]]:
    """``{step: {attempt: {"dp", "records": {file: record}}}}`` for one run.

    An attempt is a hash file's launch-ID prefix (launch ID plus attempt tag).
    """
    per_step: dict[int, dict[str, dict[str, Any]]] = {}
    if not os.path.isdir(sample_hash_dir):
        return {}
    for name in sorted(os.listdir(sample_hash_dir)):
        if not name.endswith(".jsonl"):
            continue
        attempt = name.split("-dp")[0]
        with open(os.path.join(sample_hash_dir, name)) as hash_file:
            for line in hash_file:
                record = json.loads(line)
                entry = per_step.setdefault(record["step"], {}).setdefault(attempt, {"dp": record["dp"], "records": {}})
                entry["records"][name] = record
    return per_step


def _digest(attempt: str, entry: dict[str, Any]) -> dict[str, Any]:
    """One attempt's evidence for one step: its digest, and whether it's complete."""
    # With tensor or context parallelism, several global ranks share a DP
    # rank and record the same rows: count each DP rank once, and only if
    # its copies agree.
    by_dp_rank: dict[int, list[dict[str, Any]]] = {}
    for record in entry["records"].values():
        by_dp_rank.setdefault(record["dp_rank"], []).append(record)
    peers_agree = all(
        all(copy["row_hashes"] == copies[0]["row_hashes"] for copy in copies) for copies in by_dp_rank.values()
    )
    records = [copies[0] for _, copies in sorted(by_dp_rank.items())]
    all_records = [copy for copies in by_dp_rank.values() for copy in copies]
    # Records from before schemas existed hashed tokens only, so they're never comparable.
    schemas = {record.get("schema") for record in all_records}
    rows = [row for record in records for row in record["row_hashes"]]
    complete = (
        entry["dp"] > 0
        and peers_agree
        and set(by_dp_rank) == set(range(entry["dp"]))
        and all(record["dp"] == entry["dp"] and record["row_hashes"] for record in all_records)
        and len(schemas) == 1
        and None not in schemas
    )
    return {
        "digest": hashlib.sha256("".join(sorted(rows)).encode()).hexdigest(),
        "rows": len(rows),
        "dp": entry["dp"],
        "complete": complete,
        "attempt": attempt,
        "schema": records[0].get("schema"),
    }


def load_step_digests(sample_hash_dir: str) -> dict[int, dict[str, Any]]:
    """Fold one run's hash files into ``{step: {digest, rows, dp, complete, ...}}``.

    A step re-trained after a resume appears in several attempts; launch IDs
    sort chronologically and attempt tags are numbered, so the latest
    attempt's records are the ones that count.
    """
    return {
        step: _digest(max(attempts), attempts[max(attempts)])
        for step, attempts in sorted(_read(sample_hash_dir).items())
    }


def check_replays(run_dir: str) -> list[dict[str, Any]]:
    """Check one run against itself: every step trained by more than one attempt.

    Each resume re-trains the steps since the last checkpoint, often at a
    different DP degree, so those steps get hashed by both attempts and must
    match. A long run gets this evidence at every resize without a reference
    run. Only complete evidence counts: an attempt cut off mid-step leaves a
    partial record, which is expected and skipped.
    """
    rows = []
    for step, attempts in sorted(_read(os.path.join(run_dir, SAMPLE_HASH_SUBDIR)).items()):
        complete = [digest for digest in (_digest(a, e) for a, e in sorted(attempts.items())) if digest["complete"]]
        if len(complete) < 2:
            continue
        same = len({(digest["digest"], digest["schema"]) for digest in complete}) == 1
        rows.append(
            {
                "step": step,
                "attempts": [digest["attempt"] for digest in complete],
                "dps": [digest["dp"] for digest in complete],
                "rows": complete[-1]["rows"],
                "status": MATCH if same else MISMATCH,
            }
        )
    return rows


def format_replays(rows: list[dict[str, Any]]) -> str:
    """One line per replayed step, then a summary."""
    lines = [
        f"step {row['step']:>6}  dp {' / '.join(map(str, row['dps']))}  attempts {len(row['attempts'])}  {row['status']}"
        for row in rows
    ]
    matched = sum(row["status"] == MATCH for row in rows)
    lines.append(f"{matched}/{len(rows)} replayed steps identical" if rows else "No replayed steps yet")
    return "\n".join(lines)


def compare_runs(
    reference_dir: str,
    candidate_dir: str,
    *,
    steps: int | None = None,
    start_step: int = 1,
    rows_per_step: int | None = None,
) -> list[dict[str, Any]]:
    """Compare two run directories step by step."""
    ref = load_step_digests(os.path.join(reference_dir, SAMPLE_HASH_SUBDIR))
    cand = load_step_digests(os.path.join(candidate_dir, SAMPLE_HASH_SUBDIR))
    if steps is not None and (start_step < 1 or steps < start_step):
        raise ValueError("Expected a nonempty positive step range")
    if rows_per_step is not None and rows_per_step < 1:
        raise ValueError("rows_per_step must be positive")
    step_numbers = range(start_step, steps + 1) if steps is not None else sorted(ref.keys() | cand.keys())
    return [
        {
            "step": step,
            "reference_dp": ref.get(step, {}).get("dp"),
            "candidate_dp": cand.get(step, {}).get("dp"),
            "rows": cand.get(step, ref.get(step, {})).get("rows"),
            "status": _status(ref.get(step), cand.get(step), rows_per_step),
            "reference_rows": ref.get(step, {}).get("rows"),
            "candidate_rows": cand.get(step, {}).get("rows"),
        }
        for step in step_numbers
    ]


def _status(ref: dict[str, Any] | None, cand: dict[str, Any] | None, expected_rows: int | None) -> str:
    if ref is None and cand is None:
        return INCOMPLETE
    if ref is None:
        return CANDIDATE_ONLY
    if cand is None:
        return REFERENCE_ONLY
    if not ref["complete"] or not cand["complete"] or ref["schema"] != cand["schema"]:
        return INCOMPLETE
    if expected_rows is not None and (ref["rows"] != expected_rows or cand["rows"] != expected_rows):
        return INCOMPLETE
    return MATCH if ref["digest"] == cand["digest"] else MISMATCH


def identical(rows: list[dict[str, Any]], *, strict: bool = False) -> bool:
    """Whether the runs share at least one step and agree on every step they share.

    Steps only one run trained (a shorter candidate, or one that went further
    than the reference) can't be checked, so they don't count either way.
    """
    statuses = [row["status"] for row in rows]
    if strict:
        return bool(statuses) and all(status == MATCH for status in statuses)
    return MATCH in statuses and MISMATCH not in statuses and INCOMPLETE not in statuses


def format_report(rows: list[dict[str, Any]]) -> str:
    """One line per step, then a summary."""
    lines = [
        f"step {row['step']:>4}  dp {row['reference_dp']} vs {row['candidate_dp']}  rows {row['rows']}  {row['status']}"
        for row in rows
    ]
    counts = {status: sum(row["status"] == status for row in rows) for status in STATUSES}
    shared = counts[MATCH] + counts[MISMATCH]
    unchecked = [
        f"{counts[status]} {status}" for status in (REFERENCE_ONLY, CANDIDATE_ONLY, INCOMPLETE) if counts[status]
    ]
    lines.append(
        f"{counts[MATCH]}/{shared} shared steps identical" + (f" ({', '.join(unchecked)})" if unchecked else "")
    )
    return "\n".join(lines)
