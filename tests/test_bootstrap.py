# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

import socket
from concurrent.futures import ThreadPoolExecutor

import pytest

from zephon_modal.bootstrap import coordinate


def _cluster(follower_refresh):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    events = []

    def prepare():
        events.append("cleanup committed")
        return {"attempt_tag": ".a000002-unique", "resume_from": 10}

    def refresh(state):
        assert events == ["cleanup committed"]
        follower_refresh(state)
        events.append("follower reloaded")

    def run(rank):
        return coordinate(
            key="launch",
            nodes=2,
            rank=rank,
            master_addr="127.0.0.1",
            prepare=prepare,
            refresh=refresh,
            port=port,
            timeout=2,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        leader = pool.submit(run, 0)
        follower = pool.submit(run, 1)
        return leader, follower, events


def test_cluster_agrees_after_cleanup_is_committed_and_follower_reloads():
    leader, follower, events = _cluster(lambda state: None)
    assert leader.result() == follower.result() == {"attempt_tag": ".a000002-unique", "resume_from": 10}
    assert events == ["cleanup committed", "follower reloaded"]


def test_disagreement_prevents_both_nodes_from_starting():
    def disagree(state):
        raise RuntimeError("different checkpoint")

    leader, follower, _ = _cluster(disagree)
    with pytest.raises(RuntimeError, match="different checkpoint"):
        follower.result()
    with pytest.raises(RuntimeError, match="disconnected"):
        leader.result()
