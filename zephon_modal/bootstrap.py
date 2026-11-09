# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Agree on a resume snapshot before starting any trainer processes.

This short-lived TCP rendezvous deliberately does not use Volume files for
coordination. The leader publishes its cleanup before followers refresh their
mounts. Every follower acknowledges the same selected checkpoint before any
node proceeds. No torch import is required.
"""

from __future__ import annotations

import json
import socket
import time
from collections.abc import Callable
from contextlib import ExitStack
from typing import Any

BOOTSTRAP_PORT = 29501
BOOTSTRAP_TIMEOUT = 120.0


def _send(stream: Any, value: dict[str, Any]) -> None:
    stream.write(json.dumps(value).encode() + b"\n")
    stream.flush()


def _receive(stream: Any) -> dict[str, Any]:
    line = stream.readline(1024 * 1024)
    if not line.endswith(b"\n"):
        raise RuntimeError("Bootstrap peer disconnected or sent an oversized message")
    value = json.loads(line)
    if not isinstance(value, dict):
        raise RuntimeError("Invalid bootstrap message")
    return value


def coordinate(
    *,
    key: str,
    nodes: int,
    rank: int,
    master_addr: str | None,
    prepare: Callable[[], dict[str, Any]],
    refresh: Callable[[dict[str, Any]], None],
    port: int = BOOTSTRAP_PORT,
    timeout: float = BOOTSTRAP_TIMEOUT,
) -> dict[str, Any]:
    """Publish leader state, refresh followers, and await every acknowledgement.

    ``prepare`` must durably publish cleanup before returning. ``refresh`` must
    reload the follower's mount and validate the selected checkpoint, raising
    on disagreement. Connection failures fail the launch rather than permit
    unsynchronized restore.
    """
    if nodes == 1:
        return prepare()
    if not 0 <= rank < nodes or not master_addr:
        raise ValueError("A cluster bootstrap needs a valid rank and leader address")
    with ExitStack() as stack:
        if rank == 0:
            listener = stack.enter_context(socket.socket())
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("0.0.0.0", port))
            listener.listen(nodes - 1)
            listener.settimeout(timeout)
            peers = {}
            for _ in range(nodes - 1):
                connection, _ = listener.accept()
                stack.enter_context(connection)
                connection.settimeout(timeout)
                stream = stack.enter_context(connection.makefile("rwb"))
                hello = _receive(stream)
                peer_rank = hello.get("rank")
                if hello.get("key") != key or peer_rank not in range(1, nodes) or peer_rank in peers:
                    raise RuntimeError("Unexpected bootstrap peer")
                peers[peer_rank] = stream
            state = prepare()
            for stream in peers.values():
                _send(stream, state)
            for stream in peers.values():
                if _receive(stream) != {"ready": True}:
                    raise RuntimeError("A node could not validate the resume snapshot")
            for stream in peers.values():
                _send(stream, {"go": True})
            return state
        deadline = time.monotonic() + timeout
        while True:
            try:
                connection = socket.create_connection((master_addr, port), timeout=timeout)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Cluster bootstrap leader unavailable") from None
                time.sleep(0.1)
        stack.enter_context(connection)
        stream = stack.enter_context(connection.makefile("rwb"))
        _send(stream, {"key": key, "rank": rank})
        state = _receive(stream)
        refresh(state)
        _send(stream, {"ready": True})
        if _receive(stream) != {"go": True}:
            raise RuntimeError("Cluster bootstrap was not acknowledged")
        return state
