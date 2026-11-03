# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""What zephon_modal needs from a training framework."""

from __future__ import annotations

import importlib
import signal
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from zephon_modal.checkpoints import CheckpointLayout
from zephon_modal.context import LaunchContext


@dataclass(frozen=True, kw_only=True)
class Adapter:
    """How to run one framework's trainer under torchrun.

    Adapters are plain data: they're pickled into each launch and their
    methods run inside the training container, so anything recipe-specific
    is named by an import string rather than passed as an object.

    The framework must resume from the latest checkpoint under
    ``ctx.run_dir`` by itself, store Zephon's state with its checkpoints, and
    take its DP degree from the world size.

    When Modal stops a container (preemption, ``modal container stop``),
    the runner sends torchrun ``shutdown_signal``, which it forwards to every
    rank, then waits up to ``shutdown_timeout`` seconds before killing the
    trainer and committing the Volume. Modal allows 30 seconds in all, and the
    commit needs the rest: it uploads whatever the trainer wrote since the
    last background commit, such as a half-written checkpoint.

    Attributes:
        model_parallel_size: TP x PP x CP. DP is the world size divided by
            this, and the job's data contract decides which DP degrees work.
        checkpoints: Where the trainer's DCP checkpoints live, so incomplete
            ones can be set aside before a resume. ``None`` skips that.
        shutdown_signal: What the trainer gets when the container is stopped.
        shutdown_timeout: Seconds the trainer gets to exit (or save) before it's killed.
    """

    model_parallel_size: int = 1
    checkpoints: CheckpointLayout | None = None
    shutdown_signal: signal.Signals = signal.SIGTERM
    shutdown_timeout: float = 10.0

    def workdir(self) -> str:
        """Directory torchrun runs in."""
        raise NotImplementedError

    def command(self, ctx: LaunchContext) -> list[str]:
        """Arguments after torchrun's own, e.g. ``["-m", "torchtitan.train", ...]``."""
        raise NotImplementedError

    def env(self, ctx: LaunchContext) -> dict[str, str]:
        """Extra environment variables for the trainer processes."""
        return {}

    def local_ranks_filter(self, nproc_per_node: int) -> str:
        """Local ranks whose output torchrun shows."""
        return "0"


def resolve(target: str) -> Callable[..., Any]:
    """Import ``"package.module:attribute"``."""
    module_name, sep, attribute = target.partition(":")
    if not sep or not attribute:
        raise ValueError(f"Expected 'module:attribute', got {target!r}")
    return getattr(importlib.import_module(module_name), attribute)
