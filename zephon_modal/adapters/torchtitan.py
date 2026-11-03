# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""torchtitan (torchtitan-zephon), launched as ``torchtitan.train``."""

from __future__ import annotations

from dataclasses import dataclass

from zephon_modal.adapters.base import Adapter
from zephon_modal.context import LaunchContext


@dataclass(frozen=True, kw_only=True)
class Torchtitan(Adapter):
    """Runs ``torchtitan.train --module <module> --config <config>``.

    The recipe module must be importable in the training image. It reads the
    launch from :func:`zephon_modal.context`, and can log sample hashes with
    :class:`zephon_modal.zephon.torchtitan.SampleHashingZephonDataLoader`.

    Attributes:
        module: Recipe module, passed as ``--module``.
        config: Recipe function in that module, passed as ``--config``.
        torchtitan_dir: torchtitan-zephon checkout in the image.
    """

    module: str
    config: str
    torchtitan_dir: str = "/opt/torchtitan"

    def workdir(self) -> str:
        return self.torchtitan_dir

    def command(self, ctx: LaunchContext) -> list[str]:
        return [
            "-m",
            "torchtitan.train",
            f"--module={self.module}",
            f"--config={self.config}",
            f"--output-dir={ctx.run_dir}",
        ]
