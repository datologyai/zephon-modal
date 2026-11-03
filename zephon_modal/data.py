# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""What a job's data loader requires of placements, and tells its trainers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DataContract:
    """The data loader's side of a job, as seen by the launcher.

    The base contract accepts any data-parallel degree and adds nothing to the
    launch context: enough for a workload that can resume on any GPU count
    and doesn't need identical data across resizes. Integrations such as
    :class:`zephon_modal.zephon.Zephon` subclass it.

    Contracts are plain data, pickled into each launch like adapters.
    """

    def check_dp_size(self, dp: int) -> None:
        """Raise ``ValueError`` if the loader can't train or resume at this DP degree."""

    def context(self) -> dict[str, Any]:
        """JSON-serializable settings every trainer process sees as ``LaunchContext.data``."""
        return {}

    def describe_dp_size(self, dp: int) -> str:
        """A short note on what this DP degree means for the loader, for ``placements``."""
        return ""
