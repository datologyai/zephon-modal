# Copyright 2026 DatologyAI
# SPDX-License-Identifier: Apache-2.0

"""Elastic, deterministic Zephon training runs on Modal."""

from typing import TYPE_CHECKING

from zephon_modal.adapters import Adapter, Megatron, Torchtitan
from zephon_modal.context import LaunchContext, context
from zephon_modal.placement import Placement

if TYPE_CHECKING:
    from zephon_modal.job import ElasticJob


def __getattr__(name: str) -> object:
    # ElasticJob imports modal (~0.5 s), which trainer processes and
    # `zephon --help` (which imports every plugin) don't need.
    if name == "ElasticJob":
        from zephon_modal.job import ElasticJob

        return ElasticJob
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Adapter", "ElasticJob", "LaunchContext", "Megatron", "Placement", "Torchtitan", "context"]
