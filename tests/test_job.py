import dataclasses
from types import SimpleNamespace

import modal
import pytest

from zephon_modal import ElasticJob, Placement, Torchtitan

PLACEMENTS = [Placement("H100:8", nodes=2), Placement("H100:8"), Placement("H100:4")]


def _job(**overrides) -> ElasticJob:
    kwargs = dict(
        name="test-job",
        image=modal.Image.debian_slim(),
        adapter=Torchtitan(module="recipes", config="smoke"),
        canonical_replicas=16,
        placements=PLACEMENTS,
        runs_volume=modal.Volume.from_name("test-runs", version=2),
    )
    return ElasticJob(**(kwargs | overrides))


def test_registers_a_function_per_placement():
    job = _job()
    assert list(job.functions) == ["2xh100x8", "h100x8", "h100x4"]
    assert job.placement().name == "2xh100x8"
    assert job.placement("h100x4") is PLACEMENTS[2]


def test_rejects_placements_the_canonical_replicas_dont_allow():
    with pytest.raises(ValueError, match="doesn't divide canonical_replicas"):
        _job(canonical_replicas=12)


def test_rejects_duplicate_placements():
    with pytest.raises(ValueError, match="unique"):
        _job(placements=[Placement("H100:8"), Placement("H100:8")])


def test_run_spec_carries_the_adapter():
    job = _job(compile_cache_volume=modal.Volume.from_name("cache", version=2))
    spec = job.run_spec("r1", 30, job.placement("h100x8"))
    assert spec.run_dir == "/runs/r1"
    assert spec.adapter == job.adapter
    assert spec.compile_cache_mount == "/compile-cache"
    assert (spec.runs_volume_name, spec.compile_cache_volume_name) == ("test-runs", "cache")
    assert spec.canonical_replicas == 16


def test_job_is_frozen():
    job = _job()
    with pytest.raises(dataclasses.FrozenInstanceError):
        job.canonical_replicas = 8
    assert isinstance(job.placements, tuple)


def test_rejects_unnamed_runs_volume():
    # An ephemeral Volume has no name; building one needs a Modal connection.
    with pytest.raises(ValueError, match="named Volumes"):
        _job(runs_volume=SimpleNamespace(name=None))
