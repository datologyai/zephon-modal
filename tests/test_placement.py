import pytest

from zephon_modal.placement import Placement


def test_shape_and_name():
    p = Placement("H100:8")
    assert (p.gpu_type, p.gpus_per_node, p.world_size, p.name) == ("H100", 8, 8, "h100x8")
    assert Placement("H100:8", nodes=2).name == "2xh100x8"
    assert Placement("A10G").gpus_per_node == 1
    assert Placement("H100:4", nonpreemptible=True).name == "h100x4-np"


def test_multi_node_needs_whole_nodes():
    with pytest.raises(ValueError, match="whole nodes"):
        Placement("H100:4", nodes=2)


def test_dp_size_and_canonical_replicas():
    assert Placement("H100:8", nodes=2).dp_size() == 16
    assert Placement("H100:8").dp_size(model_parallel_size=4) == 2
    with pytest.raises(ValueError, match="model_parallel_size"):
        Placement("H100:4").dp_size(model_parallel_size=8)
    Placement("H100:4").check_canonical_replicas(16)
    with pytest.raises(ValueError, match="doesn't divide canonical_replicas"):
        Placement("H100:8").check_canonical_replicas(12)
