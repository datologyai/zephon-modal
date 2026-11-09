import pytest

from zephon_modal.placement import Placement


def test_shape_and_name():
    p = Placement("H100:8")
    assert (p.gpu_type, p.gpus_per_node, p.world_size, p.name) == ("H100", 8, 8, "h100x8")
    assert Placement("H100:8", nodes=2).name == "2xh100x8"
    assert Placement("A10G").gpus_per_node == 1


def test_multi_node_needs_whole_nodes():
    with pytest.raises(ValueError, match="whole nodes"):
        Placement("H100:4", nodes=2)


def test_dp_size():
    assert Placement("H100:8", nodes=2).dp_size() == 16
    assert Placement("H100:8").dp_size(model_parallel_size=4) == 2
    with pytest.raises(ValueError, match="model_parallel_size"):
        Placement("H100:4").dp_size(model_parallel_size=8)


def test_fallbacks_are_one_modal_function_with_a_gpu_list():
    p = Placement("H100:8", fallbacks=["H100:4"])
    assert p.name == "h100x8-or-h100x4"
    assert p.modal_gpu == ["H100:8", "H100:4"]
    assert Placement("H100:8").modal_gpu == "H100:8"
    assert [shape.name for shape in p.shapes] == ["h100x8", "h100x4"]
    assert p.allocated(4) == Placement("H100:4")
    with pytest.raises(RuntimeError, match="got 2 GPUs"):
        p.allocated(2)


def test_fallbacks_must_be_single_node_with_distinct_counts():
    with pytest.raises(ValueError, match="clusters"):
        Placement("H100:8", nodes=2, fallbacks=("H100:4",))
    with pytest.raises(ValueError, match="distinct GPU counts"):
        Placement("H100:8", fallbacks=("A100-80GB:8",))
