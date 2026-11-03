from zephon_modal.adapters import Megatron, Torchtitan
from zephon_modal.context import LaunchContext
from zephon_modal.placement import Placement
from zephon_modal.runner import NodeTopology, RunSpec, launch_context, torchrun_command


def _spec(placement: Placement, adapter=None) -> RunSpec:
    return RunSpec(
        run_id="r1",
        steps=20,
        seed=42,
        canonical_replicas=16,
        launch_id="20261102T000000",
        placement=placement,
        adapter=adapter or Torchtitan(module="recipes", config="smoke"),
        runs_mount="/runs",
        runs_volume_name="runs",
    )


def test_single_node_context_uses_tcpstore():
    ctx = launch_context(_spec(Placement("H100:4")), NodeTopology(1, 0, None))
    assert ctx.world_size == 4
    assert ctx.aggregate_store == "torch://"
    assert not ctx.keep_all_checkpoints


def test_multi_node_context_uses_tcpstore():
    ctx = launch_context(_spec(Placement("H100:8", nodes=2)), NodeTopology(2, 1, "10.0.0.1"))
    assert (ctx.world_size, ctx.node_rank) == (16, 1)
    assert ctx.aggregate_store == "torch://"
    assert ctx.keep_all_checkpoints


def test_context_round_trips_through_env():
    ctx = launch_context(_spec(Placement("H100:8")), NodeTopology(1, 0, None))
    assert LaunchContext.from_env(ctx.to_env()) == ctx


def test_torchtitan_command():
    spec = _spec(Placement("H100:8", nodes=2))
    topology = NodeTopology(2, 0, "10.0.0.1")
    command = torchrun_command(spec.adapter, launch_context(spec, topology), topology)
    assert command[:6] == [
        "torchrun",
        "--nproc_per_node=8",
        "--nnodes=2",
        "--node-rank=0",
        "--master-addr=10.0.0.1",
        "--master-port=29500",
    ]
    assert command[-5:] == ["-m", "torchtitan.train", "--module=recipes", "--config=smoke", "--output-dir=/runs/r1"]


def megatron_args(ctx: LaunchContext) -> list[str]:
    return [f"--train-iters={ctx.steps}", f"--zephon-aggregate-store={ctx.aggregate_store}"]


def test_megatron_command_resolves_args_in_container():
    spec = _spec(Placement("H100:4"), Megatron(args=f"{__name__}:megatron_args"))
    topology = NodeTopology(1, 0, None)
    command = torchrun_command(spec.adapter, launch_context(spec, topology), topology)
    assert "--local-ranks-filter=0,3" in command
    assert command[-5:] == [
        "-m",
        "zephon_modal.adapters.megatron_entry",
        "pretrain_gpt_zephon.py",
        "--train-iters=20",
        "--zephon-aggregate-store=torch://",
    ]
