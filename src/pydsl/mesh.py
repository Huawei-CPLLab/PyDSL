from mlir.dialects import mesh

from pydsl.macro import CallMacro, Compiled, Evaluated
from pydsl.type import lower_single
from pydsl.protocols import SubtreeOut, ToMLIRBase


@CallMacro.generate()
def create_mesh(
    visitor: ToMLIRBase,
    mesh_name: Evaluated,
    shape: Evaluated,
) -> SubtreeOut:
    assert isinstance(mesh_name, str)
    return mesh.MeshOp(mesh_name, shape)


@CallMacro.generate()
def shard(
    visitor: ToMLIRBase,
    t_val: Compiled,
    mesh_name: Evaluated,
    split_axes: Evaluated,
    dynamic_sharded_dims_offsets=[],
    dynamic_halo_sizes=[],
) -> SubtreeOut:
    meshAxes = mesh.MeshAxesArrayAttr.get(split_axes)
    sharding = mesh.sharding(
        mesh_name, meshAxes, dynamic_sharded_dims_offsets, dynamic_halo_sizes
    )
    return type(t_val)(mesh.shard(lower_single(t_val), sharding))


@CallMacro.generate()
def all_gather(
    visitor: ToMLIRBase,
    output: Compiled,
    mesh_name: Evaluated,
    input: Compiled,
    gather_axis: Evaluated,
    mesh_axes: Evaluated = None,
) -> SubtreeOut:
    meshAxes = None
    if mesh_axes is not None:
        meshAxes = mesh.MeshAxesAttr.get(mesh_axes)

    return type(output)(
        mesh.all_gather(
            lower_single(type(output)),
            mesh_name,
            lower_single(input),
            gather_axis,
            mesh_axes=meshAxes,
        )
    )


def verify_reduction_kind(reduction_kind: type[str]):
    reduction_kinds = [
        "sum",
        "max",
        "min",
        "product",
        "avergaebitwise_and",
        "bitwise_or",
        "bitwise_xor",
        "generic",
    ]

    if reduction_kind not in reduction_kinds:
        raise ValueError(f"reduction kind {reduction_kind} does not exist")


@CallMacro.generate()
def all_reduce(
    visitor: ToMLIRBase,
    output: Compiled,
    mesh_name: Evaluated,
    input: Compiled,
    mesh_axes: Evaluated = None,
    reduction_kind: Evaluated = "sum",
) -> SubtreeOut:
    meshAxes = None
    if mesh_axes is not None:
        meshAxes = mesh.MeshAxesAttr.get(mesh_axes)

    verify_reduction_kind(reduction_kind)

    return type(output)(
        mesh.all_reduce(
            lower_single(type(output)),
            mesh_name,
            lower_single(input),
            mesh_axes=meshAxes,
            reduction=reduction_kind,
        )
    )
