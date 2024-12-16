from pydsl import mesh, linalg

from pydsl.frontend import compile, create_MLIRTarget
from pydsl.type import F16
from pydsl.tensor import TensorFactory


mesh_0 = "mesh0"
mesh_1 = "mesh1"
split_axes_0 = [[0]]
split_axes_1 = [[0], [1]]
tensor_x = TensorFactory((16, 65536, 65536), F16)
tensor_w0 = TensorFactory((16, 65536, 8192), F16)
tensor_b0 = TensorFactory((16, 65536, 8192), F16)
passes = [
    '--pass-pipeline="builtin.module(func.func(sharding-propagation,mesh-spmdization,cse))"'
]
MeshTarget = create_MLIRTarget(passes)


@compile(target_class=MeshTarget, dump_mlir=True)
class CallTest:
    def __init__():
        mesh.create_mesh(mesh_0, [8])
        mesh.create_mesh(mesh_1, [2, 4])

    def mlp_1d(x: tensor_x, w_ff0: tensor_w0, b_ff0: tensor_b0) -> tensor_b0:
        sharded_x = mesh.shard(x, mesh_0, split_axes_0)
        mm0 = linalg.batch_matmul(sharded_x, w_ff0)
        ff0 = linalg.add(mm0, b_ff0)
        return ff0

    def mlp_2d(x: tensor_x, w_ff0: tensor_w0, b_ff0: tensor_b0) -> tensor_b0:
        sharded_x = mesh.shard(x, mesh_1, split_axes_1)
        mm0 = linalg.batch_matmul(sharded_x, w_ff0)
        ff0 = linalg.add(mm0, b_ff0)
        return ff0
