import gc
import numpy as np
from numpy.lib.stride_tricks import as_strided
import weakref

from pydsl.frontend import compile
import pydsl.linalg as linalg
from pydsl.memref import DYNAMIC
import pydsl.tensor as tensor
from pydsl.tensor import Tensor, TensorFactory
from pydsl.type import F32, F64, Index, SInt32, Tuple, UInt64
from helper import failed_from, compilation_failed_from, multi_arange, run

TensorF32_2 = Tensor[F32, DYNAMIC, DYNAMIC]
TensorF32_3 = Tensor[F32, DYNAMIC, DYNAMIC, DYNAMIC]
TensorI32_2 = Tensor[SInt32, DYNAMIC, DYNAMIC]
TensorI32_3 = Tensor[SInt32, DYNAMIC, DYNAMIC, DYNAMIC]
TensorF64_1 = Tensor[F64, DYNAMIC]
TensorF64_2 = TensorFactory((DYNAMIC, DYNAMIC), F64)
TensorF64_4 = TensorFactory((DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC), F64)
TensorU64_2 = TensorFactory((DYNAMIC, DYNAMIC), UInt64)

# Important: PyDSL functions are allowed to modify tensors passed as arguments
# in-place. Thus, we should always make a copy to compare to or pass in a copy.


def test_wrong_dim():
    @compile()
    def f(t1: Tensor[F64, 8, 5, 6]):
        pass

    with failed_from(TypeError):
        n1 = np.empty([8, 6, 6])
        f(n1)

    with failed_from(TypeError):
        n1 = np.empty([8, 5, 6, 1])
        f(n1)


def test_load():
    @compile()
    def f(t1: Tensor[SInt32, 2, 3]) -> SInt32:
        return t1[1, 2] + t1[0, 0]

    n1 = np.asarray([[1, 2, 4], [8, 16, 32]], dtype=np.int32)
    assert f(n1.copy()) == n1[1, 2] + n1[0, 0]


def test_store():
    @compile()
    def f(
        t1: Tensor[UInt64, 3, 2], a: UInt64, b: UInt64
    ) -> Tensor[UInt64, 3, 2]:
        t1[1, 0] = a
        t1[2, 1] = b
        return t1

    n1 = np.asarray([[1, 2], [3, 4], [5, 6]], dtype=np.uint64)
    cor_res = n1.copy()
    cor_res[1, 0] = 7
    cor_res[2, 1] = 8
    assert (f(n1, 7, 8) == cor_res).all()


def test_load_slice_1d():
    @compile()
    def f(t1: TensorF64_1) -> TensorF64_1:
        return t1[2:9:3]

    n1 = multi_arange((10,), np.float64)
    assert (f(n1.copy()) == n1[2:9:3]).all()


def test_load_slice_3d():
    @compile()
    def f(t1: Tensor[SInt32, 10, 6, 8]) -> TensorI32_3:
        return t1[1:7:3, 0:5:1, 1:6:2]

    n1 = multi_arange((10, 6, 8), np.int32)
    assert (f(n1.copy()) == n1[1:7:3, 0:5:1, 1:6:2]).all()


def test_load_slice_implicit():
    @compile()
    def f(t1: Tensor[F64, 5, 6, 7, 8]) -> TensorF64_4:
        return t1[::, 2::2, 3]

    n1 = multi_arange((5, 6, 7, 8), np.float64)
    assert (f(n1.copy()).squeeze() == n1[::, 2::2, 3]).all()


def test_load_slice_extra_dims():
    with compilation_failed_from(IndexError):
        # More indices than rank of tensor is not allowed
        @compile()
        def f(t1: Tensor[UInt64, 10, 4]) -> TensorU64_2:
            return t1[3::, 1:4, ::2]


def test_load_slice_exp():
    @compile()
    def f(t1: Tensor[F64, 10, 10]) -> TensorF64_2:
        return linalg.exp(t1[2:7:2, 3:])

    n1 = multi_arange((10, 10), np.float64)
    assert np.allclose(f(n1.copy()), np.exp(n1[2:7:2, 3:]))


def test_load_compose_strided():
    @compile()
    def f(t1: Tensor[F32, 16, 12]) -> TensorF32_2:
        t2 = t1[1:15:2, ::3]
        return t2[1::3, 1::2]

    n1 = multi_arange((16, 12), np.float32)
    n1_sub = n1[1:15:2, ::3][1::3, 1::2]
    assert (f(n1.copy()) == n1_sub).all()


def test_store_slice_1d_multi():
    @compile()
    def f(
        t1: Tensor[SInt32, 8], t2: Tensor[SInt32, 2], t3: Tensor[SInt32, 3]
    ) -> Tensor[SInt32, 8]:
        t1[3:5] = t2
        t1[1:7:2] = t3
        return t1

    n1 = multi_arange((8,), np.int32)
    n2 = multi_arange((2,), np.int32) + 10
    n3 = multi_arange((3,), np.int32) + 20
    cor_res = n1.copy()
    cor_res[3:5] = n2
    cor_res[1:7:2] = n3
    assert (f(n1, n2, n3) == cor_res).all()


def test_store_slice_3d():
    @compile()
    def f(
        t1: Tensor[F32, 8, 10, 12], t2: Tensor[F32, 3, 4, 5]
    ) -> Tensor[F32, 8, 10, 12]:
        t1[5:, ::3, 3:12:2] = t2
        return t1

    n1 = multi_arange((8, 10, 12), np.float32)
    n2 = multi_arange((3, 4, 5), np.float32) + 1000
    cor_res = n1.copy()
    cor_res[5:, ::3, 3:12:2] = n2
    assert (f(n1, n2) == cor_res).all()


def test_store_slice_dynamic():
    @compile()
    def f(
        t1: Tensor[F64, DYNAMIC, 5, DYNAMIC],
        t2: Tensor[F64, DYNAMIC, DYNAMIC, 4],
        yLo: Index,
        yHi: Index,
        yStep: Index,
    ) -> Tensor[F64, DYNAMIC, 5, DYNAMIC]:
        t1[2:7, yLo:yHi:yStep, :] = t2
        return t1

    n1 = multi_arange((10, 5, 4), np.float64)
    n2 = multi_arange((5, 3, 4), np.float64) + 1000
    cor_res = n1.copy()
    cor_res[2:7, 1:4:1, :] = n2
    assert (f(n1, n2, 1, 4, 1) == cor_res).all()


def test_store_slice_wrong_rank():
    with compilation_failed_from(TypeError):
        # t1 has rank 3 but t2 has rank 2
        @compile()
        def f(
            t1: Tensor[UInt64, 10, 4, 7], t2: Tensor[UInt64, 6, 2]
        ) -> Tensor[UInt64, 10, 4, 7]:
            t1[0:6, 0:2] = t2
            return t1


def test_load_store_slice_self():
    @compile()
    def f(t1: Tensor[F32, 10, 10]) -> Tensor[F32, 10, 10]:
        t1[2:9:3, 3:6] = t1[4:7, 1:7:2]
        return t1

    n1 = multi_arange((10, 10), np.float32)
    cor_res = n1.copy()
    cor_res[2:9:3, 3:6] = cor_res[4:7, 1:7:2]
    assert (f(n1) == cor_res).all()


def test_store_slice_exp():
    @compile()
    def f(
        t1: Tensor[F64, 10, 10], t2: Tensor[F64, 7, 5]
    ) -> Tensor[F64, 10, 10]:
        t1[:7, 5:] = linalg.exp(t2)
        return t1

    n1 = multi_arange((10, 10), np.float64) / 50
    n2 = multi_arange((7, 5), np.float64) / 49
    cor_res = n1.copy()
    cor_res[:7, 5:] = np.exp(n2)
    assert np.allclose(f(n1, n2), cor_res)


def test_strided_ndarray_to_tensor():
    # Arbitrary strides should work since tensor gets lowered to
    # memref<..., strided<[?, ?, ?], offset: ?>>

    @compile()
    def f(t1: Tensor[SInt32, 12, DYNAMIC]) -> TensorI32_2:
        t1[1, 2] = -1000
        t1_sub = t1[1::2, 1:5]
        t1_sub[3, 2] = -1001
        return t1_sub

    i32_sz = np.int32().nbytes
    n1 = multi_arange((500,), np.int32)
    n1 = as_strided(n1, shape=(12, 6), strides=(11 * i32_sz, 2 * i32_sz))

    cor_res = n1.copy()
    cor_res[1, 2] = -1000
    cor_res = cor_res[1::2, 1:5]
    cor_res[3, 2] = -1001
    assert (f(n1) == cor_res).all()


def test_arg_copy():
    """
    Test whether we make a copy of a tensor that's passed as an argument or
    whether we end up writing to it in-place. Since we removed
    -copy-before-write from -one-shot-bufferize, we would like mlir-opt to
    optimize the below code to modify the tensor in-place.
    """

    @compile()
    def f(t1: Tensor[SInt32, 10]) -> Tensor[SInt32, 10]:
        t1[4] = 100
        t1[8] = 101
        t1[2:5] = t1[4:7]
        return t1

    n1 = multi_arange((10,), np.int32)
    n1_old = n1.copy()
    test_res = f(n1)

    # Check if n1 was also updated
    assert not (n1 == n1_old).all()
    assert (test_res == n1).all()


def test_link_ndarray():
    """
    Relatively simple check to make sure that a returned Tensor has a pointer
    to an input tensor if they overlap. test_memref has a more comprehensive
    test.
    """

    def get_root(arr: np.ndarray):
        while arr.base is not None:
            arr = arr.base
        return arr

    @compile()
    def f(t1: Tensor[F64, 64, 64]) -> TensorF64_2:
        t1[5, 10] = 12345
        return t1[5:40:7, 8::2]

    n1 = multi_arange((64, 64), np.float64)
    cor_res = n1.copy()
    cor_res[5, 10] = 12345
    cor_res = cor_res[5:40:7, 8::2]
    res1 = f(n1)
    assert (res1 == cor_res).all()

    n1_root_ref = weakref.ref(get_root(n1))
    n1 = None
    gc.collect()
    assert n1_root_ref() is not None

    res1 = None
    gc.collect()
    assert n1_root_ref() is None


def test_zero_d():
    @compile()
    def f(t1: Tensor[SInt32]) -> Tuple[SInt32, Tensor[SInt32]]:
        x = t1[()]
        t1[()] = 456
        return x, t1

    n1 = np.array(123, dtype=np.int32)
    res1, res2 = f(n1)
    assert res1 == 123 and res2 == 456
    assert isinstance(res2, np.ndarray)
    assert res2.shape == ()


def test_empty():
    @compile()
    def f(ysz: Index) -> Tensor[SInt32, 10, DYNAMIC]:
        t1 = tensor.empty((10, ysz), SInt32)
        t1[1, 2] = 100
        t1[3, 6] = 101
        t1[9, 7] = 102
        return t1

    res = f(8)
    assert res[1, 2] == 100
    assert res[3, 6] == 101
    assert res[9, 7] == 102


def test_full():
    @compile()
    def f(xsz: Index) -> Tensor[UInt64, DYNAMIC, 5]:
        t1 = tensor.full((xsz, 5), 123, UInt64)
        return t1

    test_res = f(3)
    cor_res = np.full((3, 5), 123, dtype=np.uint64)
    assert (test_res == cor_res).all()


def test_zeros():
    @compile()
    def f() -> Tensor[F32, 10, 10]:
        return tensor.zeros((10, 10), dtype=F32)

    test_res = f()
    cor_res = np.zeros((10, 10), dtype=np.float32)
    assert (test_res == cor_res).all()


def test_ones():
    @compile()
    def f() -> Tensor[F64, DYNAMIC, DYNAMIC]:
        return tensor.ones((Index(6), Index(4)), dtype=F64)

    test_res = f()
    cor_res = np.ones((6, 4), dtype=np.float32)
    assert (test_res == cor_res).all()


if __name__ == "__main__":
    run(test_wrong_dim)
    run(test_load)
    run(test_store)
    run(test_load_slice_1d)
    run(test_load_slice_3d)
    run(test_load_slice_implicit)
    run(test_load_slice_extra_dims)
    run(test_load_slice_exp)
    run(test_load_compose_strided)
    run(test_store_slice_1d_multi)
    run(test_store_slice_3d)
    run(test_store_slice_dynamic)
    run(test_store_slice_wrong_rank)
    run(test_load_store_slice_self)
    run(test_store_slice_exp)
    run(test_strided_ndarray_to_tensor)
    run(test_arg_copy)
    run(test_link_ndarray)
    run(test_zero_d)
    run(test_empty)
    run(test_full)
    run(test_zeros)
    run(test_ones)
