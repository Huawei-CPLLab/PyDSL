import numpy as np

from pydsl.frontend import compile
from pydsl.memref import DYNAMIC, MemRef, MemRefFactory
from pydsl.tensor import Tensor, TensorFactory
from pydsl.type import F32, F64, SInt32, SInt64, UInt32, UInt64
import pydsl.linalg as linalg
from helper import compilation_failed_from, multi_arange, run

TensorF64 = TensorFactory((DYNAMIC,), F64)
MemRefF64 = MemRefFactory((DYNAMIC,), F64)
TensorUI64 = TensorFactory((DYNAMIC,), UInt64)
MemRefUI64 = MemRefFactory((DYNAMIC,), UInt64)


def test_linalg_exp():
    @compile()
    def f(t1: TensorF64) -> TensorF64:
        return linalg.exp(t1)

    n1 = multi_arange((100,), np.float64) / 10 - 5
    assert np.allclose(f(n1.copy()), np.exp(n1))


def test_linalg_log():
    @compile()
    def f(t1: TensorF64) -> TensorF64:
        return linalg.log(t1)

    n1 = multi_arange((100,), np.float64) / 10 + 0.1
    assert np.allclose(f(n1.copy()), np.log(n1))


def test_linalg_abs():
    @compile()
    def f(t1: TensorF64) -> TensorF64:
        return linalg.abs(t1)

    n1 = multi_arange((100,), np.float64) / 10 - 5
    assert np.allclose(f(n1.copy()), np.abs(n1))


def test_linalg_ceil():
    @compile()
    def f(t1: TensorF64) -> TensorF64:
        return linalg.ceil(t1)

    n1 = multi_arange((100,), np.float64) / 10 - 5
    assert np.allclose(f(n1.copy()), np.ceil(n1))


def test_linalg_floor():
    @compile()
    def f(t1: TensorF64) -> TensorF64:
        return linalg.floor(t1)

    n1 = multi_arange((100,), np.float64) / 10 - 5
    assert np.allclose(f(n1.copy()), np.floor(n1))


def test_linalg_negf():
    @compile()
    def f(t1: TensorF64) -> TensorF64:
        return linalg.negf(t1)

    n1 = multi_arange((100,), np.float64) / 10 - 5
    assert np.allclose(f(n1.copy()), np.negative(n1))


def test_linalg_round():
    @compile()
    def f(t1: TensorF64) -> TensorF64:
        return linalg.round(t1)

    # MLIR and numpy don't round 0.5 the same way
    n1 = multi_arange((100,), np.float64) / 9 - 5
    assert np.allclose(f(n1.copy()), np.round(n1))


def test_linalg_sqrt():
    @compile()
    def f(t1: TensorF64) -> TensorF64:
        return linalg.sqrt(t1)

    n1 = multi_arange((100,), np.float64) / 10
    assert np.allclose(f(n1.copy()), np.sqrt(n1))


def test_linalg_rsqrt():
    @compile()
    def f(t1: TensorF64) -> TensorF64:
        return linalg.rsqrt(t1)

    n1 = multi_arange((100,), np.float64) / 10 + 0.1
    assert np.allclose(f(n1.copy()), np.reciprocal(np.sqrt(n1)))


def test_linalg_square():
    @compile()
    def f(t1: TensorF64) -> TensorF64:
        return linalg.square(t1)

    n1 = multi_arange((100,), np.float64) / 10 - 5
    assert np.allclose(f(n1.copy()), np.square(n1))


def test_linalg_tanh():
    @compile()
    def f(t1: TensorF64) -> TensorF64:
        return linalg.tanh(t1)

    n1 = multi_arange((100,), np.float64) / 10 - 5
    assert np.allclose(f(n1.copy()), np.tanh(n1))


# Numpy doesn't have erf. Scipy is needed.
# def test_linalg_erf():
#     @compile()
#     def f(t1: TensorF64) -> TensorF64:
#         return linalg.erf(t1)

#     n1 = multi_arange((100,), np.float64) / 10 - 5
#     assert np.allclose(f(n1.copy()), np.erf(n1))


def test_multiple_unary():
    @compile()
    def f(t1: TensorF64) -> TensorF64:
        t2 = linalg.exp(t1)
        t3 = linalg.sqrt(t1)
        return linalg.add(t2, t3, out=t2)

    n1 = multi_arange((50,), np.float64) / 10
    cor_res = np.exp(n1) + np.sqrt(n1)
    assert np.allclose(f(n1), cor_res)


# TODO: once we have proper function templates, can make these tests more
# general: test all possible cominations of {op, x, y, out} where op is an
# elemwise binary op, and we change type (Tensor vs MemRef) of each of
# x, y, out. Can also vary their element types.


def test_linalg_add():
    @compile()
    def f(
        m1: MemRef[UInt32, 10, 10],
        m2: MemRef[UInt32, 10, 10],
        m3: MemRef[UInt32, 10, 10],
    ) -> MemRef[UInt32, 10, 10]:
        return linalg.add(m1, m2, out=m3)

    n1 = multi_arange((10, 10), np.uint32)
    n2 = multi_arange((10, 10), np.uint32) + 1000
    n3 = multi_arange((10, 10), np.uint32) + 2000
    cor_res = n1 + n2
    res = f(n1, n2, n3)
    assert (res == cor_res).all()
    assert (n3 == cor_res).all()


def test_linalg_sub():
    @compile()
    def f(
        t1: Tensor[SInt32, 10, 10],
        t2: Tensor[SInt32, 10, 10],
        t3: Tensor[SInt64, 10, 10],
    ) -> Tensor[SInt64, 10, 10]:
        return linalg.sub(t1, t2, out=t3)

    n1 = multi_arange((10, 10), np.int32) - 12
    n2 = multi_arange((10, 10), np.int32) + 1000
    n3 = multi_arange((10, 10), np.int64) + 2000
    cor_res = n1.astype(np.int64) - n2.astype(np.int64)
    assert (f(n1, n2, n3) == cor_res).all()


def test_linalg_mul():
    @compile()
    def f(
        t1: MemRef[SInt32, 10, 10],
        t2: MemRef[F32, 10, 10],
        t3: MemRef[SInt64, 10, 10],
    ) -> MemRef[SInt64, 10, 10]:
        return linalg.mul(t1, t2, out=t3)

    n1 = multi_arange((10, 10), np.int32) - 12
    n2 = multi_arange((10, 10), np.float32) + 1000
    n3 = multi_arange((10, 10), np.int64) + 2000
    cor_res = n1.astype(np.int64) * n2.astype(np.int64)
    res = f(n1, n2, n3)
    assert (res == cor_res).all()
    assert (n3 == cor_res).all()


def test_linalg_div():
    @compile()
    def f(
        t1: Tensor[UInt32, 10, 10],
        t2: Tensor[UInt32, 10, 10],
        t3: Tensor[UInt32, 10, 10],
    ) -> Tensor[UInt32, 10, 10]:
        return linalg.div(t1, t2, out=t3)

    n1 = multi_arange((10, 10), np.uint32) - 12
    n2 = multi_arange((10, 10), np.uint32) + 1000
    n3 = multi_arange((10, 10), np.uint32) + 2000
    cor_res = n1 // n2
    assert (f(n1, n2, n3) == cor_res).all()


def test_linalg_max():
    @compile()
    def f(
        t1: MemRef[F32, 10, 10],
        t2: MemRef[F32, 10, 10],
        t3: MemRef[F64, 10, 10],
    ) -> MemRef[F64, 10, 10]:
        return linalg.max(t1, t2, out=t3)

    n1 = multi_arange((10, 10), np.float32) / 10 - 12
    n2 = multi_arange((10, 10), np.float32) / 5 - 30
    n3 = multi_arange((10, 10), np.float64) / 10 + 2000
    cor_res = np.maximum(n1.astype(np.float64), n2.astype(np.float64))
    res = f(n1, n2, n3)
    assert np.allclose(res, cor_res)
    assert np.allclose(n3, cor_res)


# Not sure if we want implicit uint -> sint casting to work in the future.
# Modify t2 to SInt32 if we decide to throw an error in that case.
def test_linalg_min():
    @compile()
    def f(
        t1: Tensor[SInt32, 10, 10],
        t2: Tensor[UInt32, 10, 10],
        t3: Tensor[SInt32, 10, 10],
    ) -> Tensor[SInt32, 10, 10]:
        return linalg.min(t1, t2, out=t3)

    n1 = multi_arange((10, 10), np.int32) - 12
    n2 = multi_arange((10, 10), np.uint32) * 2 - 30
    n3 = multi_arange((10, 10), np.int32) + 2000
    cor_res = np.minimum(n1.astype(np.int32), n2.astype(np.int32))
    assert (f(n1, n2, n3) == cor_res).all()


def test_linalg_powf():
    @compile()
    def f(
        t1: MemRef[F64, 10, 10],
        t2: MemRef[SInt32, 10, 10],
        t3: MemRef[F32, 10, 10],
    ) -> MemRef[F32, 10, 10]:
        return linalg.powf(t1, t2, out=t3)

    n1 = multi_arange((10, 10), np.float64) / 10 - 5
    n2 = multi_arange((10, 10), np.int32) - 50
    n3 = multi_arange((10, 10), np.float32) / 10 + 2000
    cor_res = np.power(n1.astype(np.float32), n2.astype(np.float32))
    res = f(n1, n2, n3)
    assert np.allclose(res, cor_res)
    assert np.allclose(n3, cor_res)


def test_elemwise_bin_mixed_tensor_memref():
    with compilation_failed_from(TypeError):
        # MLIR doesn't support mixing tensor and memref in this type of op
        @compile()
        def f(t1: Tensor[F64, 10], m2: MemRef[F64, 10]):
            linalg.add(t1, t1, m2)


def test_elemwise_bin_wrong_shape():
    with compilation_failed_from(TypeError):

        @compile()
        def f(t1: Tensor[F64, 5], t2: Tensor[F64, 8]):
            linalg.sub(t1, t1, t2)

    with compilation_failed_from(TypeError):

        @compile()
        def g(t1: Tensor[F64, 5], t2: Tensor[F64, 8]):
            linalg.sub(t1, t2, t1)


def test_linalg_fill():
    @compile()
    def fill_tensor(x: F32, t1: Tensor[F64, 10, 20]) -> Tensor[F64, 10, 20]:
        return linalg.fill(x, t1)

    @compile()
    def fill_memref(x: SInt32, m1: MemRef[SInt32, 100]) -> MemRef[SInt32, 100]:
        return linalg.fill(x, m1)

    # Check return value for tensor
    n1 = multi_arange((10, 20), np.float64)
    assert (fill_tensor(123, n1) == 123).all()

    # Check both return value and initial memory for memref
    n2 = multi_arange((100,), np.int32)
    res = fill_memref(456, n2)
    assert (res == 456).all()
    assert (n2 == 456).all()


if __name__ == "__main__":
    run(test_linalg_exp)
    run(test_linalg_log)
    run(test_linalg_abs)
    run(test_linalg_ceil)
    run(test_linalg_floor)
    run(test_linalg_negf)
    run(test_linalg_round)
    run(test_linalg_sqrt)
    run(test_linalg_rsqrt)
    run(test_linalg_square)
    run(test_linalg_tanh)
    run(test_multiple_unary)
    run(test_linalg_add)
    run(test_linalg_sub)
    run(test_linalg_mul)
    run(test_linalg_div)
    run(test_linalg_max)
    run(test_linalg_min)
    run(test_linalg_powf)
    run(test_elemwise_bin_mixed_tensor_memref)
    run(test_elemwise_bin_wrong_shape)
    run(test_linalg_fill)
