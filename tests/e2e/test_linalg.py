import numpy as np
import typing

from collections.abc import Iterable
import pydsl.arith as arith
from pydsl.frontend import compile, template
from pydsl.func import InlineFunction
from pydsl.memref import alloca, DYNAMIC, MemRef, MemRefFactory
from pydsl.tensor import Tensor, TensorFactory
from pydsl.type import F32, F64, SInt32, SInt64, UInt8, UInt32, UInt64
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
            linalg.add(t1, t1, out=m2)


def test_elemwise_bin_wrong_shape():
    with compilation_failed_from(TypeError):

        @compile()
        def f(t1: Tensor[F64, 5], t2: Tensor[F64, 8]):
            linalg.sub(t1, t1, out=t2)

    with compilation_failed_from(TypeError):

        @compile()
        def g(t1: Tensor[F64, 5], t2: Tensor[F64, 8]):
            linalg.sub(t1, t2, out=t1)


def test_linalg_fill():
    @compile()
    def fill_tensor(x: F32, t1: Tensor[F64, 10, 20]) -> Tensor[F64, 10, 20]:
        return linalg.fill(t1, x)

    @compile()
    def fill_memref(x: SInt32, m1: MemRef[SInt32, 100]) -> MemRef[SInt32, 100]:
        return linalg.fill(m1, x)

    # Check return value for tensor
    n1 = multi_arange((10, 20), np.float64)
    assert (fill_tensor(123, n1) == 123).all()

    # Check both return value and initial memory for memref
    n2 = multi_arange((100,), np.int32)
    res = fill_memref(456, n2)
    assert (res == 456).all()
    assert (n2 == 456).all()


def test_reduce():
    combs_pydsl = [arith.max, arith.min]
    combs_np = [np.maximum, np.minimum]
    inits = [3.2, 4.6]

    for comb_pydsl, comb_np, init in zip(combs_pydsl, combs_np, inits):

        @compile()
        def f_tensor(
            x: Tensor[F32, 8, 20], init: Tensor[F32, 8]
        ) -> Tensor[F32, 8]:
            return linalg.reduce(comb_pydsl, x, init=init, dims=[1])

        @compile()
        def f_memref(
            x: MemRef[F32, 8, 20], init: MemRef[F32, 8]
        ) -> MemRef[F32, 8]:
            return linalg.reduce(comb_pydsl, x, init=init, dims=[1])

        in_arr = multi_arange((8, 20), np.float32) / 10 - 5
        init_arr = np.full((8,), init, dtype=np.float32)
        cor_res = comb_np.reduce(in_arr, initial=init, axis=1)

        res_tensor = f_tensor(in_arr.copy(), init_arr.copy())
        assert np.allclose(res_tensor, cor_res)

        res_memref = f_memref(in_arr, init_arr)
        assert np.allclose(res_memref, cor_res)
        assert (res_memref == init_arr).all()


def test_reduce_bad_axes():
    def check(dims: list[int]):
        with compilation_failed_from(ValueError):

            @compile()
            def f(arr: Tensor[F32, 10, 20, 30], init: Tensor[F32, 10, 20]):
                linalg.reduce(arith.max, arr, init=init, dims=dims)

    check([2, 1])
    check([1, 3])
    check([-1, 2])
    check([4])
    check([1, 1])


def test_reduce_multi_type():
    @InlineFunction.generate()
    def sum(a: UInt64, b: UInt32) -> UInt64:
        return a + b

    @compile()
    def f(
        arr: MemRef[UInt32, DYNAMIC, DYNAMIC, DYNAMIC],
        out: MemRef[UInt64, DYNAMIC],
    ):
        linalg.reduce(sum, arr, init=out, dims=[0, 2])

    arr = multi_arange((6, 7, 8), np.uint32) + int(1e9)
    out = np.zeros((7,), dtype=np.uint64)

    cor_res = arr.astype(np.uint64)
    cor_res = np.add.reduce(cor_res, axis=2)
    cor_res = np.add.reduce(cor_res, axis=0)

    f(arr, out)
    assert (out == cor_res).all()


def test_reduce_non_commutative():
    @InlineFunction.generate()
    def combine(a, b) -> typing.Any:
        return a * 16 + b

    @compile()
    def f(arr: MemRef[UInt8, 4, 4]) -> UInt64:
        out = alloca((), UInt64)
        linalg.reduce(combine, arr, init=out, dims=[0, 1])
        return out[()]

    arr = multi_arange((4, 4), np.uint8)
    assert f(arr) == 0x123456789ABCDEF


def _broadcast_np(
    a: np.ndarray, dim_inds: Iterable[int], dim_sizes: Iterable[int]
) -> np.ndarray:
    a = np.expand_dims(a, dim_inds)

    tgt_shape = list(a.shape)
    for ind, sz in zip(dim_inds, dim_sizes, strict=True):
        tgt_shape[ind] = sz

    return np.broadcast_to(a, tgt_shape)


def test_broadcast():
    # TODO: @compile functions can be moved out of the loop once we support
    # templating

    def test(
        init_dims: tuple[int],
        new_dim_inds: tuple[int],
        new_dim_sizes: tuple[int],
    ):
        n1 = multi_arange(init_dims, np.int32)
        cor_res = _broadcast_np(n1, new_dim_inds, new_dim_sizes)
        tgt_empty = -multi_arange(cor_res.shape, np.int32)

        MemRefT1 = MemRefFactory(init_dims, SInt32)
        MemRefT2 = MemRefFactory(cor_res.shape, SInt32)
        TensorT1 = TensorFactory(init_dims, SInt32)
        TensorT2 = TensorFactory(cor_res.shape, SInt32)

        @compile()
        def f_tensor(x: TensorT1, out: TensorT2) -> TensorT2:
            return linalg.broadcast(x, out=out, dims=new_dim_inds)

        @compile()
        def f_memref(x: MemRefT1, out: MemRefT2) -> MemRefT2:
            return linalg.broadcast(x, out=out, dims=new_dim_inds)

        # For Tensor, check return value only
        assert (f_tensor(n1.copy(), tgt_empty.copy()) == cor_res).all()

        # For MemRef, check that out is modfied as well
        act_res = f_memref(n1, tgt_empty)
        assert (act_res == cor_res).all()
        assert (tgt_empty == cor_res).all()

    test((10, 20), (1,), (30,))
    test((10,), (0, 2), (20, 30))
    test((10, 20), (0, 1, 3, 4, 6, 7), (3, 4, 5, 6, 7, 8))


def test_broadcast_multi_type():
    with compilation_failed_from(TypeError):

        @compile()
        def f(arr: MemRef[SInt32, 10], out: MemRef[SInt64, 4, 10]):
            linalg.broadcast(arr, out=out, dims=[0])


def test_broadcast_bad_shapes():
    with compilation_failed_from(ValueError):

        @compile()
        def f(arr: MemRef[SInt32, 10], out: MemRef[SInt32, 4, 9]):
            linalg.broadcast(arr, out=out, dims=[0])


def test_matmul():
    @template()
    def f[TA, TB, TC](A: TA, B: TB, init: TC) -> TC:
        return linalg.matmul(A, B, init=init)

    def test(
        dimsA: tuple[int], dimsB: tuple[int], dimsC: tuple[int], good: bool
    ):
        A = multi_arange(dimsA, np.int64)
        B = multi_arange(dimsB, np.int64)
        C = multi_arange(dimsC, np.int64)

        TensorTA = TensorFactory(dimsA, SInt64)
        TensorTB = TensorFactory(dimsB, SInt64)
        TensorTC = TensorFactory(dimsC, SInt64)
        TensorTD = TensorFactory((DYNAMIC, DYNAMIC), SInt64)

        MemRefTA = MemRefFactory(dimsA, SInt64)
        MemRefTB = MemRefFactory(dimsB, SInt64)
        MemRefTC = MemRefFactory(dimsC, SInt64)

        if good:
            cor_res = A @ B + C

            # For Tensor, check return value only
            assert (
                f[TensorTA, TensorTB, TensorTC](A.copy(), B.copy(), C.copy())
                == cor_res
            ).all()
            assert (
                f[TensorTD, TensorTD, TensorTD](A.copy(), B.copy(), C.copy())
                == cor_res
            ).all()
            assert (
                f[TensorTA, TensorTD, TensorTD](A.copy(), B.copy(), C.copy())
                == cor_res
            ).all()

            # For MemRef, check that out is modfied as well
            act_res = f[MemRefTA, MemRefTB, MemRefTC](A, B, C)
            assert (act_res == cor_res).all()
            assert (C == cor_res).all()
        else:
            with compilation_failed_from(TypeError):
                f[TensorTA, TensorTB, TensorTC](A.copy(), B.copy(), C.copy())

            with compilation_failed_from(TypeError):
                f[MemRefTA, MemRefTB, MemRefTC](A, B, C)

    test((3, 4), (4, 5), (3, 5), True)
    test((3, 5), (4, 5), (3, 5), False)


def test_matmul_no_init():
    # TODO support this
    with compilation_failed_from(ValueError):

        @compile()
        def f(A: Tensor[F64, 1, 1], B: Tensor[F64, 1, 1]) -> Tensor[F64, 1, 1]:
            return linalg.matmul(A, B)


def test_batch_matmul():
    @template()
    def f[TA, TB, TC](A: TA, B: TB, init: TC) -> TC:
        return linalg.batch_matmul(A, B, init=init)

    def test(
        dimsA: tuple[int], dimsB: tuple[int], dimsC: tuple[int], good: bool
    ):
        A = multi_arange(dimsA, np.int64)
        B = multi_arange(dimsB, np.int64)
        C = multi_arange(dimsC, np.int64)

        TensorTA = TensorFactory(dimsA, SInt64)
        TensorTB = TensorFactory(dimsB, SInt64)
        TensorTC = TensorFactory(dimsC, SInt64)
        TensorTD = TensorFactory((DYNAMIC, DYNAMIC, DYNAMIC), SInt64)

        MemRefTA = MemRefFactory(dimsA, SInt64)
        MemRefTB = MemRefFactory(dimsB, SInt64)
        MemRefTC = MemRefFactory(dimsC, SInt64)

        if good:
            cor_res = A @ B + C

            # For Tensor, check return value only
            assert (
                f[TensorTA, TensorTB, TensorTC](A.copy(), B.copy(), C.copy())
                == cor_res
            ).all()
            assert (
                f[TensorTD, TensorTD, TensorTD](A.copy(), B.copy(), C.copy())
                == cor_res
            ).all()

            # For MemRef, check that out is modfied as well
            act_res = f[MemRefTA, MemRefTB, MemRefTC](A.copy(), B.copy(), C)
            assert (act_res == cor_res).all()
            assert (C == cor_res).all()
        else:
            f[TensorTA, TensorTB, TensorTC](A.copy(), B.copy(), C.copy())

    test((2, 3, 4), (2, 4, 5), (2, 3, 5), True)

    with compilation_failed_from(TypeError):
        test((2, 3, 5), (2, 4, 5), (2, 3, 5), False)

    with compilation_failed_from(TypeError):
        test((2, 3, 4), (3, 4, 5), (2, 3, 5), False)


def test_batch_matmul_no_init():
    # TODO support this
    with compilation_failed_from(ValueError):

        @compile()
        def f(
            A: Tensor[F64, 1, 1, 1], B: Tensor[F64, 1, 1, 1]
        ) -> Tensor[F64, 1, 1, 1]:
            return linalg.batch_matmul(A, B)


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
    run(test_reduce)
    run(test_reduce_bad_axes)
    run(test_reduce_multi_type)
    run(test_reduce_non_commutative)
    run(test_broadcast)
    run(test_broadcast_multi_type)
    run(test_broadcast_bad_shapes)
    run(test_matmul)
    run(test_matmul_no_init)
    run(test_batch_matmul)
    run(test_batch_matmul_no_init)
