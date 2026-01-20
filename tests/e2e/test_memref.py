import gc
import math
import numpy as np
from numpy.lib.stride_tricks import as_strided
import weakref

from pydsl.affine import affine_range as arange
from pydsl.frontend import compile
from pydsl.gpu import GPU_AddrSpace
import pydsl.linalg as linalg
import pydsl.memref as memref
from pydsl.memref import (
    alloc,
    alloca,
    dealloc,
    DYNAMIC,
    MemRef,
    MemRefFactory,
    collapse_shape,
)
from pydsl.type import Bool, F32, F64, Index, SInt16, Tuple, UInt32
from helper import compilation_failed_from, failed_from, multi_arange, run

MemRefI16_2 = MemRef.get_fully_dynamic(SInt16, 2)
MemRefU32_1 = MemRef.get_fully_dynamic(UInt32, 1)
MemRefF32_2 = MemRef.get_fully_dynamic(F32, 2)
MemRefF64_4 = MemRef.get_fully_dynamic(F64, 4)


def test_load_implicit_index_uint32():
    @compile(globals())
    def f(m2: MemRef[UInt32, 2], m2x2: MemRef[UInt32, 2, 2]):
        m2[1] = UInt32(5)
        m2x2[1, 1] = UInt32(5)

    n2 = np.empty((2,), dtype=np.uint32)
    n2x2 = np.empty((2, 2), dtype=np.uint32)
    f(n2, n2x2)
    assert n2[1] == 5
    assert n2x2[1, 1] == 5


MemRef2F64 = MemRefFactory((2,), F64)
MemRef2x2F64 = MemRefFactory((2, 2), F64)


def test_load_implicit_index_f64():
    @compile(globals())
    def f(m2: MemRef[F64, 2], m2x2: MemRef[F64, 2, 2]) -> Tuple[Bool, Bool]:
        m2[1] = 5
        m2x2[1, 1] = 5.1
        return m2[0] == 3, m2x2[0, 0] == 3.1

    n2 = np.zeros((2,), dtype=np.float64)
    n2x2 = np.zeros((2, 2), dtype=np.float64)
    n2[0] = 3
    n2x2[0, 0] = 3.1

    res1, res2 = f(n2, n2x2)

    assert res1
    assert res2
    assert math.isclose(n2[1], 5)
    assert math.isclose(n2x2[1, 1], 5.1)


def test_load_wrong_shape():
    @compile(globals())
    def f(m: MemRef[F64, 2, 2]):
        pass

    wrong_shape = np.zeros((2, 3), dtype=np.float64)

    with failed_from(TypeError):
        f(wrong_shape)


def test_store_wrong_rank():
    with compilation_failed_from(IndexError):

        @compile()
        def f(m1: MemRef[F32, 10, 10]):
            m1[1, 2, 3] = 5

    with compilation_failed_from(IndexError):

        @compile()
        def f(m1: MemRef[F32, 10, 10]):
            m1[2] = 8


def test_return_memref():
    @compile(globals())
    def f(m: MemRef[F64, 2, 2]) -> MemRef[F64, 2, 2]:
        return m

    m = np.asarray([[1, 2], [3, 4]], dtype=np.float64)
    assert (f(m) == m).all()


def test_return_tuple_memref():
    @compile(globals())
    def f(m: MemRef[F64, 2, 2]) -> Tuple[MemRef[F64, 2, 2]]:
        return (m,)

    m = np.asarray([[1, 2], [3, 4]], dtype=np.float64)
    assert (f(m)[0] == m).all()


def test_alloca_scalar():
    @compile()
    def f() -> UInt32:
        m_scalar = alloca((1,), UInt32)

        for i in arange(10):
            m_scalar[0] = i

        return m_scalar[0]

    assert f() == 9


def test_alloca_dynamic():
    @compile()
    def f(a: Index, b: Index) -> Tuple[UInt32, UInt32]:
        m = alloca((a, 2, b), UInt32)

        m[0, 0, 0] = 1
        m[a - 1, 1, b - 1] = 2

        return m[0, 0, 0], m[a - 1, 1, b - 1]

    for i in range(1, 3):
        assert f(i, i) == (1, 2)


def test_alloc_scalar():
    @compile()
    def f() -> MemRef[UInt32, 1]:
        m_scalar = alloc((1,), UInt32)
        m_scalar[0] = 1
        return m_scalar

    assert (f() == np.asarray([1], dtype=np.uint32)).all()


def test_dealloc():
    """
    Doesn't really test whether the memref is deallocated, since that's hard
    to test, but does test that the syntax of dealloc works.
    """

    @compile()
    def f() -> SInt16:
        m = alloc((3, Index(7)), SInt16)
        m[1, 2] = 5
        res = m[1, 2]
        dealloc(m)
        return res

    assert f() == 5

    mlir = f.emit_mlir()
    assert r"memref.dealloc" in mlir


def test_alloc_memory_space():
    """
    We can't use GPU address spaces on CPU, so just test if it compiles and
    the MLIR is reasonable.
    """

    @compile(auto_build=False)
    def f():
        m = alloc((4, 2), F32, memory_space=GPU_AddrSpace.Global)
        dealloc(m)

    mlir = f.emit_mlir()
    assert r"memref<4x2xf32, #gpu.address_space<global>>" in mlir


def test_alloc_align():
    @compile()
    def f() -> MemRef[F64, 4, 6]:
        return alloc((4, 6), F64, alignment=12345)

    n1 = f()
    assert n1.ctypes.data % 12345 == 0


def test_alloc_bad_align():
    with compilation_failed_from(TypeError):

        @compile()
        def f():
            alloc((4, 6), F64, alignment="xyz")

    with compilation_failed_from(ValueError):

        @compile()
        def f():
            alloc((4, 6), F64, alignment=-123)


def test_slice_memory_space():
    """
    We can't use GPU address spaces on CPU, so just test if it compiles.
    """

    MemRefT = MemRef.get((DYNAMIC,), F32, memory_space=GPU_AddrSpace.Global)

    @compile(auto_build=False)
    def f(m: MemRefT):
        n = m[:]

    # expect the line
    # %subview = memref.subview %arg0[%c0_0] [%1] [%c1] : ⤶
    # memref<?xf32, #gpu.address_space<global>> to ⤶
    # memref<?xf32, strided<[?], offset: ?>, #gpu.address_space<global>>
    mlir = f.emit_mlir()
    matched_lines = [
        line
        for line in mlir.splitlines()
        if "memref.subview" in line
        and line.count("#gpu.address_space<global>") >= 2
    ]
    assert len(matched_lines) == 1


def test_load_strided():
    MemRefStrided = MemRefFactory((10, 5), F32, offset=123, strides=(1, 10))

    @compile()
    def f(m1: MemRefStrided) -> MemRefStrided:
        return m1

    @compile()
    def g(m1: MemRefStrided) -> MemRefStrided:
        return linalg.exp(m1)

    f32_sz = np.float32().nbytes
    n1 = multi_arange((10, 5), np.float32)
    n1 = as_strided(n1, strides=(f32_sz, 10 * f32_sz))
    assert (f(n1) == n1).all()

    cor_res = np.exp(n1)
    assert np.allclose(g(n1), cor_res)


def test_load_strided_big():
    SZ = int(1e5)
    MemRefStrided = MemRefFactory((2, 4), UInt32, strides=((SZ - 1) * 4, 1))

    @compile()
    def f(m1: MemRefStrided) -> MemRefStrided:
        return m1

    u32_sz = np.uint32().nbytes
    n1 = multi_arange((SZ, 4), np.uint32)
    n1 = as_strided(
        n1, shape=(2, 4), strides=((SZ - 1) * 4 * u32_sz, 1 * u32_sz)
    )
    assert (f(n1) == n1).all()


def test_load_strided_wrong():
    MemRefStrided = MemRefFactory((6, 4, 5), F32, strides=(2, 10, 3))

    @compile()
    def f(m1: MemRefStrided):
        pass

    @compile()
    def g(m1: MemRef[F32, 6, 4, 5]):
        pass

    n1 = multi_arange((100,), np.float32)

    with failed_from(TypeError):
        # ndarray is not strided
        f(n1)

    f32_sz = np.float32().nbytes
    n1 = as_strided(
        n1, shape=(6, 4, 5), strides=(2 * f32_sz, 10 * f32_sz, 4 * f32_sz)
    )

    with failed_from(TypeError):
        # ndarray has wrong strides: (2, 10 3) vs (2, 10, 4)
        f(n1)

    with failed_from(TypeError):
        # g wants a non-strided input
        g(n1)


def test_store_return_strided_memref():
    # These strides are okay, since they do not cause any self-overlap
    # with dimension size only (5, 8)
    MemRefStrided = MemRefFactory((5, 8), SInt16, strides=(4, 7))

    @compile()
    def f(m1: MemRefStrided) -> MemRefStrided:
        m1[3, 4] = 101
        m1[0, 0] = 102
        m1[4, 7] = -103
        return m1

    i16_sz = np.int16().nbytes
    n1 = multi_arange((100,), np.int16)
    n1 = as_strided(n1, shape=(5, 8), strides=(4 * i16_sz, 7 * i16_sz))
    n1_cp = n1.copy()
    test_res = f(n1)

    assert (n1 == test_res).all()  # memref should be modified in-place
    assert not (n1_cp == test_res).all()
    n1_cp[3, 4] = 101
    n1_cp[0, 0] = 102
    n1_cp[4, 7] = -103
    assert (n1_cp == test_res).all()  # return value should also be correct


def test_dynamic_strided_memref():
    MemRefStrided = MemRefFactory((6, 4, 5), F32, strides=(7, DYNAMIC, 1))

    @compile()
    def f(m1: MemRefStrided):
        m1[4, 2, 3] = 1.234
        m1[1, 1, 2] = 2.345
        linalg.exp(m1)

    f32_sz = np.float32().nbytes
    n1 = multi_arange((200,), np.float32) / 50
    n1 = as_strided(
        n1, shape=(6, 4, 5), strides=(7 * f32_sz, 50 * f32_sz, 1 * f32_sz)
    )
    n1_cp = n1.copy()
    f(n1)

    assert not (n1 == n1_cp).all()
    n1_cp[4, 2, 3] = 1.234
    n1_cp[1, 1, 2] = 2.345
    n1_cp = np.exp(n1_cp)
    assert np.allclose(n1, n1_cp)


def test_load_slice():
    @compile()
    def f(m1: MemRef[F64, 4, DYNAMIC, 1, 6]) -> MemRefF64_4:
        return m1[1, 2:7:3, :, :]

    n1 = multi_arange((4, 8, 1, 6), np.float64)
    assert (f(n1) == n1[1, 2:7:3, :, :]).all()


def test_load_compose_strided():
    @compile()
    def f(m1: MemRef[F32, 10, 20]) -> MemRefF32_2:
        return m1[1::3, ::2][::2, :9:4]

    n1 = multi_arange((10, 20), np.float32)
    assert (f(n1) == n1[1::3, ::2][::2, :9:4]).all()


def test_load_slice_store():
    @compile()
    def f(m1: MemRef[F64, 10, DYNAMIC], xLo: Index, xHi: Index):
        m1_sub = m1[xLo:xHi:3, 3::4]
        m1_sub[2, 3] = 1004
        m1_sub[1, 0] = 1001
        m1_sub[2, 3] = 1008

    n1 = multi_arange((10, 20), np.float64)
    n2 = np.copy(n1)
    f(n2, 1, 10)
    assert not (n2 == n1).all()
    n1_sub = n1[1:10:3, 3::4]
    n1_sub[2, 3] = 1004
    n1_sub[1, 0] = 1001
    n1_sub[2, 3] = 1008
    assert (n2 == n1).all()


def test_link_ndarray():
    """
    Test whether pointers are correctly attached to ndarrays returned from a
    function if they overlap with input ndarrays.
    """

    def get_root(arr: np.ndarray):
        while arr.base is not None:
            arr = arr.base
        return arr

    @compile()
    def f(m1: MemRefU32_1, m2: MemRefU32_1) -> Tuple[MemRefU32_1, MemRefU32_1]:
        res1 = m1[123::3]
        res2 = m2
        return (res1, res2)

    # First, check what happens when m1 and m2 are derived from different roots
    n1 = multi_arange((1000,), np.uint32)
    n2 = multi_arange((500,), np.uint32) + 1234
    cor_res1 = n1.copy()[123::3]
    cor_res2 = n2.copy()
    res1, res2 = f(n1, n2)
    n1_root_ref = weakref.ref(get_root(n1))
    n2_root_ref = weakref.ref(get_root(n2))

    n1 = None
    n2 = None
    gc.collect()
    assert n1_root_ref() is not None
    assert n2_root_ref() is not None
    assert (res1 == cor_res1).all()
    assert (res2 == cor_res2).all()

    res1 = None
    gc.collect()
    assert n1_root_ref() is None
    assert n2_root_ref() is not None

    res2 = None
    gc.collect()
    assert n1_root_ref() is None
    assert n2_root_ref() is None

    # Now check what happens if m1 and m2 are derived from the same ndarray
    n3 = multi_arange((1000,), np.uint32) + 8765
    cor_res3 = n3.copy()[1::5][123::3]
    cor_res4 = n3.copy()[2::5]
    res3, res4 = f(n3[1::5], n3[2::5])
    n3_root_ref = weakref.ref(get_root(n3))

    n3 = None
    gc.collect()
    assert n3_root_ref() is not None
    assert (res3 == cor_res3).all()
    assert (res4 == cor_res4).all()

    res3 = None
    gc.collect()
    assert n3_root_ref() is not None

    res4 = None
    gc.collect()
    assert n3_root_ref() is None


def test_chain_link_ndarray():
    """
    Check if the ndarray returned by a PyDSL function can be passed to another
    PyDSL function and still result in correct deallocation prevention
    references.
    """

    def get_root(arr: np.ndarray):
        while arr.base is not None:
            arr = arr.base
        return arr

    @compile()
    def f(m1: MemRefI16_2) -> MemRefI16_2:
        return m1[1:, 1:]

    n1 = multi_arange((10, 10), np.int16)
    res = f(f(f(n1)))
    n1_ref = weakref.ref(get_root(n1))
    assert (res == n1[3:, 3:]).all()

    n1 = None
    gc.collect()
    assert n1_ref() is not None

    res = None
    gc.collect()
    assert n1_ref() is None


def test_zero_d():
    @compile()
    def f(m1: MemRef[UInt32]) -> Tuple[UInt32, MemRef[UInt32]]:
        x = m1[()]
        m1[()] = 456
        return x, m1

    n1 = np.array(123, dtype=np.uint32)
    res1, res2 = f(n1)
    assert res1 == 123 and res2 == 456
    assert isinstance(res2, np.ndarray)
    assert res2.shape == ()


def test_collapse_shape():
    @compile()
    def my_func(a: MemRef[F32, 1, 3]) -> MemRef[F32, 3]:
        return collapse_shape(a, [[0, 1]])

    n1 = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    assert all([a == b for a, b in zip(my_func(n1), [1.0, 2.0, 3.0])])


def test_cast_basic():
    @compile()
    def f(
        m1: MemRef[F32, 10, DYNAMIC, 30],
    ) -> MemRef[F32, DYNAMIC, 20, DYNAMIC]:
        m1 = m1.cast((DYNAMIC, 20, 30), strides=(600, 30, 1))
        m1 = m1.cast((DYNAMIC, DYNAMIC, DYNAMIC))
        m1 = m1.cast((10, 20, 30), strides=None)
        m1 = m1.cast((DYNAMIC, 20, DYNAMIC))
        return m1

    n1 = multi_arange((10, 20, 30), dtype=np.float32)
    cor_res = n1.copy()
    assert (f(n1) == cor_res).all()


def test_cast_strided():
    MemRef1 = MemRef.get((8, DYNAMIC), SInt16, offset=10, strides=(1, 8))
    MemRef2 = MemRef.get((8, 4), SInt16, offset=DYNAMIC, strides=(DYNAMIC, 8))

    @compile()
    def f(m1: MemRef1) -> MemRef2:
        m1.cast(strides=(1, DYNAMIC))
        m1 = m1.cast((8, 4), offset=DYNAMIC, strides=(DYNAMIC, 8))
        return m1

    i16_sz = np.int16().nbytes
    n1 = multi_arange((8, 4), np.int16)
    n1 = as_strided(n1, shape=(8, 4), strides=(i16_sz, 8 * i16_sz))
    cor_res = n1.copy()
    assert (f(n1) == cor_res).all()


def test_cast_bad():
    with compilation_failed_from(ValueError):

        @compile()
        def f1(m1: MemRef[UInt32, 5, 8]):
            m1.cast((5, 8, 7))

    with compilation_failed_from(ValueError):

        @compile()
        def f2(m1: MemRef[F64, DYNAMIC, 4]):
            m1.cast((DYNAMIC, 5))

    with compilation_failed_from(ValueError):
        MemRef1 = MemRef.get((8, DYNAMIC), F32, offset=10, strides=(1, 8))

        @compile()
        def f3(m1: MemRef1):
            m1.cast(strides=(DYNAMIC, 16))


def test_copy_basic():
    @compile()
    def f(m1: MemRef[SInt16, 10, DYNAMIC], m2: MemRef[SInt16, 10, 10]):
        memref.copy(m1, m2)

    n1 = multi_arange((10, 10), np.int16)
    n2 = np.zeros((10, 10), dtype=np.int16)
    cor_res = n1.copy()
    f(n1, n2)
    assert (n2 == cor_res).all()


def test_copy_overlap():
    @compile()
    def f(m1: MemRef[UInt32, 4000, 2000]):
        memref.copy(m1[100:3000], m1[600:3500])
        memref.copy(m1[1000:, :1900], m1[300:3300, 100:])

    n1 = multi_arange((4000, 2000), dtype=np.uint32)
    cor_res = n1.copy()
    cor_res[600:3500] = cor_res[100:3000]
    cor_res[300:3300, 100:] = cor_res[1000:, :1900]
    f(n1)
    assert (n1 == cor_res).all()


def test_copy_strided():
    @compile()
    def f(m1: MemRef[F32, 10, 10], m2: MemRef[F32, 8, 9]):
        memref.copy(m1[1::3, ::2], m2[3::2, 3:8])

    n1 = multi_arange((10, 10), np.float32)
    n2 = multi_arange((8, 9), np.float32) + 1000
    cor_res = n2
    cor_res[3::2, 3:8] = n1[1::3, ::2]
    f(n1, n2)
    assert np.allclose(n2, cor_res)


def test_copy_bad():
    with compilation_failed_from(ValueError):
        # Different shapes
        @compile()
        def f(m1: MemRef[F32, 5], m2: MemRef[F32, 6]):
            memref.copy(m1, m2)

    with compilation_failed_from(ValueError):
        # Different types
        @compile()
        def f(m1: MemRef[F32, 5], m2: MemRef[F64, 5]):
            memref.copy(m1, m2)


if __name__ == "__main__":
    run(test_load_implicit_index_uint32)
    run(test_load_implicit_index_f64)
    run(test_load_wrong_shape)
    run(test_store_wrong_rank)
    run(test_return_memref)
    run(test_return_tuple_memref)
    run(test_alloca_scalar)
    run(test_alloca_dynamic)
    run(test_alloc_scalar)
    run(test_alloc_align)
    run(test_alloc_bad_align)
    run(test_dealloc)
    run(test_alloc_memory_space)
    run(test_slice_memory_space)
    run(test_load_strided)
    run(test_load_strided_big)
    run(test_load_strided_wrong)
    run(test_store_return_strided_memref)
    run(test_dynamic_strided_memref)
    run(test_load_slice)
    run(test_load_compose_strided)
    run(test_load_slice_store)
    run(test_link_ndarray)
    run(test_chain_link_ndarray)
    run(test_zero_d)
    run(test_collapse_shape)
    run(test_cast_basic)
    run(test_cast_strided)
    run(test_cast_strided)
    run(test_cast_bad)
    run(test_copy_basic)
    run(test_copy_overlap)
    run(test_copy_strided)
    run(test_copy_bad)
