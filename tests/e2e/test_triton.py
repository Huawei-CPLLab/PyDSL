import numpy as np
from pydsl.frontend import compile
from pydsl.type import F32, UInt32, Bool, Index, Number
from pydsl.memref import MemRefFactory, DYNAMIC, MemRef
from pydsl.affine import affine_range as arange
from helper import run, multi_arange
import pytest
import shutil

try:
    import triton
    import triton.language as tl

    has_triton = shutil.which("triton-adapter-opt") is not None
except ImportError:
    has_triton = False

# Skips the entire file
pytestmark = pytest.mark.skipif(
    not has_triton, reason="Triton or triton-adapter-opt is not installed"
)


def test_triton_empty():
    @triton.jit
    def empty_kernel():
        pass

    @compile()
    def empty_func():
        empty_kernel()


def test_triton_types():
    @triton.jit
    def kernel(out_ptr, x: tl.float32, y: tl.int32, z: tl.int1):
        if z:
            tl.store(out_ptr, x)
        else:
            tl.store(out_ptr, y)

    @compile()
    def pydsl_type_kernel(
        out_fp: MemRef[F32, DYNAMIC], out_int: MemRef[UInt32, DYNAMIC]
    ):
        x: F32 = 1.0
        y: UInt32 = 1
        z: Bool = True
        kernel(out_fp, x, y, z)
        z = Bool(False)
        kernel(out_int, x, y, z)

    out_fp = np.zeros(1, dtype=np.float32)
    out_int = np.zeros(1, dtype=np.uint32)
    pydsl_type_kernel(out_fp, out_int)
    assert out_fp[0] == 1.0
    assert out_int[0] == 1


def test_triton_vec_add():
    @triton.jit
    def kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)

    size = 98432
    ArrayType = MemRefFactory((DYNAMIC,), F32)

    @compile()
    def func_test(x: ArrayType, y: ArrayType, out: ArrayType):
        size: Index = 98432
        BLOCK_SIZE = 64
        index_size = size // BLOCK_SIZE
        for i in arange(0, index_size):
            kernel(x, y, out, size, BLOCK_SIZE, i)

    out = np.zeros(size, dtype=np.float32)
    x = multi_arange((size,), dtype=np.float32)
    y = multi_arange((size,), dtype=np.float32)
    func_test(x, y, out)
    manual_out = x + y
    assert np.allclose(out, manual_out)


def test_triton_multiple_dimensions():
    @triton.jit
    def kernel(a_ptr, out_ptr):
        pid_a = tl.program_id(axis=0)
        pid_b = tl.program_id(axis=1)
        pid_c = tl.program_id(axis=2)
        offsets = (
            pid_a * 16
            + tl.arange(0, 16)
            + (pid_b * 16 * 16 + pid_c * 16 * 16 * 16)
        )

        a = tl.load(a_ptr + offsets)
        tl.store(out_ptr + offsets, a + 1.0)

    ArrayType = MemRefFactory((DYNAMIC,), F32)

    @compile()
    def func_test(a: ArrayType, out: ArrayType):
        for i in arange(16):
            for j in arange(16):
                for k in arange(16):
                    kernel(a, out, i, j, k)

    size = 16 * 16 * 16 * 16  # 65536
    out = np.zeros(size, dtype=np.float32)
    x = np.zeros(size, dtype=np.float32)
    func_test(x, out)
    manual_out = x + 1.0
    assert np.allclose(out, manual_out)


def test_triton_multiple_funcs_and_multiple_calls():
    @triton.jit
    def kernel1(x_ptr):
        x = tl.load(x_ptr)
        tl.store(x_ptr, x + 1.0)

    @triton.jit
    def kernel2(x_ptr):
        x = tl.load(x_ptr)
        tl.store(x_ptr, x + 2.0)

    ArrayType = MemRefFactory((DYNAMIC,), F32)

    @compile()
    def func_test(x: ArrayType):
        kernel1(x)
        kernel2(x)
        kernel1(x)

    mlir = func_test.emit_mlir()
    assert mlir.count("func.func @kernel1") == 1
    assert "func.func @kernel2" in mlir
    x = np.zeros(1, dtype=np.float32)
    func_test(x)
    assert x[0] == 4.0


def test_triton_different_signatures():
    @triton.jit
    def kernel(a_ptr, out_ptr):
        a = tl.load(a_ptr)
        tl.store(out_ptr, a)

    int_arr = MemRefFactory((DYNAMIC,), UInt32)
    fp_arr = MemRefFactory((DYNAMIC,), F32)

    @compile()
    def func_test(
        m_int: int_arr, out_int: int_arr, m_fp: fp_arr, out_fp: fp_arr
    ):
        kernel(m_int, out_int)
        kernel(m_fp, out_fp)

    m_int = np.asarray([123], np.uint32)
    out_int = np.zeros(1, dtype=np.uint32)
    m_fp = np.asarray([456], np.float32)
    out_fp = np.zeros(1, dtype=np.float32)
    func_test(m_int, out_int, m_fp, out_fp)
    assert m_int[0] == out_int[0]
    assert m_fp[0] == out_fp[0]


def test_triton_unused_args():
    @triton.jit
    def kernel(x_ptr, y_ptr, out_ptr, a, BLOCK_SIZE: tl.constexpr, b):
        pid = tl.program_id(axis=0)
        x = tl.load(x_ptr + pid)
        tl.store(out_ptr + pid, x + a)

    ArrayType = MemRefFactory((DYNAMIC,), F32)

    @compile()
    def func_test(x: ArrayType, y: ArrayType, out: ArrayType, size: Index):
        a: UInt32 = 3
        BLOCK_SIZE: Number = 64
        b: UInt32 = 1
        for i in arange(size):
            kernel(x, y, out, a, BLOCK_SIZE, b, i)

    size = 3
    x = np.zeros(size, dtype=np.float32)
    y = np.zeros(size, dtype=np.float32)
    out = np.zeros(size, dtype=np.float32)
    func_test(x, y, out, size)
    assert np.allclose(out, x + 3)


def test_triton_inlined_args():
    @triton.jit
    def kernel(x_ptr, out_ptr, a, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offsets)
        tl.store(out_ptr + offsets, x + a)

    ArrayType = MemRefFactory((DYNAMIC,), F32)

    @compile()
    def func_test(x: ArrayType, out: ArrayType, size: Index):
        for i in arange(size // 64):  # size // BLOCK_SIZE
            kernel(x, out, UInt32(3), 64, i)
        for i in arange(size // 128):
            kernel(out, out, UInt32(15), 128, i)

    size = 65536
    x = multi_arange((size,), np.float32)
    out = np.zeros(size, dtype=np.float32)
    func_test(x, out, size)
    assert np.allclose(out, x + 3 + 15)


if __name__ == "__main__":
    run(test_triton_empty)
    run(test_triton_types)
    run(test_triton_vec_add)
    run(test_triton_multiple_dimensions)
    run(test_triton_multiple_funcs_and_multiple_calls)
    run(test_triton_different_signatures)
    run(test_triton_unused_args)
    run(test_triton_inlined_args)
