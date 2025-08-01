from pydsl.frontend import compile
from pydsl.type import F32, Index, UInt32, Number
from pydsl.memref import MemRefFactory, DYNAMIC
from pydsl.affine import affine_range as arange, symbol as S
import numpy as np

import triton
import triton.language as tl


@triton.jit
def kernel(
    x_ptr,  # *Pointer* to first input vector.
    y_ptr,  # *Pointer* to second input vector.
    output_ptr,  # *Pointer* to output vector.
    n_elements,  # Size of the vector.
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process
    # NOTE: `constexpr` so it can be used as a shape value.
):
    # There are multiple 'programs' processing different data. We identify
    # which program we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64,
    # the programs would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)


size = 98432
VecType = MemRefFactory((DYNAMIC,), F32)


@compile(dump_mlir=True, auto_build=True)
def func_test(x: VecType, y: VecType, out: VecType):
    size: UInt32 = 98432
    BLOCK_SIZE: Number = 64
    idx: UInt32 = 1
    indexSize: Index = 1538  # size // BLOCK_SIZE
    for i in arange(0, indexSize):
        blockIndex = UInt32(i)
        kernel(x, y, out, size, BLOCK_SIZE, blockIndex, idx, idx)


np.random.seed(0)
out = np.zeros(size, dtype=np.float32)
emptyMemref = np.zeros(size, dtype=np.ubyte)
x = np.random.rand(size).astype(np.float32)
y = np.random.rand(size).astype(np.float32)
func_test(x, y, out)
manual_out = x + y
for i in range(size):
    if abs(out[i] - manual_out[i]) > 1e-5:
        print(f"Mismatch at index {i}: {out[i]} != {manual_out[i]}")
        break
else:
    print("All values match!")
