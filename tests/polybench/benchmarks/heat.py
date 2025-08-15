import sys


from pydsl.type import Index, F32
from pydsl.memref import MemRefFactory, DYNAMIC
from pydsl.frontend import compile
from pydsl.affine import (
    affine_range as arange,
)
import numpy as np
from timeit import timeit
import ctypes

MemrefF32 = MemRefFactory((DYNAMIC, DYNAMIC, DYNAMIC), F32)


@compile()
def heat(tsteps: Index, n: Index, A: MemrefF32, B: MemrefF32) -> F32:
    a: F32 = 2.0
    b: F32 = 0.125
    for t in arange(tsteps):
        for i in arange(1, n - 1):
            for j in arange(1, n - 1):
                for k in arange(1, n - 1):
                    B[i, j, k] = (
                        b * (A[i + 1, j, k] - a * A[i, j, k] + A[i - 1, j, k])
                        + b
                        * (A[i, j + 1, k] - a * A[i, j, k] + A[i, j - 1, k])
                        + b
                        * (A[i, j, k + 1] - a * A[i, j, k] + A[i, j, k - 1])
                        + A[i, j, k]
                    )
        for i in arange(1, n - 1):
            for j in arange(1, n - 1):
                for k in arange(1, n - 1):
                    A[i, j, k] = (
                        b * (B[i + 1, j, k] - a * B[i, j, k] + B[i - 1, j, k])
                        + b
                        * (B[i, j + 1, k] - a * B[i, j, k] + B[i, j - 1, k])
                        + b
                        * (B[i, j, k + 1] - a * B[i, j, k] + B[i, j, k - 1])
                        + B[i, j, k]
                    )

    return b


def main(
    current_dataset: str, output_array: bool, c_test: bool, ctest_obj: str
):
    datasets = {
        "MINI_DATASET": (20, 10),
        "SMALL_DATASET": (40, 20),
        "MEDIUM_DATASET": (100, 40),
        "LARGE_DATASET": (500, 120),
        "EXTRALARGE_DATASET": (1000, 200),
    }

    tsteps, n = datasets.get(current_dataset, (40, 20))

    results = {}
    results["array"] = ""
    results["perf"] = -1.0
    results["c_correctness"] = ""
    results["c_perf"] = -1.0

    if c_test:
        lib = ctypes.CDLL(ctest_obj)
        heat_c = lib.kernel_heat_3d

        heat_c.argtypes = [
            ctypes.c_int,  # tsteps
            ctypes.c_int,  # n
            ctypes.POINTER(ctypes.c_float),  # A
            ctypes.POINTER(ctypes.c_float),  # B
        ]

    A = np.fromfunction(
        lambda i, j, k: (i + j + (n - k)) * 10 / (n),
        (n, n, n),
        dtype=np.float32,
    )
    B = np.empty_like(A)
    B[:] = A
    A_copy = A.copy()
    B_copy = B.copy()
    perf = timeit(lambda: heat(tsteps, n, A, B), number=1)
    if output_array:
        arr_out = "==BEGIN DUMP_ARRAYS==\n"
        arr_out += "begin dump: A"
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if (i * n * n + j * n + k) % 20 == 0:
                        arr_out += "\n"
                    arr_out += f"{A[i, j, k]:.2f} "
        arr_out += "\nend   dump: A\n"
        arr_out += "==END   DUMP_ARRAYS==\n"
        results["array"] = arr_out
    results["perf"] = perf

    if c_test:
        a_ptr = A_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        b_ptr = B_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        perf = timeit(lambda: heat_c(tsteps, n, a_ptr, b_ptr), number=1)
        max_difference = 0.0
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    max_difference = max(
                        max_difference, abs(A_copy[i, j, k] - A[i, j, k])
                    )
        results["c_perf"] = perf
        if max_difference < 0.001:
            results["c_correctness"] = "results are correct."
        else:
            results["c_correctness"] = (
                f"results incorrect! Max value difference is {max_difference:2f}"
            )
    return results


if __name__ == "__main__":
    result = main("SMALL_DATASET", True, False, "")
    print(result["array"])
    print(result["perf"])
