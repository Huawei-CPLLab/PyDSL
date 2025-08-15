import sys


from pydsl.type import Index, F32
from pydsl.memref import MemRefFactory, DYNAMIC
from pydsl.math import sqrt
from pydsl.frontend import compile
from pydsl.affine import (
    affine_range as arange,
)

MemrefF32 = MemRefFactory((DYNAMIC, DYNAMIC), F32)
import numpy as np
from timeit import timeit
import ctypes


@compile()
def cholesky(n: Index, A: MemrefF32) -> None:
    a: F32 = 1.0
    b: F32 = 0.0
    for i in arange(n):
        for j in arange(i):
            for k in arange(j):
                A[i, j] = A[i, j] - A[i, k] * A[j, k]
            A_ij = A[i, j]
            A_jj = A[j, j]
            A_div = A_ij / A_jj
            A[i, j] = A_div
        for k in arange(i):
            A_ik1 = A[i, k]
            A_ik2 = A_ik1 * A_ik1
            A_ii = A[i, i]
            A_diff = A_ii - A_ik2
            A[i, i] = A_diff
        A_ii = A[i, i]
        A_sqrt = sqrt(A_ii)
        A[i, i] = A_sqrt


def main(
    current_dataset: str, output_array: bool, c_test: bool, ctest_obj: str
):
    datasets = {
        "MINI_DATASET": 40,
        "SMALL_DATASET": 120,
        "MEDIUM_DATASET": 400,
        "LARGE_DATASET": 2000,
        "EXTRALARGE_DATASET": 4000,
    }

    n = datasets.get(current_dataset, 120)

    results = {}
    results["array"] = ""
    results["perf"] = -1.0
    results["c_correctness"] = ""
    results["c_perf"] = -1.0
    if c_test:
        lib = ctypes.CDLL(ctest_obj)
        cholesky_c = lib.kernel_cholesky

        cholesky_c.argtypes = [
            ctypes.c_int,  # n
            ctypes.POINTER(ctypes.c_float),  # a
        ]

    a = np.zeros((n, n)).astype(np.float32)

    # init array
    for i in range(n):
        for j in range(i + 1):
            a[i, j] = -(j % n) / n + 1
        for j in range(i + 1, n):
            a[i, j] = 0.0
        a[i, i] = 1.0

    # make the matrix positive semmi-definite
    b = np.zeros((n, n)).astype(np.float32)
    for r in range(n):
        for s in range(n):
            b[r, s] = 0.0
    for t in range(n):
        for r in range(n):
            for s in range(n):
                b[r, s] += a[r, t] * a[s, t]
    for r in range(n):
        for s in range(n):
            a[r, s] = b[r, s]
    a_copy = a.copy()

    perf = timeit(lambda: cholesky(n, a), number=1)
    if output_array:
        arr_out = "==BEGIN DUMP_ARRAYS==\n"
        arr_out += "begin dump: a"
        for i in range(n):
            for j in range(n):
                if ((i * n + j) % 20) == 0:
                    arr_out += "\n"
                arr_out += f"{a[i, j]:.2f} "
        arr_out += "\nend   dump: a\n"
        arr_out += "==END   DUMP_ARRAYS==\n"
        results["array"] = arr_out
    results["perf"] = perf

    if c_test:
        a_ptr = a_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        perf = timeit(lambda: cholesky_c(n, a_ptr), number=1)
        max_difference = 0.0
        for i in range(n):
            for j in range(n):
                max_difference = max(
                    max_difference, abs(a_copy[i, j] - a[i, j])
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
