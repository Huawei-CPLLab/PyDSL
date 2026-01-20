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

MemrefF32NN = MemRefFactory((DYNAMIC, DYNAMIC), F32)
MemrefF32NM = MemRefFactory((DYNAMIC, DYNAMIC), F32)


@compile()
def syrk(
    n: Index, m: Index, alpha: F32, beta: F32, C: MemrefF32NN, A: MemrefF32NM
) -> None:
    for i in arange(n):
        for j in arange(i + 1):
            C[i, j] = C[i, j] * beta
        for k in arange(m):
            for j in arange(i + 1):
                C[i, j] = C[i, j] + alpha * A[i, k] * A[j, k]


def main(
    current_dataset: str, output_array: bool, c_test: bool, ctest_obj: str
):
    datasets = {
        "MINI_DATASET": (20, 30),
        "SMALL_DATASET": (60, 80),
        "MEDIUM_DATASET": (200, 240),
        "LARGE_DATASET": (1000, 1200),
        "EXTRALARGE_DATASET": (2000, 2600),
    }

    m, n = datasets.get(current_dataset, (60, 80))

    results = {}
    results["array"] = ""
    results["perf"] = -1.0
    results["c_correctness"] = ""
    results["c_perf"] = -1.0
    if c_test:
        lib = ctypes.CDLL(ctest_obj)
        syrk_c = lib.kernel_syrk

        syrk_c.argtypes = [
            ctypes.c_int,  # tsteps
            ctypes.c_int,  # n
            ctypes.c_float,  # alpha
            ctypes.c_float,  # beta
            ctypes.POINTER(ctypes.c_float),  # c
            ctypes.POINTER(ctypes.c_float),  # a
            ctypes.POINTER(ctypes.c_float),  # b
        ]

    a = np.zeros((n, m)).astype(np.float32)
    b = np.zeros((n, m)).astype(np.float32)
    c = np.zeros((n, n)).astype(np.float32)
    alpha = 1.5
    beta = 1.2

    # init array
    for i in range(n):
        for j in range(m):
            a[i, j] = ((i * j + 1) % n) / n
    for i in range(n):
        for j in range(n):
            c[i, j] = ((i * j + 2) % m) / m

    c_copy = c.copy()
    a_copy = a.copy()
    b_copy = b.copy()

    perf = timeit(lambda: syrk(n, m, alpha, beta, c, a), number=1)
    if output_array:
        arr_out = "==BEGIN DUMP_ARRAYS==\n"
        arr_out += "begin dump: C"
        for i in range(n):
            for j in range(n):
                if ((i * n + j) % 20) == 0:
                    arr_out += "\n"
                arr_out += f"{c[i, j]:.2f} "
        arr_out += "\nend   dump: C\n"
        arr_out += "==END   DUMP_ARRAYS==\n"
        results["array"] = arr_out
    results["perf"] = perf

    if c_test:
        c_ptr = c_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        a_ptr = a_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        b_ptr = b_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        perf = timeit(
            lambda: syrk_c(n, m, alpha, beta, c_ptr, a_ptr, b_ptr), number=1
        )
        max_difference = 0.0
        for i in range(n):
            for j in range(n):
                max_difference = max(
                    max_difference, abs(c_copy[i, j] - c[i, j])
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
