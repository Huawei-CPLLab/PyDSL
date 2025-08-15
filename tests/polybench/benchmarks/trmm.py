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


MemrefF32MM = MemRefFactory((DYNAMIC, DYNAMIC), F32)
MemrefF32MN = MemRefFactory((DYNAMIC, DYNAMIC), F32)


@compile()
def trmm(
    m: Index, n: Index, alpha: F32, A: MemrefF32MM, B: MemrefF32MN
) -> None:
    for i in arange(m):
        for j in arange(n):
            for k in arange(i + 1, m):
                B[i, j] = B[i, j] + B[k, j] * A[k, i]
            B[i, j] = alpha * B[i, j]


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
        trmm_c = lib.kernel_trmm

        trmm_c.argtypes = [
            ctypes.c_int,  # m
            ctypes.c_int,  # n
            ctypes.c_float,  # alpha
            ctypes.POINTER(ctypes.c_float),  # a
            ctypes.POINTER(ctypes.c_float),  # b
        ]

    a = np.zeros((m, m)).astype(np.float32)
    b = np.zeros((m, n)).astype(np.float32)
    alpha = 1.5

    # init array
    for i in range(m):
        for j in range(i):
            a[i, j] = ((i + j) % m) / m
        a[i, i] = 1
        for j in range(n):
            b[i, j] = ((n + (i - j)) % n) / n

    a_copy = a.copy()
    b_copy = b.copy()

    perf = timeit(lambda: trmm(m, n, alpha, a, b), number=1)
    if output_array:
        arr_out = "==BEGIN DUMP_ARRAYS==\n"
        arr_out += "begin dump: B"
        for i in range(m):
            for j in range(n):
                if ((i * m + j) % 20) == 0:
                    arr_out += "\n"
                arr_out += f"{b[i, j]:.2f} "
        arr_out += "\nend   dump: B\n"
        arr_out += "==END   DUMP_ARRAYS==\n"
        results["array"] = arr_out
    results["perf"] = perf

    if c_test:
        a_ptr = a_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        b_ptr = b_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        perf = timeit(lambda: trmm_c(m, n, alpha, a_ptr, b_ptr), number=1)
        max_difference = 0.0
        for i in range(m):
            for j in range(n):
                max_difference = max(
                    max_difference, abs(b_copy[i, j] - b[i, j])
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
