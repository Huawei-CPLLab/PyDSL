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

MemrefF32 = MemRefFactory((DYNAMIC, DYNAMIC), F32)


@compile()
def seidal(tsteps: Index, n: Index, A: MemrefF32) -> None:
    a: F32 = 9.0
    for t in arange(tsteps):
        for i in arange(1, n - 1):
            for j in arange(1, n - 1):
                A[i, j] = (
                    A[i - 1, j - 1]
                    + A[i - 1, j]
                    + A[i - 1, j + 1]
                    + A[i, j - 1]
                    + A[i, j]
                    + A[i, j + 1]
                    + A[i + 1, j - 1]
                    + A[i + 1, j]
                    + A[i + 1, j + 1]
                ) / a


def main(
    current_dataset: str, output_array: bool, c_test: bool, ctest_obj: str
):
    datasets = {
        "MINI_DATASET": (20, 40),
        "SMALL_DATASET": (40, 120),
        "MEDIUM_DATASET": (100, 400),
        "LARGE_DATASET": (500, 2000),
        "EXTRALARGE_DATASET": (1000, 4000),
    }

    tsteps, n = datasets.get(current_dataset, (40, 120))

    results = {}
    results["array"] = ""
    results["perf"] = -1.0
    results["c_correctness"] = ""
    results["c_perf"] = -1.0
    if c_test:
        lib = ctypes.CDLL(ctest_obj)
        seidal_c = lib.kernel_seidel_2d

        seidal_c.argtypes = [
            ctypes.c_int,  # tsteps
            ctypes.c_int,  # n
            ctypes.POINTER(ctypes.c_float),  # A
        ]

    a = np.zeros((n, n)).astype(np.float32)

    # init array
    for i in range(n):
        for j in range(n):
            a[i, j] = (i * (j + 2) + 2) / n

    a_copy = a.copy()

    perf = timeit(lambda: seidal(tsteps, n, a), number=1)
    if output_array:
        arr_out = "==BEGIN DUMP_ARRAYS==\n"
        arr_out += "begin dump: A"
        for i in range(n):
            for j in range(n):
                if ((i * n + j) % 20) == 0:
                    arr_out += "\n"
                arr_out += f"{a[i, j]:.2f} "
        arr_out += "\nend   dump: A\n"
        arr_out += "==END   DUMP_ARRAYS==\n"
        results["array"] = arr_out
    results["perf"] = perf

    if c_test:
        a_ptr = a_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        perf = timeit(lambda: seidal_c(tsteps, n, a_ptr), number=1)
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
