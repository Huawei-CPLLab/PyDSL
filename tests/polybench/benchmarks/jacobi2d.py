import sys


from pydsl.type import F32, Index
from pydsl.memref import MemRefFactory, DYNAMIC
from pydsl.frontend import compile
from pydsl.affine import affine_range as arange
import numpy as np
from timeit import timeit
import ctypes

MemRefF32 = MemRefFactory((DYNAMIC, DYNAMIC), F32)


@compile()
def jacobi(T: Index, N: Index, a: MemRefF32, b: MemRefF32) -> None:
    for _ in arange(T):
        for i in arange(1, N - 1):
            for j in arange(1, N - 1):
                const: F32 = 0.2
                b[i, j] = (
                    a[i, j]
                    + a[i, j - 1]
                    + a[i, j + 1]
                    + a[i - 1, j]
                    + a[i + 1, j]
                ) * const

        for i in arange(1, N - 1):
            for j in arange(1, N - 1):
                const: F32 = 0.2
                a[i, j] = (
                    b[i, j]
                    + b[i, j - 1]
                    + b[i, j + 1]
                    + b[i - 1, j]
                    + b[i + 1, j]
                ) * const


def main(
    current_dataset: str, output_array: bool, c_test: bool, ctest_obj: str
):
    datasets = {
        "MINI_DATASET": (20, 30),
        "SMALL_DATASET": (40, 90),
        "MEDIUM_DATASET": (100, 250),
        "LARGE_DATASET": (500, 1300),
        "EXTRALARGE_DATASET": (1000, 2800),
    }

    tsteps, n = datasets.get(current_dataset, (40, 90))

    results = {}
    results["array"] = ""
    results["perf"] = -1.0
    results["c_correctness"] = ""
    results["c_perf"] = -1.0
    if c_test:
        lib = ctypes.CDLL(ctest_obj)
        jacobi2d_c = lib.kernel_jacobi_2d

        jacobi2d_c.argtypes = [
            ctypes.c_int,  # tsteps
            ctypes.c_int,  # n
            ctypes.POINTER(ctypes.c_float),  # a
            ctypes.POINTER(ctypes.c_float),  # b
        ]

    a = np.zeros((n, n)).astype(np.float32)
    b = np.zeros((n, n)).astype(np.float32)

    # init array
    for i in range(n):
        for j in range(n):
            a[i, j] = (i * (j + 2) + 2) / n
            b[i, j] = (i * (j + 3) + 3) / n

    a_copy = a.copy()
    b_copy = b.copy()

    perf = timeit(lambda: jacobi(tsteps, n, a, b), number=1)
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
        b_ptr = b_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        perf = timeit(lambda: jacobi2d_c(tsteps, n, a_ptr, b_ptr), number=1)
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
