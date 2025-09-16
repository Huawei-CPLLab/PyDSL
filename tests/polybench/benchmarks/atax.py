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

MemrefF32NM = MemRefFactory((DYNAMIC, DYNAMIC), F32)
MemrefF32N = MemRefFactory((DYNAMIC,), F32)
MemrefF32M = MemRefFactory((DYNAMIC,), F32)


@compile()
def atax(
    m: Index,
    n: Index,
    A: MemrefF32NM,
    x: MemrefF32N,
    y: MemrefF32N,
    tmp: MemrefF32M,
) -> None:
    b: F32 = 0.0
    for i in arange(n):
        y[i] = b
    for i in arange(m):
        tmp[i] = b
        for j in arange(n):
            tmp[i] = tmp[i] + A[i, j] * x[j]
        for j in arange(n):
            y[j] = y[j] + A[i, j] * tmp[i]


def main(
    current_dataset: str, output_array: bool, c_test: bool, ctest_obj: str
):
    datasets = {
        "MINI_DATASET": (38, 42),
        "SMALL_DATASET": (116, 124),
        "MEDIUM_DATASET": (390, 410),
        "LARGE_DATASET": (1900, 2100),
        "EXTRALARGE_DATASET": (1800, 2200),
    }

    m, n = datasets.get(current_dataset, (116, 124))
    results = {}
    results["array"] = ""
    results["perf"] = -1.0
    results["c_correctness"] = ""
    results["c_perf"] = -1.0
    if c_test:
        lib = ctypes.CDLL(ctest_obj)
        atax_c = lib.kernel_atax

        atax_c.argtypes = [
            ctypes.c_int,  # m
            ctypes.c_int,  # n
            ctypes.POINTER(ctypes.c_float),  # A
            ctypes.POINTER(ctypes.c_float),  # x
            ctypes.POINTER(ctypes.c_float),  # y
            ctypes.POINTER(ctypes.c_float),  # tmp
        ]

    a = np.zeros((m, n)).astype(np.float32)
    x = np.zeros((n)).astype(np.float32)
    y = np.zeros((n)).astype(np.float32)
    tmp = np.zeros((m)).astype(np.float32)

    for i in range(n):
        x[i] = 1 + (i / n)
    for i in range(m):
        for j in range(n):
            a[i, j] = ((i + j) % n) / (5 * m)

    a_copy = a.copy()
    x_copy = x.copy()
    y_copy = y.copy()
    tmp_copy = tmp.copy()

    perf = timeit(lambda: atax(m, n, a, x, y, tmp), number=1)
    if output_array:
        arr_out = "==BEGIN DUMP_ARRAYS==\n"
        arr_out += "begin dump: U"
        for i in range(n):
            if (i % 20) == 0:
                arr_out += "\n"
            arr_out += f"{y[i]:.2f} "
        arr_out += "\nend   dump: U\n"
        arr_out += "==END   DUMP_ARRAYS==\n"
        results["array"] = arr_out
    results["perf"] = perf

    if c_test:
        a_ptr = a_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        x_ptr = x_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        y_ptr = y_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        tmp_ptr = tmp_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        perf = timeit(
            lambda: atax_c(m, n, a_ptr, x_ptr, y_ptr, tmp_ptr), number=1
        )
        max_difference = 0.0
        for i in range(n):
            max_difference = max(max_difference, abs(y_copy[i] - y[i]))
        results["c_perf"] = perf
        if max_difference < 0.001:
            results["c_correctness"] = "results are correct."
        else:
            results["c_corrrectness"] = (
                f"results incorrect! Max value difference is {max_difference:2f}"
            )
    return results


if __name__ == "__main__":
    result = main("SMALL_DATASET", True, False, "")
    print(result["array"])
    print(result["perf"])
