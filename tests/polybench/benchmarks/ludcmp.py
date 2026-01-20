import sys


from pydsl.type import Index, F32
from pydsl.memref import MemRef, MemRefFactory, DYNAMIC, alloca
from pydsl.frontend import compile
from pydsl.affine import (
    affine_range as arange,
)
import numpy as np
from timeit import timeit
import ctypes

MemrefF32 = MemRefFactory((DYNAMIC, DYNAMIC), F32)
MemrefF321D = MemRefFactory((DYNAMIC,), F32)


@compile()
def ludcmp(
    n: Index,
    A: MemrefF32,
    b: MemrefF321D,
    x: MemrefF321D,
    y: MemrefF321D,
) -> None:
    w = alloca((1,), F32)
    for i in arange(n):
        for j in arange(i):
            w[0] = A[i, j]
            for k in arange(j):
                w[0] = w[0] - A[i, k] * A[k, j]
            A[i, j] = w[0] / A[j, j]
        for j in arange(i, n):
            w[0] = A[i, j]
            for k in arange(i):
                w[0] = w[0] - A[i, k] * A[k, j]
            A[i, j] = w[0]
    for i in arange(n):
        w[0] = b[i]
        for j in arange(i):
            w[0] = w[0] - A[i, j] * y[j]
        y[i] = w[0]

    for i in arange(n):
        w[0] = y[n - i - 1]
        for j in arange(n - i, n):
            w[0] = w[0] - A[n - i - 1, j] * x[j]
        x[n - i - 1] = w[0] / A[n - i - 1, n - i - 1]


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

    results = {}
    results["array"] = ""
    results["perf"] = -1.0
    results["c_correctness"] = ""
    results["c_perf"] = -1.0
    if c_test:
        lib = ctypes.CDLL(ctest_obj)
        ludcmp_c = lib.kernel_ludcmp

        ludcmp_c.argtypes = [
            ctypes.c_int,  # n
            ctypes.POINTER(ctypes.c_float),  # a
            ctypes.POINTER(ctypes.c_float),  # b
            ctypes.POINTER(ctypes.c_float),  # x
            ctypes.POINTER(ctypes.c_float),  # y
        ]

    n = datasets.get(current_dataset, 120)

    proper_init = n <= 120

    a = np.zeros((n, n)).astype(np.float32)
    b = np.zeros(n).astype(np.float32)
    x = np.zeros(n).astype(np.float32)
    y = np.zeros(n).astype(np.float32)

    a_copy = a.copy()
    b_copy = b.copy()
    x_copy = x.copy()
    y_copy = y.copy()

    # init array
    for i in range(n):
        b[i] = (i + 1) / n / 2 + 4
    for i in range(n):
        for j in range(i + 1):
            a[i, j] = (-j % n) / n
            if (
                (-j % n) / n == 0
            ):  # why is this how it works? This gives the same results as ludcmp.c for the values of a, but it shouldn't.
                a[i, j] = 1
        for j in range(i + 1, n):
            a[i, j] = 0
        a[i, i] = 1

    # make the matrix positive semmi-definite
    if proper_init:
        b_temp = np.zeros((n, n)).astype(np.float32)
        for t in range(n):
            for r in range(n):
                for s in range(n):
                    b_temp[r, s] += a[r, t] * a[s, t]
        for r in range(n):
            for s in range(n):
                a[r, s] = b_temp[r, s]

    perf = timeit(lambda: ludcmp(n, a, b, x, y), number=1)
    if output_array:
        arr_out = "==BEGIN DUMP_ARRAYS==\n"
        arr_out += "begin dump: x"
        for i in range(n):
            if (i) % 20 == 0:
                arr_out += "\n"
            arr_out += f"{x[i]:.2f} "
        arr_out += "\nend   dump: x\n"
        arr_out += "==END   DUMP_ARRAYS==\n"
        results["array"] = arr_out
    results["perf"] = perf

    if c_test:
        a_ptr = a_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        b_ptr = b_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        x_ptr = x_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        y_ptr = y_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        perf = timeit(
            lambda: ludcmp_c(n, a_ptr, b_ptr, x_ptr, y_ptr), number=1
        )
        max_difference = 0.0
        for i in range(n):
            max_difference = max(max_difference, abs(x_copy[i] - x[i]))
        results["c_perf"] = perf
        if max_difference == 0.0:
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
