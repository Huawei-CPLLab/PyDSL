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

MemrefF322D = MemRefFactory((DYNAMIC, DYNAMIC), F32)
MemrefF321D = MemRefFactory((DYNAMIC,), F32)


@compile()
def gemver(
    n: Index,
    alpha: F32,
    beta: F32,
    A: MemrefF322D,
    u1: MemrefF321D,
    v1: MemrefF321D,
    u2: MemrefF321D,
    v2: MemrefF321D,
    w: MemrefF321D,
    x: MemrefF321D,
    y: MemrefF321D,
    z: MemrefF321D,
) -> None:
    b: F32 = 0.0
    for i in arange(n):
        for j in arange(n):
            A[i, j] = A[i, j] + u1[i] * v1[j] + u2[i] * v2[j]

    for i in arange(n):
        for j in arange(n):
            x[i] = x[i] + beta * A[j, i] * y[j]

    for i in arange(n):
        x[i] = x[i] + z[i]

    for i in arange(n):
        for j in arange(n):
            w[i] = w[i] + alpha * A[i, j] * x[j]


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
        gemver_c = lib.kernel_gemver

        gemver_c.argtypes = [
            ctypes.c_int,  # n
            ctypes.c_float,  # alpha
            ctypes.c_float,  # beta
            ctypes.POINTER(ctypes.c_float),  # A
            ctypes.POINTER(ctypes.c_float),  # u1
            ctypes.POINTER(ctypes.c_float),  # v1
            ctypes.POINTER(ctypes.c_float),  # u2
            ctypes.POINTER(ctypes.c_float),  # v2
            ctypes.POINTER(ctypes.c_float),  # w
            ctypes.POINTER(ctypes.c_float),  # x
            ctypes.POINTER(ctypes.c_float),  # y
            ctypes.POINTER(ctypes.c_float),  # z
        ]

    a = np.zeros((n, n)).astype(np.float32)
    u1 = np.zeros(n).astype(np.float32)
    v1 = np.zeros(n).astype(np.float32)
    u2 = np.zeros(n).astype(np.float32)
    v2 = np.zeros(n).astype(np.float32)
    w = np.zeros(n).astype(np.float32)
    x = np.zeros(n).astype(np.float32)
    y = np.zeros(n).astype(np.float32)
    z = np.zeros(n).astype(np.float32)
    alpha = 1.5
    beta = 1.2

    # init array
    for i in range(n):
        u1[i] = i
        u2[i] = ((i + 1) / n) / 2
        v1[i] = ((i + 1) / n) / 4
        v2[i] = ((i + 1) / n) / 6
        y[i] = ((i + 1) / n) / 8
        z[i] = ((i + 1) / n) / 9
        x[i] = 0
        w[i] = 0
        for j in range(n):
            a[i, j] = (i * j % n) / n

    a_copy = a.copy()
    u1_copy = u1.copy()
    v1_copy = v1.copy()
    u2_copy = u2.copy()
    v2_copy = v2.copy()
    w_copy = w.copy()
    x_copy = x.copy()
    y_copy = y.copy()
    z_copy = z.copy()

    perf = timeit(
        lambda: gemver(n, alpha, beta, a, u1, v1, u2, v2, w, x, y, z), number=1
    )
    if output_array:
        arr_out = "==BEGIN DUMP_ARRAYS==\n"
        arr_out += "begin dump: w"
        for i in range(n):
            if ((i) % 20) == 0:
                arr_out += "\n"
            arr_out += f"{w[i]:.2f} "
        arr_out += "\nend   dump: w\n"
        arr_out += "==END   DUMP_ARRAYS==\n"
        results["array"] = arr_out
    results["perf"] = perf

    if c_test:
        a_ptr = a_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        u1_ptr = u1_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        v1_ptr = v1_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        u2_ptr = u2_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        v2_ptr = v2_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        w_ptr = w_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        x_ptr = x_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        y_ptr = y_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        z_ptr = z_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        perf = timeit(
            lambda: gemver_c(
                n,
                alpha,
                beta,
                a_ptr,
                u1_ptr,
                v1_ptr,
                u2_ptr,
                v2_ptr,
                w_ptr,
                x_ptr,
                y_ptr,
                z_ptr,
            ),
            number=1,
        )
        max_difference = 0.0
        for i in range(n):
            max_difference = max(max_difference, abs(w_copy[i] - w[i]))
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
