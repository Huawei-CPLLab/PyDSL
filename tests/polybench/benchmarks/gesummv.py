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
def gesummv(
    n: Index,
    alpha: F32,
    beta: F32,
    A: MemrefF322D,
    B: MemrefF322D,
    tmp: MemrefF321D,
    x: MemrefF321D,
    y: MemrefF321D,
) -> None:
    b: F32 = 0.0
    for i in arange(n):
        tmp[i] = b
        y[i] = b
        for j in arange(n):
            tmp[i] = A[i, j] * x[j] + tmp[i]
            y[i] = B[i, j] * x[j] + y[i]
        y[i] = alpha * tmp[i] + beta * y[i]


def main(
    current_dataset: str, output_array: bool, c_test: bool, ctest_obj: str
):
    datasets = {
        "MINI_DATASET": 30,
        "SMALL_DATASET": 90,
        "MEDIUM_DATASET": 250,
        "LARGE_DATASET": 1300,
        "EXTRALARGE_DATASET": 2800,
    }

    n = datasets.get(current_dataset, 90)

    results = {}
    results["array"] = ""
    results["perf"] = -1.0
    results["c_correctness"] = ""
    results["c_perf"] = -1.0
    if c_test:
        lib = ctypes.CDLL(ctest_obj)
        gesummv_c = lib.kernel_gesummv

        gesummv_c.argtypes = [
            ctypes.c_int,  # n
            ctypes.c_float,  # alpha
            ctypes.c_float,  # beta
            ctypes.POINTER(ctypes.c_float),  # a
            ctypes.POINTER(ctypes.c_float),  # b
            ctypes.POINTER(ctypes.c_float),  # tmp
            ctypes.POINTER(ctypes.c_float),  # x
            ctypes.POINTER(ctypes.c_float),  # y
        ]

    a = np.zeros((n, n)).astype(np.float32)
    b = np.zeros((n, n)).astype(np.float32)
    tmp = np.zeros(n).astype(np.float32)
    x = np.zeros(n).astype(np.float32)
    y = np.zeros(n).astype(np.float32)
    alpha = 1.5
    beta = 1.2

    # init array
    for i in range(n):
        x[i] = (i % n) / n
        for j in range(n):
            a[i, j] = ((i * j + 1) % n) / n
            b[i, j] = ((i * j + 2) % n) / n

    a_copy = a.copy()
    b_copy = b.copy()
    tmp_copy = tmp.copy()
    x_copy = x.copy()
    y_copy = y.copy()

    perf = timeit(lambda: gesummv(n, alpha, beta, a, b, tmp, x, y), number=1)
    if output_array:
        arr_out = "==BEGIN DUMP_ARRAYS==\n"
        arr_out += "begin dump: y"
        for i in range(n):
            if ((i) % 20) == 0:
                arr_out += "\n"
            arr_out += f"{y[i]:.2f} "
        arr_out += "\nend   dump: y\n"
        arr_out += "==END   DUMP_ARRAYS==\n"
        results["array"] = arr_out
    results["perf"] = perf

    if c_test:
        a_ptr = a_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        b_ptr = b_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        tmp_ptr = tmp_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        x_ptr = x_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        y_ptr = y_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        perf = timeit(
            lambda: gesummv_c(
                n, alpha, beta, a_ptr, b_ptr, tmp_ptr, x_ptr, y_ptr
            ),
            number=1,
        )
        max_difference = 0.0
        for i in range(n):
            max_difference = max(max_difference, abs(y_copy[i] - y[i]))
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
