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
def trisolv(n: Index, L: MemrefF322D, x: MemrefF321D, b: MemrefF321D) -> None:
    for i in arange(n):
        x[i] = b[i]
        for j in arange(i):
            x[i] = x[i] - L[i, j] * x[j]
        a = x[i]
        b = L[i, i]
        divf = a / b
        x[i] = divf


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
        trisolv_c = lib.kernel_trisolv

        trisolv_c.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
        ]  # n  # A
    l = np.zeros((n, n)).astype(np.float32)
    x = np.zeros(n).astype(np.float32)
    b = np.zeros(n).astype(np.float32)

    # init array
    for i in range(n):
        x[i] = -999
        b[i] = i
        for j in range(i + 1):
            l[i, j] = (i + n - j + 1) * 2 / n

    l_copy = l.copy()
    x_copy = x.copy()
    b_copy = b.copy()

    perf = timeit(lambda: trisolv(n, l, x, b), number=1)
    if output_array:
        arr_out = "==BEGIN DUMP_ARRAYS==\n"
        arr_out += "begin dump: x"
        for i in range(n):
            arr_out += f"{x[i]:.2f} "
            if ((i) % 20) == 0:
                arr_out += "\n"
        arr_out += "\nend   dump: x\n"
        arr_out += "==END   DUMP_ARRAYS==\n"
        results["array"] = arr_out
    results["perf"] = perf

    if c_test:
        l_ptr = l_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        x_ptr = x_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        b_ptr = b_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        perf = timeit(lambda: trisolv_c(n, l_ptr, x_ptr, b_ptr), number=1)
        max_difference = 0.0
        for i in range(n):
            max_difference = max(max_difference, abs(x_copy[i] - x[i]))
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
