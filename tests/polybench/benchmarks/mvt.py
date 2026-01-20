import sys


from pydsl.type import Index, F32
from pydsl.memref import MemRefFactory, DYNAMIC
from pydsl.frontend import compile
from pydsl.affine import affine_range as arange
import numpy as np
from timeit import timeit
import ctypes

MemrefF322D = MemRefFactory((DYNAMIC, DYNAMIC), F32)
MemrefF321D = MemRefFactory((DYNAMIC,), F32)


@compile()
def mvt(
    n: Index,
    x1: MemrefF321D,
    x2: MemrefF321D,
    y1: MemrefF321D,
    y2: MemrefF321D,
    A: MemrefF322D,
) -> None:
    for i in arange(n):
        for j in arange(n):
            x1[i] = x1[i] + A[i, j] * y1[j]
    for i in arange(n):
        for j in arange(n):
            x2[i] = x2[i] + A[i, j] * y2[j]


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
        mvt_c = lib.kernel_mvt

        mvt_c.argtypes = [
            ctypes.c_int,  # n
            ctypes.POINTER(ctypes.c_float),  # x1
            ctypes.POINTER(ctypes.c_float),  # x2
            ctypes.POINTER(ctypes.c_float),  # y1
            ctypes.POINTER(ctypes.c_float),  # y2
            ctypes.POINTER(ctypes.c_float),  # a
        ]

    a = np.zeros((n, n)).astype(np.float32)
    x1 = np.zeros(n).astype(np.float32)
    x2 = np.zeros(n).astype(np.float32)
    y1 = np.zeros(n).astype(np.float32)
    y2 = np.zeros(n).astype(np.float32)

    # init array
    for i in range(n):
        x1[i] = (i % n) / n
        x2[i] = ((i + 1) % n) / n
        y1[i] = ((i + 3) % n) / n
        y2[i] = ((i + 4) % n) / n
        for j in range(n):
            a[i, j] = (i * j % n) / n

    a_copy = a.copy()
    x1_copy = x1.copy()
    x2_copy = x2.copy()
    y1_copy = y1.copy()
    y2_copy = y2.copy()

    perf = timeit(lambda: mvt(n, x1, x2, y1, y2, a), number=1)
    if output_array:
        arr_out = "==BEGIN DUMP_ARRAYS==\n"
        arr_out += "begin dump: x1"
        for i in range(n):
            if ((i) % 20) == 0:
                arr_out += "\n"
            arr_out += f"{x1[i]:.2f} "
        arr_out += "\nend   dump: x1\n"
        arr_out += "begin dump: x2"
        for i in range(n):
            if ((i) % 20) == 0:
                arr_out += "\n"
            arr_out += f"{x2[i]:.2f} "
        arr_out += "\nend   dump: x2\n"
        arr_out += "==END   DUMP_ARRAYS==\n"
        results["array"] = arr_out
    results["perf"] = perf

    if c_test:
        a_ptr = a_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        x1_ptr = x1_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        x2_ptr = x2_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        y1_ptr = y1_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        y2_ptr = y2_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        perf = timeit(
            lambda: mvt_c(n, x1_ptr, x2_ptr, y1_ptr, y2_ptr, a_ptr), number=1
        )
        max_difference = 0.0
        for i in range(n):
            max_difference = max(max_difference, abs(x1_copy[i] - x1[i]))
        for i in range(n):
            max_difference = max(max_difference, abs(x2_copy[i] - x2[i]))
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
