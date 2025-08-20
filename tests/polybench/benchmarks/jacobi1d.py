import sys


from pydsl.type import F32, Index
from pydsl.memref import MemRefFactory, DYNAMIC
from pydsl.frontend import compile
from pydsl.affine import affine_range as arange
import numpy as np
from timeit import timeit
import ctypes

MemRefF32 = MemRefFactory((DYNAMIC,), F32)


@compile()
def jacobi1d(T: Index, N: Index, A: MemRefF32, B: MemRefF32) -> None:
    cst1: F32 = 0.33333

    for t in arange(T):
        for i in arange(1, N - 1):
            B[i] = cst1 * (A[i - 1] + A[i] + A[i + 1])
        for i in arange(1, N - 1):
            A[i] = cst1 * (B[i - 1] + B[i] + B[i + 1])


def main(
    current_dataset: str, output_array: bool, c_test: bool, ctest_obj: str
):
    datasets = {
        "MINI_DATASET": (20, 30),
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
        jacobi1d_c = lib.kernel_jacobi_1d

        jacobi1d_c.argtypes = [
            ctypes.c_int,  # tsteps
            ctypes.c_int,  # n
            ctypes.POINTER(ctypes.c_float),  # a
            ctypes.POINTER(ctypes.c_float),  # b
        ]

    a = np.zeros(n).astype(np.float32)
    b = np.zeros(n).astype(np.float32)

    # init array
    for i in range(n):
        a[i] = (i + 2) / n
        b[i] = (i + 3) / n

    a_copy = a.copy()
    b_copy = b.copy()

    perf = timeit(lambda: jacobi1d(tsteps, n, a, b), number=1)
    if output_array:
        arr_out = "==BEGIN DUMP_ARRAYS==\n"
        arr_out += "begin dump: A"
        for i in range(n):
            if ((i) % 20) == 0:
                arr_out += "\n"
            arr_out += f"{a[i]:.2f} "
        arr_out += "\nend   dump: A\n"
        arr_out += "==END   DUMP_ARRAYS==\n"
        results["array"] = arr_out
    results["perf"] = perf

    if c_test:
        a_ptr = a_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        b_ptr = b_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        perf = timeit(lambda: jacobi1d_c(tsteps, n, a_ptr, b_ptr), number=1)
        max_difference = 0.0
        for i in range(n):
            max_difference = max(max_difference, abs(a_copy[i] - a[i]))
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
