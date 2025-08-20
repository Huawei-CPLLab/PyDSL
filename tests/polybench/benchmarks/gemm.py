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

MemrefF32IJ = MemRefFactory((DYNAMIC, DYNAMIC), F32)
MemrefF32IK = MemRefFactory((DYNAMIC, DYNAMIC), F32)
MemrefF32KJ = MemRefFactory((DYNAMIC, DYNAMIC), F32)


@compile()
def gemm(
    ni: Index,
    nj: Index,
    nk: Index,
    alpha: F32,
    beta: F32,
    C: MemrefF32IJ,
    A: MemrefF32IK,
    B: MemrefF32KJ,
) -> None:
    b: F32 = 0.0
    for i in arange(ni):
        for j in arange(nj):
            C[i, j] = C[i, j] * beta
        for k in arange(nk):
            for j in arange(nj):
                C[i, j] = C[i, j] + alpha * A[i, k] * B[k, j]


def main(
    current_dataset: str, output_array: bool, c_test: bool, ctest_obj: str
):
    datasets = {
        "ODD_DATASET": (1000, 1, 1200),
        "MINI_DATASET": (20, 25, 30),
        "SMALL_DATASET": (60, 70, 80),
        "MEDIUM_DATASET": (200, 220, 240),
        "LARGE_DATASET": (1000, 1100, 1200),
        "EXTRALARGE_DATASET": (2000, 2300, 2600),
    }

    ni, nj, nk = datasets.get(current_dataset, (60, 70, 80))

    results = {}
    results["array"] = ""
    results["perf"] = -1.0
    results["c_correctness"] = ""
    results["c_perf"] = -1.0
    if c_test:
        lib = ctypes.CDLL(ctest_obj)
        gemm_c = lib.kernel_gemm

        gemm_c.argtypes = [
            ctypes.c_int,  # ni
            ctypes.c_int,  # nj
            ctypes.c_int,  # nk
            ctypes.c_float,  # alpha
            ctypes.c_float,  # beta
            ctypes.POINTER(ctypes.c_float),  # C
            ctypes.POINTER(ctypes.c_float),  # A
            ctypes.POINTER(ctypes.c_float),  # B
        ]

    a = np.zeros((ni, nk)).astype(np.float32)
    b = np.zeros((nk, nj)).astype(np.float32)
    c = np.zeros((ni, nj)).astype(np.float32)
    alpha = 1.5
    beta = 1.2
    # init array
    for i in range(ni):
        for j in range(nj):
            c[i, j] = ((i * j + 1) % ni) / ni
    for i in range(ni):
        for j in range(nk):
            a[i, j] = (i * (j + 1) % nk) / nk
    for i in range(nk):
        for j in range(nj):
            b[i, j] = (i * (j + 2) % nj) / nj

    a_copy = a.copy()
    b_copy = b.copy()
    c_copy = c.copy()

    perf = timeit(lambda: gemm(ni, nj, nk, alpha, beta, c, a, b), number=1)
    if output_array:
        arr_out = "==BEGIN DUMP_ARRAYS==\n"
        arr_out += "begin dump: C"
        for i in range(ni):
            for j in range(nj):
                if ((i * ni + j) % 20) == 0:
                    arr_out += "\n"
                arr_out += f"{c[i, j]:.2f} "
        arr_out += "\nend   dump: C\n"
        arr_out += "==END   DUMP_ARRAYS==\n"
        results["array"] = arr_out
    results["perf"] = perf

    if c_test:
        a_ptr = a_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        b_ptr = b_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        c_ptr = c_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        perf = timeit(
            lambda: gemm_c(ni, nj, nk, alpha, beta, c_ptr, a_ptr, b_ptr),
            number=1,
        )
        max_difference = 0.0
        for i in range(ni):
            for j in range(nj):
                max_difference = max(
                    max_difference, abs(c_copy[i, j] - c[i, j])
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
