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

MemrefF32_IJ = MemRefFactory((DYNAMIC, DYNAMIC), F32)
MemrefF32_IK = MemRefFactory((DYNAMIC, DYNAMIC), F32)
MemrefF32_KJ = MemRefFactory((DYNAMIC, DYNAMIC), F32)
MemrefF32_JL = MemRefFactory((DYNAMIC, DYNAMIC), F32)
MemrefF32_IL = MemRefFactory((DYNAMIC, DYNAMIC), F32)


@compile()
def Twomm(
    ni: Index,
    nj: Index,
    nk: Index,
    nl: Index,
    alpha: F32,
    beta: F32,
    tmp: MemrefF32_IJ,
    A: MemrefF32_IK,
    B: MemrefF32_KJ,
    C: MemrefF32_JL,
    D_arr: MemrefF32_IL,
) -> None:
    b: F32 = 0.0
    for i in arange(ni):
        for j in arange(nj):
            tmp[i, j] = b
            for k in arange(nk):
                tmp[i, j] = tmp[i, j] + alpha * A[i, k] * B[k, j]
    for i in arange(ni):
        for j in arange(nl):
            D_arr[i, j] = D_arr[i, j] * beta
            for k in arange(nj):
                D_arr[i, j] = D_arr[i, j] + tmp[i, k] * C[k, j]


def main(
    current_dataset: str, output_array: bool, c_test: bool, ctest_obj: str
):
    datasets = {
        "MINI_DATASET": (16, 18, 22, 24),
        "SMALL_DATASET": (40, 50, 70, 80),
        "MEDIUM_DATASET": (180, 190, 210, 220),
        "LARGE_DATASET": (800, 900, 1100, 1200),
        "EXTRALARGE_DATASET": (1600, 1800, 2200, 2400),
    }

    ni, nj, nk, nl = datasets.get(current_dataset, (40, 50, 70, 80))

    results = {}
    results["array"] = ""
    results["perf"] = -1.0
    results["c_correctness"] = ""
    results["c_perf"] = -1.0
    if c_test:
        lib = ctypes.CDLL(ctest_obj)
        Twomm_c = lib.kernel_2mm

        Twomm_c.argtypes = [
            ctypes.c_int,  # ni
            ctypes.c_int,  # nj
            ctypes.c_int,  # nk
            ctypes.c_int,  # nl
            ctypes.c_float,  # alpha
            ctypes.c_float,  # beta
            ctypes.POINTER(ctypes.c_float),  # tmp
            ctypes.POINTER(ctypes.c_float),  # A
            ctypes.POINTER(ctypes.c_float),  # B
            ctypes.POINTER(ctypes.c_float),  # C
            ctypes.POINTER(ctypes.c_float),  # D
        ]

    tmp = np.zeros((ni, nj)).astype(np.float32)
    a = np.zeros((ni, nk)).astype(np.float32)
    b = np.zeros((nk, nj)).astype(np.float32)
    c = np.zeros((nj, nl)).astype(np.float32)
    d = np.zeros((ni, nl)).astype(np.float32)
    # init array
    alpha = 1.5
    beta = 1.2
    for i in range(ni):
        for j in range(nk):
            a[i, j] = ((i * j + 1) % ni) / ni
    for i in range(nk):
        for j in range(nj):
            b[i, j] = (i * (j + 1) % nj) / nj
    for i in range(nj):
        for j in range(nl):
            c[i, j] = ((i * (j + 3) + 1) % nl) / nl
    for i in range(ni):
        for j in range(nl):
            d[i, j] = (i * (j + 2) % nk) / nk

    a_copy = a.copy()
    b_copy = b.copy()
    c_copy = c.copy()
    d_copy = d.copy()

    perf = timeit(
        lambda: Twomm(ni, nj, nk, nl, alpha, beta, tmp, a, b, c, d), number=1
    )
    if output_array:
        arr_out = "==BEGIN DUMP_ARRAYS==\n"
        arr_out += "begin dump: D"
        for i in range(ni):
            for j in range(nl):
                if (i * ni + j) % 20 == 0:
                    arr_out += "\n"
                arr_out += f"{d[i, j]:.2f} "
        arr_out += "\nend   dump: D\n"
        arr_out += "==END   DUMP_ARRAYS==\n"
        results["array"] = arr_out
    results["perf"] = perf

    if c_test:
        a_ptr = a_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        b_ptr = b_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        c_ptr = c_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        d_ptr = d_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        tmp_ptr = tmp.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        perf = timeit(
            lambda: Twomm_c(
                ni,
                nj,
                nk,
                nl,
                alpha,
                beta,
                tmp_ptr,
                a_ptr,
                b_ptr,
                c_ptr,
                d_ptr,
            ),
            number=1,
        )
        max_difference = 0.0
        for i in range(ni):
            for j in range(nl):
                max_difference = max(
                    max_difference, abs(d_copy[i, j] - d[i, j])
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
