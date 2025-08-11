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
MemrefF32_JM = MemRefFactory((DYNAMIC, DYNAMIC), F32)
MemrefF32_ML = MemRefFactory((DYNAMIC, DYNAMIC), F32)
MemrefF32_IL = MemRefFactory((DYNAMIC, DYNAMIC), F32)


@compile()
def Threemm(
    ni: Index,
    nj: Index,
    nk: Index,
    nl: Index,
    nm: Index,
    E: MemrefF32_IJ,
    A: MemrefF32_IK,
    B: MemrefF32_KJ,
    F: MemrefF32_JL,
    C: MemrefF32_JM,
    D_arr: MemrefF32_ML,
    G: MemrefF32_IL,
) -> F32:
    b: F32 = 0.0
    for i in arange(ni):
        for j in arange(nj):
            E[i, j] = b
            for k in arange(nk):
                E[i, j] = E[i, j] + A[i, k] * B[k, j]
    for i in arange(nj):
        for j in arange(nl):
            F[i, j] = b
            for k in arange(nm):
                F[i, j] = F[i, j] + C[i, k] * D_arr[k, j]
    for i in arange(ni):
        for j in arange(nl):
            G[i, j] = b
            for k in arange(nj):
                G[i, j] = G[i, j] + E[i, k] * F[k, j]

    return b


def main(
    current_dataset: str, output_array: bool, c_test: bool, ctest_obj: str
):
    datasets = {
        "MINI_DATASET": (16, 18, 20, 22, 24),
        "SMALL_DATASET": (40, 50, 60, 70, 80),
        "MEDIUM_DATASET": (180, 190, 200, 210, 220),
        "LARGE_DATASET": (800, 900, 1000, 1100, 1200),
        "EXTRALARGE_DATASET": (1600, 1800, 2000, 2200, 2400),
    }

    ni, nj, nk, nl, nm = datasets.get(current_dataset, (40, 50, 60, 70, 80))

    results = {}
    results["array"] = ""
    results["perf"] = -1.0
    results["c_correctness"] = ""
    results["c_perf"] = -1.0
    if c_test:
        lib = ctypes.CDLL(ctest_obj)
        Threemm_c = lib.kernel_3mm

        Threemm_c.argtypes = [
            ctypes.c_int,  # ni
            ctypes.c_int,  # nj
            ctypes.c_int,  # nk
            ctypes.c_int,  # nl
            ctypes.c_int,  # nm
            ctypes.POINTER(ctypes.c_float),  # E
            ctypes.POINTER(ctypes.c_float),  # A
            ctypes.POINTER(ctypes.c_float),  # B
            ctypes.POINTER(ctypes.c_float),  # F
            ctypes.POINTER(ctypes.c_float),  # C
            ctypes.POINTER(ctypes.c_float),  # D
            ctypes.POINTER(ctypes.c_float),  # G
        ]

    e = np.zeros((ni, nj)).astype(np.float32)
    a = np.zeros((ni, nk)).astype(np.float32)
    b = np.zeros((nk, nj)).astype(np.float32)
    f = np.zeros((nj, nl)).astype(np.float32)
    c = np.zeros((nj, nm)).astype(np.float32)
    d = np.zeros((nm, nl)).astype(np.float32)
    g = np.zeros((ni, nl)).astype(np.float32)
    # init array
    alpha = 1.5
    beta = 1.2
    for i in range(ni):
        for j in range(nk):
            a[i, j] = ((i * j + 1) % ni) / (5 * ni)
    for i in range(nk):
        for j in range(nj):
            b[i, j] = ((i * (j + 1) + 2) % nj) / (5 * nj)
    for i in range(nj):
        for j in range(nm):
            c[i, j] = ((i * (j + 3)) % nl) / (5 * nl)
    for i in range(nm):
        for j in range(nl):
            d[i, j] = ((i * (j + 2) + 2) % nk) / (5 * nk)
    e_copy = e.copy()
    a_copy = a.copy()
    b_copy = b.copy()
    f_copy = f.copy()
    c_copy = c.copy()
    d_copy = d.copy()
    g_copy = g.copy()

    perf = timeit(
        lambda: Threemm(ni, nj, nk, nl, nm, e, a, b, f, c, d, g), number=1
    )
    if output_array:
        arr_out = "==BEGIN DUMP_ARRAYS==\n"
        arr_out += "begin dump: G"
        for i in range(ni):
            for j in range(nl):
                if (i * ni + j) % 20 == 0:
                    arr_out += "\n"
                arr_out += f"{g[i, j]:.2f} "
        arr_out += "\nend   dump: G\n"
        arr_out += "==END   DUMP_ARRAYS==\n"
        results["array"] = arr_out
    results["perf"] = perf

    if c_test:
        a_ptr = a_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        b_ptr = b_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        c_ptr = c_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        d_ptr = d_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        e_ptr = e_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        f_ptr = f_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        g_ptr = g_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        perf = timeit(
            lambda: Threemm_c(
                ni,
                nj,
                nk,
                nl,
                nm,
                e_ptr,
                a_ptr,
                b_ptr,
                f_ptr,
                c_ptr,
                d_ptr,
                g_ptr,
            ),
            number=1,
        )
        max_difference = 0.0
        for i in range(ni):
            for j in range(nl):
                max_difference = max(
                    max_difference, abs(g_copy[i, j] - g[i, j])
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
