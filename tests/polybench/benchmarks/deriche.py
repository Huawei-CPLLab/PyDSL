import sys


from pydsl.type import Index, F32
from pydsl.math import exp, pow
from pydsl.memref import MemRef, MemRefFactory, DYNAMIC, alloca
from pydsl.frontend import compile
from pydsl.affine import (
    affine_range as arange,
)
import numpy as np
from timeit import timeit
import ctypes

Memref2DF32 = MemRefFactory((DYNAMIC, DYNAMIC), F32)
Memref1DF32 = MemRefFactory((DYNAMIC,), F32)


# m,n,data,corr
@compile()
def deriche(
    w: Index,
    h: Index,
    alpha: F32,
    imgIn: Memref2DF32,
    imgOut: Memref2DF32,
    y1: Memref2DF32,
    y2: Memref2DF32,
) -> None:
    xm = alloca((1,), F32)
    tm = alloca((1,), F32)
    ym1 = alloca((1,), F32)
    ym2 = alloca((1,), F32)
    xp1 = alloca((1,), F32)
    xp2 = alloca((1,), F32)
    tp1 = alloca((1,), F32)
    tp2 = alloca((1,), F32)
    yp1 = alloca((1,), F32)
    yp2 = alloca((1,), F32)

    cst1: F32 = 1.0
    cst2: F32 = 2.0
    zero: F32 = 0.0
    k = (
        (cst1 - exp(-alpha))
        * (cst1 - exp((-alpha)))
        / (cst1 + cst2 * alpha * exp((-alpha)) - exp(cst2 * alpha))
    )

    a1 = k
    a5 = k
    a2 = k * exp((-alpha)) * (alpha - cst1)
    a6 = k * exp((-alpha)) * (alpha - cst1)
    a3 = k * exp((-alpha)) * (alpha + cst1)
    a7 = k * exp((-alpha)) * (alpha + cst1)
    a4 = (-k) * exp(alpha * (-cst2))
    a8 = (-k) * exp(alpha * (-cst2))

    b1 = pow(cst2, (-alpha))
    b2 = -exp(-alpha * cst2)

    for i in arange(w):
        ym1[0] = zero
        ym2[0] = zero
        xm[0] = zero
        for j in arange(h):
            y1[i, j] = (
                a1 * imgIn[i, j] + a2 * xm[0] + b1 * ym1[0] + b2 * ym2[0]
            )
            xm[0] = imgIn[i, j]
            ym2[0] = ym1[0]
            ym1[0] = y1[i, j]

    for i in arange(w):
        yp1[0] = zero
        yp2[0] = zero
        xp1[0] = zero
        xp2[0] = zero
        for j in arange(h):
            y2[i, h - j - 1] = (
                a3 * xp1[0] + a4 * xp2[0] + b1 * yp1[0] + b2 * yp2[0]
            )
            xp2[0] = xp1[0]
            xp1[0] = imgIn[i, h - j - 1]
            yp2[0] = yp1[0]
            yp1[0] = y2[i, h - j - 1]

    for i in arange(w):
        for j in arange(h):
            imgOut[i, j] = cst1 * (y1[i, j] + y2[i, j])

    for j in arange(h):
        tm[0] = zero
        ym1[0] = zero
        ym2[0] = zero
        for i in arange(w):
            y1[i, j] = (
                a5 * imgOut[i, j] + a6 * tm[0] + b1 * ym1[0] + b2 * ym2[0]
            )
            tm[0] = imgOut[i, j]
            ym2[0] = ym1[0]
            ym1[0] = y1[i, j]

    for j in arange(h):
        tp1[0] = zero
        tp2[0] = zero
        yp1[0] = zero
        yp2[0] = zero
        for i in arange(w):
            y2[w - i - 1, j] = (
                a7 * tp1[0] + a8 * tp2[0] + b1 * yp1[0] + b2 * yp2[0]
            )
            tp2[0] = tp1[0]
            tp1[0] = imgOut[w - i - 1, j]
            yp2[0] = yp1[0]
            yp1[0] = y2[w - i - 1, j]

    for i in arange(w):
        for j in arange(h):
            imgOut[i, j] = cst1 * (y1[i, j] + y2[i, j])


def main(
    current_dataset: str, output_array: bool, c_test: bool, ctest_obj: str
):
    datasets = {
        "MINI_DATASET": (64, 64),
        "SMALL_DATASET": (192, 128),
        "MEDIUM_DATASET": (720, 480),
        "LARGE_DATASET": (4096, 2160),
        "EXTRALARGE_DATASET": (7860, 4320),
    }

    results = {}
    results["array"] = ""
    results["perf"] = -1.0
    results["c_correctness"] = ""
    results["c_perf"] = -1.0
    if c_test:
        lib = ctypes.CDLL(ctest_obj)
        deriche_c = lib.kernel_deriche

        deriche_c.argtypes = [
            ctypes.c_int,  # w
            ctypes.c_int,  # h
            ctypes.c_float,  # alpha
            ctypes.POINTER(ctypes.c_float),  # imgIn
            ctypes.POINTER(ctypes.c_float),  # imgOut
            ctypes.POINTER(ctypes.c_float),  # y1
            ctypes.POINTER(ctypes.c_float),  # y2
        ]

    w, h = datasets.get(current_dataset, (192, 128))

    imgIn = np.zeros((w, h)).astype(np.float32)
    imgOut = np.zeros((w, h)).astype(np.float32)
    y1 = np.zeros((w, h)).astype(np.float32)
    y2 = np.zeros((w, h)).astype(np.float32)

    alpha = np.float32(0.25)
    # init array
    for i in range(w):
        for j in range(h):
            imgIn[i, j] = ((313 * i + 991 * j) % 65536) / 65535.0

    imgIn_copy = imgIn.copy()
    imgOut_copy = imgOut.copy()
    y1_copy = y1.copy()
    y2_copy = y2.copy()

    perf = timeit(
        lambda: deriche(w, h, alpha, imgIn, imgOut, y1, y2),
        number=1,
    )
    if output_array:
        arr_out = "==BEGIN DUMP_ARRAYS==\n"
        arr_out += "begin dump: imgOut"
        for i in range(w):
            for j in range(h):
                if ((i * h + j) % 20) == 0:
                    arr_out += "\n"
                arr_out += f"{imgOut[i, j]:.2f} "
        arr_out += "\nend   dump: imgOut\n"
        arr_out += "==END   DUMP_ARRAYS==\n"
        results["array"] = arr_out
    results["perf"] = perf

    if c_test:
        imgIn_ptr = imgIn_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        imgOut_ptr = imgOut_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        y1_ptr = y1_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        y2_ptr = y2_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        perf = timeit(
            lambda: deriche_c(
                w, h, alpha, imgIn_ptr, imgOut_ptr, y1_ptr, y2_ptr
            ),
            number=1,
        )
        max_difference = 0.0
        for i in range(w):
            for j in range(h):
                max_difference = max(
                    max_difference, abs(imgOut_copy[i, j] - imgOut[i, j])
                )
        results["c_perf"] = perf
        if max_difference < 0.001:
            results["c_correctness"] = "results are correct."
        else:
            print(
                f"results incorrect! Max value difference is {max_difference:2f}"
            )
    return results


if __name__ == "__main__":
    result = main("SMALL_DATASET", True, False, "")
    print(result["array"])
    print(result["perf"])
