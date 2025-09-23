import sys


from pydsl.type import F32, Index, F32
from pydsl.memref import MemRefFactory, DYNAMIC
from pydsl.frontend import compile
from pydsl.math import sqrt
from pydsl.affine import (
    affine_range as arange,
)
import numpy as np
from timeit import timeit
import ctypes

MemrefF32NM = MemRefFactory((DYNAMIC, DYNAMIC), F32)
MemrefF32MM = MemRefFactory((DYNAMIC, DYNAMIC), F32)
MemrefF32M = MemRefFactory((DYNAMIC,), F32)


# m,n,data,corr
@compile()
def correlation(
    m: Index,
    n: Index,
    float_n: F32,
    data: MemrefF32NM,
    corr: MemrefF32MM,
    mean: MemrefF32M,
    stddev: MemrefF32M,
) -> None:
    eps: F32 = 0.1

    for j in arange(m):
        mean[j] = F32(0.0)
        for i in arange(n):
            mean[j] = mean[j] + data[i, j]
        mean[j] = mean[j] / float_n

    for j in arange(m):
        stddev[j] = F32(0.0)
        for i in arange(n):
            stddev[j] = stddev[j] + (data[i, j] - mean[j]) * (
                data[i, j] - mean[j]
            )
        stddev[j] = stddev[j] / float_n
        stddev[j] = sqrt(stddev[j])
        stddev[j] = F32(1.0) if stddev[j] <= eps else stddev[j]

    for i in arange(n):
        for j in arange(m):
            data[i, j] = data[i, j] - mean[j]
            data[i, j] = data[i, j] / (sqrt(float_n) * stddev[j])

    for i in arange(m - 1):
        corr[i, i] = F32(1.0)
        for j in arange(i + 1, m):
            corr[i, j] = F32(0.0)
            for k in arange(n):
                corr[i, j] = corr[i, j] + (data[k, i] * data[k, j])
            corr[j, i] = corr[i, j]

    corr[m - 1, m - 1] = F32(1.0)


def main(
    current_dataset: str, output_array: bool, c_test: bool, ctest_obj: str
):
    datasets = {
        "MINI_DATASET": (28, 32),
        "SMALL_DATASET": (80, 100),
        "MEDIUM_DATASET": (240, 260),
        "LARGE_DATASET": (1200, 1400),
        "EXTRALARGE_DATASET": (2600, 3000),
    }

    m, n = datasets.get(current_dataset, (80, 100))
    results = {}
    results["array"] = ""
    results["perf"] = -1.0
    results["c_correctness"] = ""
    results["c_perf"] = -1.0
    if c_test:
        lib = ctypes.CDLL(ctest_obj)
        correlation_c = lib.kernel_correlation

        correlation_c.argtypes = [
            ctypes.c_int,  # m
            ctypes.c_int,  # n
            ctypes.c_float,  # float_n
            ctypes.POINTER(ctypes.c_float),  # data
            ctypes.POINTER(ctypes.c_float),  # corr
            ctypes.POINTER(ctypes.c_float),  # mean
            ctypes.POINTER(ctypes.c_float),  # stddev
        ]

    float_n = float(n)
    data = np.zeros((n, m)).astype(np.float32)
    corr = np.zeros((m, m)).astype(np.float32)
    mean = np.zeros(m).astype(np.float32)
    stddev = np.zeros(m).astype(np.float32)
    # init array
    for i in range(n):
        for j in range(m):
            data[i][j] = (i * j) / m + i

    data_copy = data.copy()
    corr_copy = corr.copy()
    mean_copy = mean.copy()
    stddev_copy = stddev.copy()

    perf = timeit(
        lambda: correlation(m, n, float_n, data, corr, mean, stddev), number=1
    )
    if output_array:
        arr_out = "==BEGIN DUMP_ARRAYS==\n"
        arr_out += "begin dump: corr"
        for i in range(m):
            for j in range(m):
                if (i * m + j) % 20 == 0:
                    arr_out += "\n"
                arr_out += f"{corr[i, j]:.2f} "
        arr_out += "\nend   dump: corr\n"
        arr_out += "==END   DUMP_ARRAYS==\n"
        results["array"] = arr_out
    results["perf"] = perf

    if c_test:
        data_ptr = data_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        corr_ptr = corr_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        mean_ptr = mean_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        stddev_ptr = stddev_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        perf = timeit(
            lambda: correlation_c(
                m, n, float_n, data_ptr, corr_ptr, mean_ptr, stddev_ptr
            ),
            number=1,
        )
        max_difference = 0.0
        for i in range(m):
            for j in range(m):
                max_difference = max(
                    max_difference, abs(corr_copy[i, j] - corr[i, j])
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
