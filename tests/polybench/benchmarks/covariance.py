import sys


from pydsl.type import Index, F32
from pydsl.memref import MemRefFactory, DYNAMIC
from pydsl.frontend import compile
from pydsl.affine import affine_range as arange
import numpy as np
from timeit import timeit
import ctypes

MemrefF32NM = MemRefFactory((DYNAMIC, DYNAMIC), F32)
MemrefF32MM = MemRefFactory((DYNAMIC, DYNAMIC), F32)
MemrefF32M = MemRefFactory((DYNAMIC,), F32)


@compile()
def covariance(
    m: Index,
    n: Index,
    float_n: F32,
    data: MemrefF32NM,
    cov: MemrefF32MM,
    mean: MemrefF32M,
) -> None:
    a: F32 = 1.0
    b: F32 = 0.0
    for j in arange(m):
        mean[j] = b
        for i in arange(n):
            mean[j] = mean[j] + data[i, j]
        mean[j] = mean[j] / float_n
    for i in arange(n):
        for j in arange(m):
            data[i, j] = data[i, j] - mean[j]
    for i in arange(m):
        for j in arange(i, m):
            cov[i, j] = b
            for k in arange(n):
                cov[i, j] = cov[i, j] + data[k, i] * data[k, j]
            cov[i, j] = cov[i, j] / (float_n - a)
            cov[j, i] = cov[i, j]


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
        covariance_c = lib.kernel_covariance

        covariance_c.argtypes = [
            ctypes.c_int,  # m
            ctypes.c_int,  # n
            ctypes.c_float,  # float_n
            ctypes.POINTER(ctypes.c_float),  # data
            ctypes.POINTER(ctypes.c_float),  # cov
            ctypes.POINTER(ctypes.c_float),  # mean
        ]
    float_n = np.float32(float(n))

    data = np.zeros((n, m)).astype(np.float32)
    cov = np.zeros((m, m)).astype(np.float32)
    mean = np.zeros(m).astype(np.float32)

    # init array
    for i in range(n):
        for j in range(m):
            data[i, j] = i * j / m

    data_copy = data.copy()
    cov_copy = cov.copy()
    mean_copy = mean.copy()

    perf = timeit(lambda: covariance(m, n, float_n, data, cov, mean), number=1)
    if output_array:
        arr_out = "==BEGIN DUMP_ARRAYS==\n"
        arr_out += "begin dump: cov"
        for i in range(m):
            for j in range(m):
                if ((i * n + j) % 20) == 0:
                    arr_out += "\n"
                arr_out += f"{cov[i, j]:.2f} "
        arr_out += "\nend   dump: cov\n"
        arr_out += "==END   DUMP_ARRAYS==\n"
        results["array"] = arr_out
    results["perf"] = perf

    if c_test:
        data_ptr = data_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        cov_ptr = cov_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        mean_ptr = mean_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        perf = timeit(
            lambda: covariance_c(m, n, float_n, data_ptr, cov_ptr, mean_ptr),
            number=1,
        )
        max_difference = 0.0
        for i in range(m):
            for j in range(m):
                max_difference = max(
                    max_difference, abs(cov_copy[i, j] - cov[i, j])
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
