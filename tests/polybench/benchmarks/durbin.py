import sys


from pydsl.type import Index, F32
from pydsl.memref import MemRefFactory, DYNAMIC
from pydsl.frontend import compile
from pydsl.affine import affine_range as arange
import numpy as np
from timeit import timeit
import ctypes


MemrefF321D = MemRefFactory((DYNAMIC,), F32)
MemrefF321 = MemRefFactory((1,), F32)


@compile()
def durbin(
    n: Index,
    r: MemrefF321D,
    y: MemrefF321D,
    z: MemrefF321D,
    alpha_mem: MemrefF321,
    beta_mem: MemrefF321,
    sum: MemrefF321,
) -> None:
    a: F32 = 0.5
    b: Index = 0
    c: F32 = 0.7
    zero: F32 = 0.0
    y[0] = zero - r[0]
    beta_mem[0] = F32(1)
    alpha_mem[0] = zero - r[0]

    for k in arange(1, n):
        beta_mem[0] = (F32(1) - alpha_mem[0] * alpha_mem[0]) * beta_mem[0]
        sum[0] = zero
        for i in arange(k):
            sum[0] = sum[0] + r[k - i - 1] * y[i]
        alpha_mem[0] = zero - (r[k] + sum[0]) / beta_mem[0]
        for i in arange(k):
            z[i] = y[i] + alpha_mem[0] * y[k - i - 1]
        for i in arange(k):
            y[i] = z[i]

        y[k] = alpha_mem[0]


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

    results = {}
    results["array"] = ""
    results["perf"] = -1.0
    results["c_correctness"] = ""
    results["c_perf"] = -1.0
    if c_test:
        lib = ctypes.CDLL(ctest_obj)
        durbin_c = lib.kernel_durbin

        durbin_c.argtypes = [
            ctypes.c_int,  # n
            ctypes.POINTER(ctypes.c_float),  # r
            ctypes.POINTER(ctypes.c_float),  # y
        ]

    n = datasets.get(current_dataset, 120)

    r = np.zeros(n).astype(np.float32)
    y = np.zeros(n).astype(np.float32)
    z = np.zeros(n).astype(np.float32)
    alpha = np.zeros(1).astype(np.float32)
    beta = np.zeros(1).astype(np.float32)
    sum = np.zeros(1).astype(np.float32)

    # init array
    for i in range(n):
        r[i] = n + 1 - i

    r_copy = r.copy()
    y_copy = y.copy()

    perf = timeit(lambda: durbin(n, r, y, z, alpha, beta, sum), number=1)
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
        r_ptr = r_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        y_ptr = y_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        perf = timeit(lambda: durbin_c(n, r_ptr, y_ptr), number=1)
        max_difference = 0.0
        for i in range(n):
            max_difference = max(max_difference, abs(y_copy[i] - y[i]))
        results["c_perf"] = perf
        if max_difference == 0.0:
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
