import sys


from pydsl.type import Index, F32
from pydsl.memref import MemRef, MemRefFactory, DYNAMIC, alloca
from pydsl.math import sqrt
from pydsl.frontend import compile
from pydsl.affine import (
    affine_range as arange,
)
import numpy as np
from timeit import timeit
import ctypes

MemrefF32MN = MemRefFactory((DYNAMIC, DYNAMIC), F32)
MemrefF32NN = MemRefFactory((DYNAMIC, DYNAMIC), F32)
MemrefF321D = MemRefFactory((DYNAMIC,), F32)


@compile()
def gramschmidt(
    m: Index,
    n: Index,
    A: MemrefF32MN,
    R: MemrefF32NN,
    Q: MemrefF32MN,
) -> None:
    b: F32 = 0.0
    temp = alloca((1,), F32)
    for k in arange(n):
        temp[0] = b
        for i in arange(m):
            temp[0] = temp[0] + A[i, k] * A[i, k]
        R[k, k] = sqrt(temp[0])
        for i in arange(m):
            Q[i, k] = A[i, k] / R[k, k]
        for j in arange(k + 1, n):
            R[k, j] = b
            for i in arange(m):
                R[k, j] = R[k, j] + Q[i, k] * A[i, j]
            for i in arange(m):
                A[i, j] = A[i, j] - Q[i, k] * R[k, j]


def main(
    current_dataset: str, output_array: bool, c_test: bool, ctest_obj: str
):
    datasets = {
        "MINI_DATASET": (20, 30),
        "SMALL_DATASET": (60, 80),
        "MEDIUM_DATASET": (200, 240),
        "LARGE_DATASET": (1000, 1200),
        "EXTRALARGE_DATASET": (2000, 2600),
    }

    m, n = datasets.get(current_dataset, (60, 80))

    results = {}
    results["array"] = ""
    results["perf"] = -1.0
    results["c_correctness"] = ""
    results["c_perf"] = -1.0
    if c_test:
        lib = ctypes.CDLL(ctest_obj)
        gramschmidt_c = lib.kernel_gramschmidt

        gramschmidt_c.argtypes = [
            ctypes.c_int,  # m
            ctypes.c_int,  # n
            ctypes.POINTER(ctypes.c_float),  # a
            ctypes.POINTER(ctypes.c_float),  # r
            ctypes.POINTER(ctypes.c_float),  # q
        ]

    a = np.zeros((m, n)).astype(np.float32)
    q = np.zeros((m, n)).astype(np.float32)
    r = np.zeros((n, n)).astype(np.float32)

    # init array
    for i in range(m):
        for j in range(n):
            a[i, j] = ((((i * j) % m) / m) * 100) + 10

    a_copy = a.copy()
    q_copy = q.copy()
    r_copy = r.copy()

    perf = timeit(lambda: gramschmidt(m, n, a, r, q), number=1)
    if output_array:
        arr_out = "==BEGIN DUMP_ARRAYS==\n"
        arr_out += "begin dump: R"
        for i in range(n):
            for j in range(n):
                if ((i * n + j) % 20) == 0:
                    arr_out += "\n"
                arr_out += f"{r[i, j]:.2f} "
        arr_out += "\nend   dump: R\n"
        arr_out += "begin dump: Q"
        for i in range(m):
            for j in range(n):
                if ((i * n + j) % 20) == 0:
                    arr_out += "\n"
                arr_out += f"{q[i, j]:.2f} "
        arr_out += "\nend   dump: Q\n"
        arr_out += "==END   DUMP_ARRAYS==\n"
        results["array"] = arr_out
    results["perf"] = perf

    if c_test:
        a_ptr = a_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        r_ptr = r_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        q_ptr = q_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        perf = timeit(
            lambda: gramschmidt_c(m, n, a_ptr, r_ptr, q_ptr), number=1
        )
        max_difference = 0.0
        for i in range(m):
            for j in range(n):
                max_difference = max(
                    max_difference, abs(q_copy[i, j] - q[i, j])
                )
        for i in range(n):
            for j in range(n):
                max_difference = max(
                    max_difference, abs(r_copy[i, j] - r[i, j])
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
