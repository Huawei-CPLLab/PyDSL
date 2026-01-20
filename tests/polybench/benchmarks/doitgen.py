import sys


from pydsl.type import Index, F32
from pydsl.memref import MemRefFactory, DYNAMIC
from pydsl.frontend import compile
from pydsl.affine import affine_range as arange
import numpy as np
from timeit import timeit
import ctypes

MemrefF323D = MemRefFactory((DYNAMIC, DYNAMIC, DYNAMIC), F32)
MemrefF322D = MemRefFactory((DYNAMIC, DYNAMIC), F32)
MemrefF321D = MemRefFactory((DYNAMIC,), F32)


@compile()
def doitgen(
    nr: Index,
    nq: Index,
    np: Index,
    A: MemrefF323D,
    C4: MemrefF322D,
    sum: MemrefF321D,
) -> None:
    zero: F32 = 0.0
    for r in arange(nr):
        for q in arange(nq):
            for p in arange(np):
                sum[p] = zero
                for s in arange(np):
                    sum[p] = sum[p] + A[r, q, s] * C4[s, p]
            for p in arange(np):
                A[r, q, p] = sum[p]


def main(
    current_dataset: str, output_array: bool, c_test: bool, ctest_obj: str
):
    datasets = {
        "MINI_DATASET": (8, 10, 12),
        "SMALL_DATASET": (20, 25, 30),
        "MEDIUM_DATASET": (40, 50, 60),
        "LARGE_DATASET": (140, 150, 160),
        "EXTRALARGE_DATASET": (220, 250, 270),
    }

    n_q, n_r, n_p = datasets.get(current_dataset, (20, 25, 30))

    results = {}
    results["array"] = ""
    results["perf"] = -1.0
    results["c_correctness"] = ""
    results["c_perf"] = -1.0
    if c_test:
        lib = ctypes.CDLL(ctest_obj)
        doitgen_c = lib.kernel_doitgen

        doitgen_c.argtypes = [
            ctypes.c_int,  # nr
            ctypes.c_int,  # nq
            ctypes.c_int,  # np
            ctypes.POINTER(ctypes.c_float),  # A
            ctypes.POINTER(ctypes.c_float),  # C4
            ctypes.POINTER(ctypes.c_float),  # sum
        ]

    a = np.zeros((n_r, n_q, n_p)).astype(np.float32)
    sum_var = np.zeros(n_p).astype(np.float32)
    c4 = np.zeros((n_p, n_p)).astype(np.float32)

    # init array
    for i in range(n_r):
        for j in range(n_q):
            for k in range(n_p):
                a[i, j, k] = ((i * j + k) % n_p) / n_p
    for i in range(n_p):
        for j in range(n_p):
            c4[i, j] = (i * j % n_p) / n_p
    a_copy = a.copy()
    sum_var_copy = sum_var.copy()
    c4_copy = c4.copy()

    perf = timeit(lambda: doitgen(n_r, n_q, n_p, a, c4, sum_var), number=1)
    if output_array:
        arr_out = "==BEGIN DUMP_ARRAYS==\n"
        arr_out += "begin dump: a"
        for i in range(n_r):
            for j in range(n_q):
                for k in range(n_p):
                    if ((i * n_q * n_p + j * n_p + k) % 20) == 0:
                        arr_out += "\n"
                    arr_out += f"{a[i, j, k]:.2f} "
        arr_out += "\nend   dump: a\n"
        arr_out += "==END   DUMP_ARRAYS==\n"
        results["array"] = arr_out
    results["perf"] = perf

    if c_test:
        a_ptr = a_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        sum_var_ptr = sum_var_copy.ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        )
        c4_ptr = c4_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        perf = timeit(
            lambda: doitgen_c(n_r, n_q, n_p, a_ptr, c4_ptr, sum_var_ptr),
            number=1,
        )
        max_difference = 0.0
        for i in range(n_r):
            for j in range(n_q):
                for k in range(n_p):
                    max_difference = max(
                        max_difference, abs(a_copy[i, j, k] - a[i, j, k])
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
