import sys


from pydsl.type import Index, F32
from pydsl.memref import MemRefFactory, DYNAMIC
from pydsl.frontend import compile
from pydsl.affine import affine_range as arange
import numpy as np
from timeit import timeit
import ctypes

MemrefF32NM = MemRefFactory((DYNAMIC, DYNAMIC), F32)
MemrefF32N = MemRefFactory((DYNAMIC,), F32)
MemrefF32M = MemRefFactory((DYNAMIC,), F32)


@compile()
def bicg(
    m: Index,
    n: Index,
    A: MemrefF32NM,
    s: MemrefF32M,
    q: MemrefF32N,
    p: MemrefF32M,
    r: MemrefF32N,
) -> None:
    a: F32 = 1.0
    b: F32 = 0.0
    for i in arange(m):
        s[i] = b
    for i in arange(n):
        q[i] = b
        for j in arange(m):
            s[j] = s[j] + r[i] * A[i, j]
            q[i] = q[i] + p[j] * A[i, j]


def main(
    current_dataset: str, output_array: bool, c_test: bool, ctest_obj: str
):
    datasets = {
        "MINI_DATASET": (38, 42),
        "SMALL_DATASET": (116, 124),
        "MEDIUM_DATASET": (390, 410),
        "LARGE_DATASET": (1900, 2100),
        "EXTRALARGE_DATASET": (1800, 2200),
    }

    m, n = datasets.get(current_dataset, (116, 124))
    results = {}
    results["array"] = ""
    results["perf"] = -1.0
    results["c_correctness"] = ""
    results["c_perf"] = -1.0
    if c_test:
        lib = ctypes.CDLL(ctest_obj)
        bicg_c = lib.kernel_bicg

        bicg_c.argtypes = [
            ctypes.c_int,  # m
            ctypes.c_int,  # n
            ctypes.POINTER(ctypes.c_float),  # A
            ctypes.POINTER(ctypes.c_float),  # s
            ctypes.POINTER(ctypes.c_float),  # q
            ctypes.POINTER(ctypes.c_float),  # p
            ctypes.POINTER(ctypes.c_float),  # r
        ]

    a = np.zeros((n, m)).astype(np.float32)
    s = np.zeros((m)).astype(np.float32)
    q = np.zeros((n)).astype(np.float32)
    p = np.zeros((m)).astype(np.float32)
    r = np.zeros((n)).astype(np.float32)

    for i in range(m):
        p[i] = (i % m) / m
    for i in range(n):
        r[i] = (i % n) / n
        for j in range(m):
            a[i, j] = (i * (j + 1) % n) / n

    a_copy = a.copy()
    s_copy = s.copy()
    q_copy = q.copy()
    p_copy = p.copy()
    r_copy = r.copy()

    perf = timeit(lambda: bicg(m, n, a, s, q, p, r), number=1)
    if output_array:
        arr_out = "==BEGIN DUMP_ARRAYS==\n"
        arr_out += "begin dump: s"
        for i in range(m):
            if (i % 20) == 0:
                arr_out += "\n"
            arr_out += f"{s[i]:.2f} "
        arr_out += "\nend   dump: s\n"

        arr_out += "begin dump: q"
        for i in range(n):
            if (i % 20) == 0:
                arr_out += "\n"
            arr_out += f"{q[i]:.2f} "
        arr_out += "\nend   dump: q\n"
        arr_out += "==END   DUMP_ARRAYS==\n"
        results["array"] = arr_out
    results["perf"] = perf

    if c_test:
        a_ptr = a_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        s_ptr = s_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        q_ptr = q_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        p_ptr = p_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        r_ptr = r_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        perf = timeit(
            lambda: bicg_c(m, n, a_ptr, s_ptr, q_ptr, p_ptr, r_ptr), number=1
        )
        max_difference = 0.0
        for i in range(m):
            max_difference = max(max_difference, abs(s_copy[i] - s[i]))
        for i in range(n):
            max_difference = max(max_difference, abs(q_copy[i] - q[i]))
        results["c_perf"] = perf
        if max_difference < 0.001:
            results["c_correctness"] = "results are correct."
        else:
            results["c_coorecntess"] = (
                f"results incorrect! Max value difference is {max_difference:2f}"
            )
    return results


if __name__ == "__main__":
    result = main("SMALL_DATASET", True, False, "")
    print(result["array"])
    print(result["perf"])
