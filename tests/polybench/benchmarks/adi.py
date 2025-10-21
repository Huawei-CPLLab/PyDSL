import sys


from pydsl.type import UInt32, Index, F32
from pydsl.memref import MemRefFactory, DYNAMIC
from pydsl.frontend import compile
from pydsl.affine import (
    affine_range as arange,
)
import numpy as np
from timeit import timeit
import ctypes

MemrefF32 = MemRefFactory((DYNAMIC, DYNAMIC), F32)


@compile()
def adi(
    tsteps: Index,
    n: Index,
    U: MemrefF32,
    V: MemrefF32,
    P: MemrefF32,
    Q: MemrefF32,
) -> None:
    b1: F32 = 2.0
    b2: F32 = 1.0
    zero: F32 = 0.0
    dx = b2 / F32(UInt32(n))
    dy = b2 / F32(UInt32(n))
    dt = b2 / F32(UInt32(tsteps))
    mul1 = b1 * dt / (dx * dx)
    mul2 = b2 * dt / (dy * dy)

    a = -mul1 / b1
    b = b2 + mul1
    c = a
    d = -mul2 / b1
    e = b2 + mul2
    f = d
    for t in arange(1, tsteps + 1):
        # Column Sweep
        for i in arange(1, n - 1):
            V[0, i] = b2
            P[i, 0] = zero
            Q[i, 0] = V[0, i]
            for j in arange(1, n - 1):
                P[i, j] = -c / (a * P[i, j - 1] + b)
                Q[i, j] = (
                    -d * U[j, i - 1]
                    + (b2 + b1 * d) * U[j, i]
                    - f * U[j, i + 1]
                    - a * Q[i, j - 1]
                ) / (a * P[i, j - 1] + b)

            V[n - 1, i] = b2
            for j in arange(1, n - 1):
                V[n - j - 1, i] = (
                    P[i, n - j - 1] * V[n - j, i] + Q[i, n - j - 1]
                )

        # Row Sweep
        for i in arange(1, n - 1):
            U[i, 0] = b2
            P[i, 0] = zero
            Q[i, 0] = b2
            for j in arange(1, n - 1):
                P[i, j] = -f / (d * P[i, j - 1] + e)
                Q[i, j] = (
                    -a * V[i - 1, j]
                    + (b2 + b1 * a) * V[i, j]
                    - c * V[i + 1, j]
                    - d * Q[i, j - 1]
                ) / (d * P[i, j - 1] + e)

            U[i, n - 1] = b2
            for j in arange(1, n - 1):
                U[i, n - j - 1] = (
                    P[i, n - j - 1] * U[i, n - j] + Q[i, n - j - 1]
                )


def init_array(n, arr):
    for i in range(n):
        for j in range(n):
            arr[i, j] = (i + n - j) / n


def main(
    current_dataset: str, output_array: bool, c_test: bool, ctest_obj: str
):
    datasets = {
        "MINI_DATASET": (20, 20),
        "SMALL_DATASET": (40, 60),
        "MEDIUM_DATASET": (100, 200),
        "LARGE_DATASET": (500, 1000),
        "EXTRALARGE_DATASET": (1000, 2000),
    }

    tsteps, n = datasets.get(current_dataset, (40, 60))

    results = {}
    results["array"] = ""
    results["perf"] = -1.0
    results["c_correctness"] = ""
    results["c_perf"] = -1.0
    if c_test:
        lib = ctypes.CDLL(ctest_obj)
        adi_c = lib.kernel_adi

        adi_c.argtypes = [
            ctypes.c_int,  # tsteps
            ctypes.c_int,  # n
            ctypes.POINTER(ctypes.c_float),  # u
            ctypes.POINTER(ctypes.c_float),  # v
            ctypes.POINTER(ctypes.c_float),  # p
            ctypes.POINTER(ctypes.c_float),  # q
        ]

    u = np.zeros((n, n)).astype(np.float32)
    v = np.zeros((n, n)).astype(np.float32)
    p = np.zeros((n, n)).astype(np.float32)
    q = np.zeros((n, n)).astype(np.float32)
    # init array
    init_array(n, u)
    init_array(n, v)
    init_array(n, p)
    init_array(n, q)

    u_copy = u.copy()
    v_copy = v.copy()
    p_copy = p.copy()
    q_copy = q.copy()

    perf = timeit(lambda: adi(tsteps, n, u, v, p, q), number=1)
    if output_array:
        arr_out = "==BEGIN DUMP_ARRAYS==\n"
        arr_out += "begin dump: u"
        for i in range(n):
            for j in range(n):
                if (i * n + j) % 20 == 0:
                    arr_out += "\n"
                arr_out += f"{u[i, j]:.2f} "
        arr_out += "\nend   dump: u\n"
        arr_out += "==END   DUMP_ARRAYS==\n"
        results["array"] = arr_out
    results["perf"] = perf

    if c_test:
        u_ptr = u_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        v_ptr = v_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        p_ptr = p_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        q_ptr = q_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        perf = timeit(
            lambda: adi_c(tsteps, n, u_ptr, v_ptr, p_ptr, q_ptr), number=1
        )
        max_difference = 0.0
        for i in range(n):
            for j in range(n):
                max_difference = max(
                    max_difference, abs(u_copy[i, j] - u[i, j])
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
