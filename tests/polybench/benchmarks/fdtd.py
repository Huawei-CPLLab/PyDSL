import sys


from pydsl.type import Index, F32
from pydsl.memref import MemRefFactory, DYNAMIC
from pydsl.frontend import compile
from pydsl.affine import affine_range as arange
import numpy as np
from timeit import timeit
import ctypes

MemrefF322D = MemRefFactory((DYNAMIC, DYNAMIC), F32)
MemrefF321D = MemRefFactory((DYNAMIC,), F32)


@compile()
def fdtd(
    tmax: Index,
    nx: Index,
    ny: Index,
    ex: MemrefF322D,
    ey: MemrefF322D,
    hz: MemrefF322D,
    fict: MemrefF321D,
) -> None:
    a: F32 = 0.5
    c: F32 = 0.7
    for t in arange(tmax):
        for j in arange(ny):
            ey[0, j] = fict[t]
        for i in arange(1, nx):
            for j in arange(ny):
                ey[i, j] = ey[i, j] - a * (hz[i, j] - hz[i - 1, j])
        for i in arange(nx):
            for j in arange(1, ny):
                ex[i, j] = ex[i, j] - a * (hz[i, j] - hz[i, j - 1])
        for i in arange(nx - 1):
            for j in arange(ny - 1):
                hz[i, j] = hz[i, j] - c * (
                    ex[i, j + 1] - ex[i, j] + ey[i + 1, j] - ey[i, j]
                )


def main(
    current_dataset: str, output_array: bool, c_test: bool, ctest_obj: str
):
    datasets = {
        "MINI_DATASET": (20, 20, 30),
        "SMALL_DATASET": (40, 60, 80),
        "MEDIUM_DATASET": (100, 200, 240),
        "LARGE_DATASET": (500, 1000, 1200),
        "EXTRALARGE_DATASET": (1000, 2000, 2600),
    }

    tmax, nx, ny = datasets.get(current_dataset, (40, 60, 80))

    results = {}
    results["array"] = ""
    results["perf"] = -1.0
    results["c_correctness"] = ""
    results["c_perf"] = -1.0
    if c_test:
        lib = ctypes.CDLL(ctest_obj)
        fdtd_c = lib.kernel_fdtd_2d

        fdtd_c.argtypes = [
            ctypes.c_int,  # tmax
            ctypes.c_int,  # nx
            ctypes.c_int,  # ny
            ctypes.POINTER(ctypes.c_float),  # ex
            ctypes.POINTER(ctypes.c_float),  # ey
            ctypes.POINTER(ctypes.c_float),  # hz
            ctypes.POINTER(ctypes.c_float),  # _fict_
        ]

    ex = np.zeros((nx, ny)).astype(np.float32)
    ey = np.zeros((nx, ny)).astype(np.float32)
    hz = np.zeros((nx, ny)).astype(np.float32)
    fict = np.zeros(tmax).astype(np.float32)

    # init array
    for i in range(tmax):
        fict[i] = i
    for i in range(nx):
        for j in range(ny):
            ex[i, j] = (i * (j + 1)) / nx
            ey[i, j] = (i * (j + 2)) / ny
            hz[i, j] = (i * (j + 3)) / nx
    ex_copy = ex.copy()
    ey_copy = ey.copy()
    hz_copy = hz.copy()
    fict_copy = fict.copy()

    perf = timeit(lambda: fdtd(tmax, nx, ny, ex, ey, hz, fict), number=1)
    if output_array:
        arr_out = "==BEGIN DUMP_ARRAYS==\n"
        arr_out += "begin dump: ex"
        for i in range(nx):
            for j in range(ny):
                if ((i * nx + j) % 20) == 0:
                    arr_out += "\n"
                arr_out += f"{ex[i, j]:.2f} "
        arr_out += "\nend   dump: ex\n"
        arr_out += "==END   DUMP_ARRAYS==\n"
        arr_out += "begin dump: ey"
        for i in range(nx):
            for j in range(ny):
                if ((i * nx + j) % 20) == 0:
                    arr_out += "\n"
                arr_out += f"{ey[i, j]:.2f} "
        arr_out += "\nend   dump: ey\n"
        arr_out += "begin dump: hz"
        for i in range(nx):
            for j in range(ny):
                if ((i * nx + j) % 20) == 0:
                    arr_out += "\n"
                arr_out += f"{hz[i, j]:.2f} "
        arr_out += "\nend   dump: hz\n"
        results["array"] = arr_out
    results["perf"] = perf

    if c_test:
        ex_ptr = ex_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        ey_ptr = ey_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        hz_ptr = hz_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        fict_ptr = fict_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        perf = timeit(
            lambda: fdtd_c(tmax, nx, ny, ex_ptr, ey_ptr, hz_ptr, fict_ptr),
            number=1,
        )
        max_difference = 0.0
        for i in range(nx):
            for j in range(ny):
                max_difference = max(
                    max_difference, abs(ex_copy[i, j] - ex[i, j])
                )
        for i in range(nx):
            for j in range(ny):
                max_difference = max(
                    max_difference, abs(ey_copy[i, j] - ey[i, j])
                )
        for i in range(nx):
            for j in range(ny):
                max_difference = max(
                    max_difference, abs(hz_copy[i, j] - hz[i, j])
                )
        results["c_perf"] = perf
        if (
            max_difference < 0.5
        ):  # some times difference is like 0.4, only happens with clang
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
