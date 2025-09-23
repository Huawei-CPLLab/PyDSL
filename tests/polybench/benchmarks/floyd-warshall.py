import sys


from pydsl.type import Index, SInt32
from pydsl.memref import MemRefFactory, DYNAMIC
from pydsl.frontend import compile
from pydsl.affine import affine_range as arange
import numpy as np
from timeit import timeit
import ctypes

MemrefSInt32 = MemRefFactory((DYNAMIC, DYNAMIC), SInt32)


@compile()
def floyd_warshall(n: Index, path: MemrefSInt32) -> None:
    for k in arange(n):
        for i in arange(n):
            for j in arange(n):
                path[i, j] = (
                    path[i, j]
                    if path[i, j] < path[i, k] + path[k, j]
                    else path[i, k] + path[k, j]
                )


def main(
    current_dataset: str, output_array: bool, c_test: bool, ctest_obj: str
):
    datasets = {
        "MINI_DATASET": 60,
        "SMALL_DATASET": 180,
        "MEDIUM_DATASET": 500,
        "LARGE_DATASET": 2800,
        "EXTRALARGE_DATASET": 5600,
    }

    n = datasets.get(current_dataset, 180)

    results = {}
    results["array"] = ""
    results["perf"] = -1.0
    results["c_correctness"] = ""
    results["c_perf"] = -1.0
    if c_test:
        lib = ctypes.CDLL(ctest_obj)
        floyd_warshall_c = lib.kernel_floyd_warshall

        floyd_warshall_c.argtypes = [
            ctypes.c_int,  # n
            ctypes.POINTER(ctypes.c_float),  # path
        ]

    path = np.zeros((n, n)).astype(np.int32)

    # init array
    for i in range(n):
        for j in range(n):
            path[i, j] = (i * j) % 7 + 1
            if (i + j) % 13 == 0 or (i + j) % 7 == 0 or (i + j) % 11 == 0:
                path[i, j] = 999

    path_copy = path.copy()

    perf = timeit(lambda: floyd_warshall(n, path), number=1)
    if output_array:
        arr_out = "==BEGIN DUMP_ARRAYS==\n"
        arr_out += "begin dump: path"
        for i in range(n):
            for j in range(n):
                if ((i * n + j) % 20) == 0:
                    arr_out += "\n"
                arr_out += f"{path[i, j]:.2f} "
        arr_out += "\nend   dump: path\n"
        arr_out += "==END   DUMP_ARRAYS==\n"
        results["array"] = arr_out
    results["perf"] = perf

    if c_test:
        path_ptr = path_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        perf = timeit(lambda: floyd_warshall_c(n, path_ptr), number=1)
        max_difference = 0.0
        for i in range(n):
            for j in range(n):
                max_difference = max(
                    max_difference, abs(path_copy[i, j] - path[i, j])
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
