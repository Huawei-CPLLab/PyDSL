from pydsl.affine import affine_map as am
from pydsl.affine import affine_range as arange
from pydsl.affine import dimension as D
from pydsl.affine import symbol as S
from pydsl.frontend import PolyCTarget, compile
from pydsl.memref import DYNAMIC, MemRefFactory
from pydsl.scf import range
from pydsl.transform import (
    distribute,
    fuse,
    fuse_into,
    get_loop,
    int_attr,
    parallel,
    recursively,
    reorder,
    tag,
    tile,
)
from pydsl.transform import match_tag as match
from pydsl.type import F32, F64, AnyOp, Index, Tuple
from helper import run

MemRefRank3F32 = MemRefFactory((DYNAMIC, DYNAMIC, DYNAMIC), F32)
MemRefRank2F32 = MemRefFactory((DYNAMIC, DYNAMIC), F32)
MemRefRank2F64 = MemRefFactory((DYNAMIC, DYNAMIC), F64)

# TODO: Due to the difficulty of testing algorithms, we will just see if they
# compile without any error. This is enough to catch a lot of regression bugs.

# TODO: Commented out tests require fuse or fuse_into, which seem to be not
# implemented right now. Re-add once those are implemented.


def heat_transform_seq(targ: AnyOp):
    fuse(match(targ, "fuse_1"), match(targ, "fuse_2"), 3)
    tile(match(targ, "tile"), [16, 8, 8, 8], 8)


def explicit_affine_heat(
    tsteps: Index, n: Index, A: MemRefRank3F32, B: MemRefRank3F32
):
    a: F32 = 2.0
    b: F32 = 0.125
    """@tag("tile")"""
    for t in arange(S(tsteps)):
        """@tag("fuse_1")"""
        for i in arange(1, S(n) - 1):
            for j in arange(1, S(n) - 1):
                for k in arange(1, S(n) - 1):
                    B[am(D(i), D(j), D(k))] = (
                        b
                        * (
                            A[am(D(i) + 1, D(j), D(k))]
                            - a * A[am(D(i), D(j), D(k))]
                            + A[am(D(i) - 1, D(j), D(k))]
                        )
                        + b
                        * (
                            A[am(D(i), D(j) + 1, D(k))]
                            - a * A[am(D(i), D(j), D(k))]
                            + A[am(D(i), D(j) - 1, D(k))]
                        )
                        + b
                        * (
                            A[am(D(i), D(j), D(k) + 1)]
                            - a * A[am(D(i), D(j), D(k))]
                            + A[am(D(i), D(j), D(k) - 1)]
                        )
                        + A[am(D(i), D(j), D(k))]
                    )

        """@tag("fuse_2")"""
        for i in arange(1, S(n) - 1):
            for j in arange(1, S(n) - 1):
                for k in arange(1, S(n) - 1):
                    A[am(D(i), D(j), D(k))] = (
                        b
                        * (
                            B[am(D(i) + 1, D(j), D(k))]
                            - a * B[am(D(i), D(j), D(k))]
                            + B[am(D(i) - 1, D(j), D(k))]
                        )
                        + b
                        * (
                            B[am(D(i), D(j) + 1, D(k))]
                            - a * B[am(D(i), D(j), D(k))]
                            + B[am(D(i), D(j) - 1, D(k))]
                        )
                        + b
                        * (
                            B[am(D(i), D(j), D(k) + 1)]
                            - a * B[am(D(i), D(j), D(k))]
                            + B[am(D(i), D(j), D(k) - 1)]
                        )
                        + B[am(D(i), D(j), D(k))]
                    )


# def test_compile_explicit_affine_heat():
#     compile(
#         globals(),
#         transform_seq=heat_transform_seq,
#         auto_build=False,
#     )(explicit_affine_heat)


def implicit_affine_heat(
    tsteps: Index, n: Index, A: MemRefRank3F32, B: MemRefRank3F32
) -> Tuple[MemRefRank3F32, MemRefRank3F32]:
    a: F32 = 2.0
    b: F32 = 0.125
    """@tag("tile")"""
    for _ in arange(tsteps):
        """@tag("fuse_1")"""
        for i in arange(1, n - 1):
            for j in arange(1, n - 1):
                for k in arange(1, n - 1):
                    B[i, j, k] = A[i, j, k] + b * (
                        A[i + 1, j, k]
                        - a * A[i, j, k]
                        + A[i - 1, j, k]
                        + A[i, j + 1, k]
                        - a * A[i, j, k]
                        + A[i, j - 1, k]
                        + A[i, j, k + 1]
                        - a * A[i, j, k]
                        + A[i, j, k - 1]
                    )

        """@tag("fuse_2")"""
        for i in arange(1, n - 1):
            for j in arange(1, n - 1):
                for k in arange(1, n - 1):
                    A[i, j, k] = B[i, j, k] + b * (
                        B[i + 1, j, k]
                        - a * B[i, j, k]
                        + B[i - 1, j, k]
                        + B[i, j + 1, k]
                        - a * B[i, j, k]
                        + B[i, j - 1, k]
                        + B[i, j, k + 1]
                        - a * B[i, j, k]
                        + B[i, j, k - 1]
                    )

    return A, B


# def test_compile_implicit_affine_heat():
#     compile(
#         globals(),
#         transform_seq=heat_transform_seq,
#         auto_build=False,
#     )(implicit_affine_heat)


def scf_jacobi(T: Index, N: Index, a: MemRefRank2F32, b: MemRefRank2F32):
    c1: Index = 1

    for _ in range(T):
        for i in range(N - c1):
            for j in range(N - c1):
                const: F32 = 0.2
                b[i, j] = (
                    a[i, j]
                    + a[i, j - c1]
                    + a[i, j + c1]
                    + a[i - c1, j]
                    + a[i + c1, j]
                ) * const

        for i in range(c1, N - c1):
            for j in range(c1, N - c1):
                const: F32 = 0.2
                a[i, j] = (
                    b[i, j]
                    + b[i, j - c1]
                    + b[i, j + c1]
                    + b[i - c1, j]
                    + b[i + c1, j]
                ) * const


def test_compile_scf_jacobi():
    compile(
        globals(),
        auto_build=False,
    )(scf_jacobi)


def affine_jacobi(T: Index, N: Index, a: MemRefRank2F32, b: MemRefRank2F32):
    for _ in arange(S(T)):
        for i in arange(1, S(N) - 1):
            for j in arange(1, S(N) - 1):
                const: F32 = 0.2
                b[am(D(i), D(j))] = (
                    a[am(D(i), D(j))]
                    + a[am(D(i), D(j) - 1)]
                    + a[am(D(i), D(j) + 1)]
                    + a[am(D(i) - 1, D(j))]
                    + a[am(D(i) + 1, D(j))]
                ) * const

        for i in arange(1, S(N) - 1):
            for j in arange(1, S(N) - 1):
                const: F32 = 0.2
                a[am(D(i), D(j))] = (
                    b[am(D(i), D(j))]
                    + b[am(D(i), D(j) - 1)]
                    + b[am(D(i), D(j) + 1)]
                    + b[am(D(i) - 1, D(j))]
                    + b[am(D(i) + 1, D(j))]
                ) * const


def test_compile_affine_jacobi():
    compile(
        globals(),
        auto_build=False,
    )(affine_jacobi)


def correlation(
    v1: Index, v0: Index, arg1: MemRefRank2F32, arg2: MemRefRank2F32
) -> F32:
    a: F32 = 1.0
    b: F32 = 0.0
    """@tag("tile_and_distribute")"""
    for arg3 in arange(S(v1) - 1):
        """@int_attr("set", 0)"""
        arg2[am(D(arg3), D(arg3))] = a
        for arg4 in arange(D(arg3) + 1, S(v1)):
            """@int_attr("set", 1)"""
            arg2[am(D(arg4), D(arg3))] = b
            for arg5 in arange(S(v0)):
                """@recursively(lambda x: int_attr(x, "set", 2))"""
                arg2[am(D(arg4), D(arg3))] = arg2[am(D(arg4), D(arg3))] + (
                    arg1[am(D(arg5), D(arg3))] * arg1[am(D(arg5), D(arg4))]
                )

            with recursively(lambda x: int_attr(x, "set", 3)):
                arg2[am(D(arg3), D(arg4))] = arg2[am(D(arg4), D(arg3))]

    arg1[am(S(v1) - 1, S(v1) - 1)] = a

    return a


def test_compile_correlation():
    compile(
        globals(),
        auto_build=False,
    )(correlation)


def lu_transform_seq(targ: AnyOp):
    fuse_into(
        fuse_into(
            fuse_into(
                fuse_into(match(targ, "fuse_1"), match(targ, "fuse_target1")),
                match(targ, "fuse_target2"),
            ),
            match(targ, "fuse_target3"),
        ),
        match(targ, "fuse_target4"),
    )

    fuse(match(targ, "fuse_4"), match(targ, "fuse_3"), 2)

    tile(match(targ, "tile"), [32, 32, 32], 6)


def lu(v0: Index, arg1: MemRefRank2F64):
    """@tag("tile")"""
    for arg2 in arange(S(v0)):
        """@tag("fuse_4")"""
        for arg3 in arange(D(arg2)):
            """@tag("fuse_1")"""
            for arg4 in arange(D(arg3)):
                arg1[am(D(arg2), D(arg3))] = arg1[am(D(arg2), D(arg3))] - (
                    arg1[am(D(arg2), D(arg4))] * arg1[am(D(arg4), D(arg3))]
                )

            """@tag("fuse_target1")"""
            v1 = arg1[am(D(arg3), D(arg3))]

            """@tag("fuse_target2")"""
            v2 = arg1[am(D(arg2), D(arg3))]

            """@tag("fuse_target3")"""
            v3 = v2 / v1

            """@tag("fuse_target4")"""
            arg1[am(D(arg2), D(arg3))] = v3

        """@tag("fuse_3")"""
        for arg3 in arange(D(arg2), S(v0)):
            for arg4 in arange(D(arg2)):
                arg1[am(D(arg2), D(arg3))] = arg1[am(D(arg2), D(arg3))] - (
                    arg1[am(D(arg2), D(arg4))] * arg1[am(D(arg4), D(arg3))]
                )


# def test_compile_lu():
#     compile(
#         globals(),
#         transform_seq=lu_transform_seq,
#         auto_build=False,
#     )(lu)


if __name__ == "__main__":
    # run(test_compile_explicit_affine_heat)
    # run(test_compile_implicit_affine_heat)
    run(test_compile_scf_jacobi)
    run(test_compile_affine_jacobi)
    run(test_compile_correlation)
    # run(test_compile_lu)
