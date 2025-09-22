from pydsl.affine import affine_range as arange
from pydsl.frontend import CTarget, compile
from pydsl.memref import DYNAMIC, MemRef
from pydsl.transform import decorate_next, tile, match_tag as match, tag
from pydsl.type import F32, AnyOp, Index
from helper import failed_from, run

# TODO I don't know what's going on here. Does CTarget support
# transform.validator now?


# def test_reject_unsupported_dialect():
#     """
#     In this test, CTarget is assumed to reject the transform.validator dialect
#     which is used by the tile macro.

#     Rejection of an unsupported dialect should be a ValueError.
#     """

#     with pytest.raises(ValueError):

#         def transform_seq(targ: AnyOp):
#             _ = tile(match(targ, "tile1"), [32, 32], 4)

#         @compile(transform_seq=transform_seq, target_class=CTarget)
#         def truncated_gemver(
#             n: Index,
#             alpha: F32,
#             beta: F32,
#             A: MemRef[F32, DYNAMIC, DYNAMIC],
#             u1: MemRef[F32, DYNAMIC],
#             v1: MemRef[F32, DYNAMIC],
#             u2: MemRef[F32, DYNAMIC],
#             v2: MemRef[F32, DYNAMIC],
#         ) -> None:
#             decorate_next(tag("tile1"))
#             for i in arange(n):
#                 for j in arange(n):
#                     A[i, j] = A[i, j] + u1[i] * v1[j] + u2[i] * v2[j]

if __name__ == "__main__":
    # run(test_reject_unsupported_dialect)
    pass
