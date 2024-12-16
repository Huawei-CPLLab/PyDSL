from ctypes import Array, c_bool, c_int32, c_void_p

from pydsl.frontend import CTypeTree_from_Structure, CTypeTree_to_Structure


def recursive_eq(a, b) -> bool:
    """
    TESTING NOTES: this does not support float anywhere nor any explicit ctype
    definition such as c_void_p(0). This is because their equalities are very
    difficult to test against. For example, c_void_p(None) != None and floats
    often comes with small floating-point error.

    Keep your CTypeTree limited to elements whose equalities are easy to check,
    such as tuples, integers, bools, None, or ctypes.Array containing the
    aforementioned types.
    """
    if (not isinstance(a, tuple)) or (not isinstance(b, tuple)):
        if isinstance(a, Array) and isinstance(b, Array):
            return [*a] == [*b]

        return a == b

    if len(a) != len(b):
        return False

    return all([recursive_eq(ai, bi) for ai, bi in zip(a, b, strict=False)])


def test_identity_of_CTypeTree_to_and_from():
    ct = (
        (
            c_bool,
            (c_void_p,),
            (c_int32 * 2, (c_void_p,), (c_bool, (c_bool,)), ()),
        ),
    )

    c1 = ((False, (None,), ((c_int32 * 2)(5, 8), (5,), (True, (False,)), ())),)

    s = CTypeTree_to_Structure(ct, c1)
    c2 = CTypeTree_from_Structure(ct, s)
    assert recursive_eq(c1, c2)
