import numpy as np
import math

from pydsl.arith import min
from pydsl.frontend import template
from pydsl.tensor import Tensor
from pydsl.type import F32, F64, Index, SInt8, UInt64

from helper import failed_from, run


def test_basic_template():
    @template()
    def calc[T](a: T) -> T:
        return a * a

    assert math.isclose(calc[F32](3.0), 9.0)
    assert calc[SInt8](4) == 16
    assert calc[UInt64](9) == 81
    assert math.isclose(calc[F32](1.5), 2.25)
    assert calc[SInt8](-4) == 16
    assert calc[UInt64](17) == 289


def test_no_params():
    @template()
    def calc(a: UInt64) -> UInt64:
        return a + a

    assert calc(3) == 6


def test_template_tensor():
    @template()
    def calc[T, N, M](mat: Tensor[T, N, M]) -> Tensor[T, N, M]:
        n = Index(N)
        m = Index(M)
        mat[n - 1, m - 1] = 1
        return mat

    arr = calc[F32, 10, 5](np.zeros((10, 5), dtype=np.float32))
    assert math.isclose(arr[9][4], 1.0)
    arr = calc[F64, 1, 40](np.zeros((1, 40), dtype=np.float64))
    assert math.isclose(arr[0][39], 1.0)


def test_call_macro():
    @template()
    def calc[T](a: T, b: T) -> T:
        return min(a, b)

    assert calc[SInt8](3, 4) == 3
    assert calc[Index](420, 69) == 69


def test_type_inference():
    @template()
    def calc[T](a: T) -> T:
        return a

    with failed_from(NotImplementedError):
        assert calc(3) == 3


if __name__ == "__main__":
    run(test_basic_template)
    run(test_no_params)
    run(test_template_tensor)
    run(test_call_macro)
    run(test_type_inference)
