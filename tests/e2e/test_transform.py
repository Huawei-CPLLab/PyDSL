"""
Every test in test_transform should perform 2 checks:
- Runtime correctness (if function returns/mutates something)
- Transform ops present and built correctly in emitted MLIR
"""

import numpy as np

from pydsl.affine import affine_range as arange
from pydsl.frontend import compile, PolyCTarget
from pydsl.memref import MemRef
from pydsl.scf import range as srange
from pydsl.transform import (
    cse,
    int_attr,
    distributed_parallel,
    loop_coalesce,
    outline_loop,
    recursively,
    tag,
)
from pydsl.transform import (
    match_tag as match,
)
from pydsl.type import F32, AnyOp, Index, UInt32
from tests.e2e.helper import SequentialTokenConsumer


def test_multiple_recursively_tag():
    @compile(globals())
    def identity():
        with recursively(lambda x: tag(x, "mytag")):
            a: F32 = 0.0
            b: F32 = 1.0

    # No runtime check: function doesn't return/mutate anything

    # Transform op check:
    mlir = identity.emit_mlir()

    assert r"arith.constant {mytag} 0" in mlir
    assert r"arith.constant {mytag} 1" in mlir


def test_multiple_recursively_int_attr():
    @compile(globals())
    def identity():
        with recursively(lambda x: int_attr(x, "myintattr", 3)):
            a: F32 = 0.0
            b: F32 = 1.0

    # No runtime check: function doesn't return/mutate anything

    # Transform op check:
    mlir = identity.emit_mlir()

    assert r"arith.constant {myintattr = 3 : index} 0" in mlir
    assert r"arith.constant {myintattr = 3 : index} 1" in mlir


def transform_seq_test_cse_then_coalesce(targ: AnyOp):
    cse(match(targ, "coalesce_func"))
    loop_coalesce(match(targ, "coalesce_loop"))


def test_cse_then_coalesce():
    @compile(
        globals(),
        transform_seq=transform_seq_test_cse_then_coalesce,
        override_fields=False,
    )
    class Module:
        """@tag("coalesce_func")"""

        def f(m: MemRef[UInt32, 4, 4]):
            """@tag("coalesce_loop")"""
            for i in srange(4):
                for j in srange(4):
                    m[i, j] = i + j

    # Runtime check:
    m = np.zeros((4, 4), dtype=np.uint32)
    res = np.fromfunction(lambda i, j: i + j, (4, 4))
    Module.get_module_attr("f")(m)
    assert (m == res).all()

    # Transform op check:
    s = SequentialTokenConsumer(Module.emit_mlir())
    match_func = s.match_single_assignment(
        r"transform.structured.match attributes {coalesce_func}"
    )
    s.match_op(f"transform.apply_cse to %{match_func}")
    match_loop = s.match_single_assignment(
        r"transform.structured.match attributes {coalesce_loop}"
    )
    s.match_single_assignment(f"transform.loop.coalesce %{match_loop}")


def transform_seq_test_distributed_parallel(targ: AnyOp):
    distributed_parallel(match(targ, "distributed_for"), 10)


def test_distributed_parallel():
    ...
    # NotImplemented


def transform_seq_test_outline_loop(targ: AnyOp):
    outline_loop(match(targ, "outlined"), "outlined")


def test_outline_loop():
    @compile(
        globals(),
        transform_seq=transform_seq_test_outline_loop,
    )
    def f(m: MemRef[UInt32, 4, 4]):
        m[1, 1] = 3

        """@tag("outlined")"""
        for i in arange(4):
            for j in arange(4):
                m[i, j] = i + j

        m[2, 2] = 13

    # Runtime check:
    m = np.zeros((4, 4), dtype=np.uint32)
    res = np.fromfunction(lambda i, j: i + j, (4, 4))
    res[2, 2] = 13
    f(m)
    assert (m == res).all()

    # Transform op check:
    s = SequentialTokenConsumer(f.emit_mlir())
    match_func = s.match_single_assignment(
        r"transform.structured.match attributes {outlined}"
    )
    s.match_op(
        r"transform.loop.outline %" + match_func + r' {func_name = "outlined"}'
    )
