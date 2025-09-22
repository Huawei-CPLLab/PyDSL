from abc import ABC, abstractmethod
import ast
from collections.abc import Callable, Iterable
from functools import reduce
from typing import List, Optional

from mlir.dialects import transform
from mlir.dialects.transform import loop, structured
from mlir.ir import IndexType, IntegerAttr, OpView, UnitAttr

from pydsl.macro import CallMacro, Compiled, Evaluated, Uncompiled
from pydsl.protocols import SubtreeOut, ToMLIRBase, lower_single
from pydsl.type import AnyOp, Tuple, get_operator, supports_operator


class Transform(ABC):
    tag_attribute: str = ""

    @abstractmethod
    def target_op_type() -> str:
        return NotImplemented

    @abstractmethod
    def apply_transform(op) -> None:
        return NotImplemented

    def tag(self, op, i) -> None:
        self.tag_attribute = f"transform{i}"
        op.attributes[self.tag_attribute] = UnitAttr.get()


@CallMacro.generate()
def loop_coalesce(visitor: "ToMLIRBase", op: Compiled[AnyOp]) -> AnyOp:
    return AnyOp(
        loop.LoopCoalesceOp(
            transform.OperationType.get("scf.for"), lower_single(op)
        )
    )


@CallMacro.generate()
def linalg_fuse_into_containing(
    visitor: "ToMLIRBase",
    producer_op: Compiled[AnyOp],
    container_op: Compiled[AnyOp],
) -> AnyOp:
    return AnyOp(
        structured.FuseIntoContainingOp(
            transform.AnyOpType.get(),
            transform.AnyOpType.get(),
            lower_single(producer_op),
            lower_single(container_op),
        )
    )


@CallMacro.generate()
def linalg_tile_with_for(
    visitor: "ToMLIRBase", target: Compiled[AnyOp], tile_sizes: Evaluated[int]
) -> AnyOp:
    return AnyOp(
        structured.TileUsingForOp(
            [transform.AnyOpType.get()] * len(tile_sizes),
            lower_single(target).result,
            sizes=tile_sizes,
        )
    )


@CallMacro.generate()
def linalg_fuse(
    visitor: "ToMLIRBase", target: Compiled[AnyOp], tile_sizes: Evaluated[int]
) -> AnyOp:
    return AnyOp(
        structured.FuseOp(
            transform.AnyOpType.get(),
            [transform.AnyOpType.get()] * len(tile_sizes),
            lower_single(target),
            tile_sizes=tile_sizes,
        )
    )


def attr_setter(attr_name: str, value):
    """
    Returns a function that sets an MLIR operation attribute.

    - attr_name: str, the name of the attribute
    - value: the MLIR attribute value
    """

    def decorator(mlir):
        op = get_operator(mlir)
        op.attributes[attr_name] = value
        return mlir

    return decorator


@CallMacro.generate()
def tag(visitor: ToMLIRBase, attr_name: Evaluated[str]) -> Evaluated[Callable]:
    """
    Tags the `mlir` MLIR operation with a MLIR unit attribute with name
    `attr_name`.

    Arguments:
    - `mlir`: AST. The AST node whose equivalent MLIR Operator is to be tagged
      with the unit attribute
    - `attr_name`: str. The name of the unit attribute
    """
    return attr_setter(attr_name, UnitAttr.get())


@CallMacro.generate()
def int_attr(
    visitor: ToMLIRBase, attr_name: Evaluated[str], value: Evaluated[int]
) -> Evaluated[Callable]:
    return attr_setter(attr_name, IntegerAttr.get(IndexType.get(), value))


@CallMacro.generate(with_callsite=True)
def decorate_next(
    visitor: ToMLIRBase, self: ast.AST, *decorators: List[Evaluated[Callable]]
) -> OpView:
    parent = getattr(self, "parent", None)
    if not isinstance(parent, ast.Expr):
        raise SyntaxError(
            "decorate_next must appear as a free-standing expression."
        )

    next_stmt = getattr(parent, "next_line", None)
    if next_stmt is None:
        raise SyntaxError(
            "decorate_next could not find a valid next statement."
        )

    # Visit the statement and apply decorators
    rst = visitor.visit(next_stmt)
    rst = reduce(lambda op, f: f(op), decorators, rst)

    visitor.skip_next = True
    return rst


@CallMacro.generate()
def match_tag(
    visitor: ToMLIRBase, target: Compiled[AnyOp], attr_name: Evaluated[str]
) -> AnyOp:
    return AnyOp(
        structured.MatchOp(
            transform.AnyOpType.get(),
            lower_single(target),
            op_attrs={attr_name: UnitAttr.get()},
            # ops=['scf.for'] # TODO: assume this is always undefined for now
        )
    )


@CallMacro.generate()
def fuse_into(
    visitor: ToMLIRBase, loop: Compiled[AnyOp], target: Compiled
) -> SubtreeOut:
    raise NotImplementedError()


@CallMacro.generate()
def fuse(
    visitor: ToMLIRBase,
    target1: Compiled[AnyOp],
    target2: Compiled[AnyOp],
    depth: Evaluated,
) -> SubtreeOut:
    raise NotImplementedError()


@CallMacro.generate()
def skew(
    visitor: ToMLIRBase,
    outer: Compiled[AnyOp],
    inner: Compiled[AnyOp],
    amount: Evaluated,
):
    raise NotImplementedError()


@CallMacro.generate()
def blockreorder(
    visitor: ToMLIRBase,
    permutation: Evaluated[list[int]],
    retlen: Evaluated[int],
    *inputs: list[Compiled[AnyOp]],
):
    raise NotImplementedError()


@CallMacro.generate()
def tile(
    visitor: ToMLIRBase,
    target: Compiled[AnyOp],
    tile_sizes: Evaluated[int],
    retlen: Evaluated[int],
) -> SubtreeOut:
    raise NotImplementedError()


@CallMacro.generate()
def reorder(
    visitor: ToMLIRBase, first_loop: Compiled[AnyOp], second_loop: Compiled
) -> SubtreeOut:
    raise NotImplementedError()


@CallMacro.generate()
def unroll(
    visitor: ToMLIRBase, target: Compiled[AnyOp], factor: Evaluated
) -> SubtreeOut:
    raise NotImplementedError()


@CallMacro.generate()
def distribute(
    visitor: ToMLIRBase, target: Compiled[AnyOp], retlen: Evaluated
) -> SubtreeOut:
    raise NotImplementedError()


@CallMacro.generate()
def get_loop(
    visitor: ToMLIRBase, target: Compiled[AnyOp], index: Evaluated
) -> SubtreeOut:
    if hasattr(target, "loops"):
        return target.loops[index]
    return target.operation.results[index]


@CallMacro.generate()
def parallel(
    visitor: ToMLIRBase, target: Compiled[AnyOp], force: Evaluated = False
) -> SubtreeOut:
    raise NotImplementedError()


@CallMacro.generate()
def vectorize(visitor: ToMLIRBase, target: Compiled) -> SubtreeOut:
    raise NotImplementedError()


@CallMacro.generate()
def distributed_parallel(
    visitor: ToMLIRBase,
    target: Compiled[AnyOp],
    nproc: Evaluated[int] = 1,
) -> AnyOp:
    raise NotImplementedError()


@CallMacro.generate()
def cse(visitor: ToMLIRBase, target: Compiled[AnyOp]) -> AnyOp:
    """
    Perform constant sub-expressions elimination using transform.apply_cse
    """
    return AnyOp(
        transform.ApplyCommonSubexpressionEliminationOp(lower_single(target))
    )


@CallMacro.generate()
def recursively(
    visitor: "ToMLIRBase",
    targets: Uncompiled[OpView],
    func: Evaluated[Callable],
) -> list[SubtreeOut]:
    if not isinstance(targets, Iterable):
        targets = [targets]

    # This function filters out all elements that are not operators and pass
    # only operators into func
    def func_with_filter(x):
        return func(x) if supports_operator(x) else x

    return [
        visitor.visit_with_interception(t, func_with_filter) for t in targets
    ]


@CallMacro.generate()
def outline_loop(
    visitor: "ToMLIRBase",
    target: Compiled[AnyOp],
    fn_name: Evaluated[Optional[str]],
) -> Tuple[AnyOp, AnyOp]:
    outline = loop.LoopOutlineOp(
        lower_single(type(target)),
        lower_single(type(target)),
        lower_single(target),
        func_name=fn_name,
    )

    return Tuple.from_values(visitor, *(outline.results))
