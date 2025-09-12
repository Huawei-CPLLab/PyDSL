import ast
import typing
from pydsl.attribute import Attribute
from dataclasses import dataclass
import mlir.ir as mlir
from pydsl.macro import CallMacro, MethodType, Uncompiled
from pydsl.protocols import ToMLIRBase


class _NonNull(Attribute):
    @property
    def attr_name(self):
        return "llvm.nonnull"

    def lower(self):
        return (mlir.UnitAttr.get(),)


nonnull = _NonNull()


@dataclass
class align(Attribute):
    value: int

    @CallMacro.generate(method_type=MethodType.CLASS_ONLY)
    def on_Call(
        visitor: ToMLIRBase, cls: type[typing.Self], rep: Uncompiled
    ) -> typing.Any:
        match rep:
            case ast.Constant():
                return cls(rep.value)
            case _:
                return cls(visitor.visit(rep))

    @property
    def attr_name(self):
        return "llvm.align"

    def lower(self):
        i64 = mlir.IntegerType.get_signless(64)
        return (mlir.IntegerAttr.get(i64, self.value),)
