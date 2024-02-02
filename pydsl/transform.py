from abc import ABC, abstractmethod
from typing import Any, List, TYPE_CHECKING
from mlir._mlir_libs._mlir.ir import OpView

if TYPE_CHECKING:
    # This is for imports for type hinting purposes only and which can result in cyclic imports.
    from pydsl.compiler import ToMLIR

import mlir.ir as mlir
from mlir.ir import *
import mlir.dialects.transform as transform
from mlir.dialects.transform import loop
from mlir.dialects.transform import structured
# DISABLED DUE TO CONFLICT
# from mlir.dialects.transform import affine as taffine

from pydsl.metafunc import CallingMetafunction
from pydsl.type import lower_single


class loop_coalesce(CallingMetafunction):
    def argtypes() -> List[CallingMetafunction.ArgType]:
        return [tag.ArgType.MLIR]
    
    def _on_Call(visitor: "ToMLIR", args: List[Any]) -> OpView:
        target = args[0]
        return loop.LoopCoalesceOp(transform.OperationType.get("scf.for"), target)


class tag(CallingMetafunction):
    """
    Tags the `tree` MLIR operation with a MLIR unit attribute with name `name`

    Arguments:
    - `tree`: AST. The AST node whose equivalent MLIR Operator is to be tagged with the unit attribute
    - `name`: str. The name of the unit attribute
    """

    def argtypes() -> List[CallingMetafunction.ArgType]:
        return [tag.ArgType.MLIR, tag.ArgType.PYTHON]
    

    def _on_Call(visitor: "ToMLIR", args: List[Any]) -> OpView:
        mlir, attr_name = args

        if not issubclass(type(mlir), OpView):
            mlir = lower_single(mlir).owner
        
        if type(attr_name) is not str: raise TypeError("Attribute name is not a string")
        mlir.attributes[attr_name] = UnitAttr.get()
        return mlir


class match_tag(CallingMetafunction):
    
    def argtypes():
        return [match_tag.ArgType.MLIR, match_tag.ArgType.PYTHON]
    

    def _on_Call(visitor: "ToMLIR", args: List[Any]) -> OpView:
        target, attr_name = args
        
        return structured.MatchOp(
            transform.AnyOpType.get(),
            target,
            op_attrs={attr_name: UnitAttr.get()},
            # ops=['scf.for'] # TODO: assume this is always undefined for now
        )
    

class fuse_into(CallingMetafunction):

    def argtypes() -> List['fuse_into.ArgType']:
        return [fuse_into.ArgType.MLIR, fuse_into.ArgType.MLIR]
    
    def _on_Call(visitor: "ToMLIR", args: List[Any]) -> OpView:
        loop, target = args

        return taffine.ValidatorFuseIntoOp(
            transform.AnyOpType.get(), # TODO: assume this is always AnyOpType for now
            loop,
            target
        )
    

class fuse(CallingMetafunction):

    def argtypes() -> List['fuse.ArgType']:
        return [fuse.ArgType.MLIR, fuse.ArgType.MLIR, fuse.ArgType.PYTHON]
    
    def _on_Call(visitor: "ToMLIR", args: List[Any]) -> OpView:
        target1, target2, depth = args

        return taffine.ValidatorFuseOp(
            transform.AnyOpType.get(), # TODO: assume this is always AnyOpType for now
            target1,
            target2,
            depth
        )
    

class tile(CallingMetafunction):

    def argtypes() -> List['tile.ArgType']:
        return [tile.ArgType.MLIR, tile.ArgType.PYTHON, tile.ArgType.PYTHON]
    
    def _on_Call(visitor: "ToMLIR", args: List[Any]) -> OpView:
        target, tile_sizes, retlen = args

        return taffine.ValidatorLoopTilingOp(
            [transform.AnyOpType.get()] * retlen,
            target,
            tile_sizes=tile_sizes
        )
