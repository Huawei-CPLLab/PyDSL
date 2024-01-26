from ctypes import POINTER, c_uint64
from functools import cache, reduce
import typing
from typing import Any, Tuple

import mlir.ir as mlir
from mlir.ir import *
import mlir.dialects.memref as memref
import mlir.dialects.affine as affine

from pydsl.type import Index, Lowerable, lower_flatten, lower_single
from pydsl.affine import AffineMapExpr


# magic value that equates to MLIR's "?" dimension symbol
DYNAMIC = -9223372036854775808

class MemRef:

    value: Value
    shape: typing.Tuple[int] = None
    element_type: Lowerable = None
    _default_subclass_name = "MemrefUnnamedSubclass"

    @staticmethod
    @cache
    def class_factory(shape: typing.Tuple[int], element_type, name=_default_subclass_name):
        """
        Create a new subclass of Memref dynamically with the specified dimensions and type
        """
        # TODO: this check cannot be done right now because types can't be lowered outside of MLIR context
        # if len(lower(element_type)) != 1: raise TypeError(f"Memref cannot store composite types, got {element_type} which lowers to {lower(element_type)}")

        return type(
            name, (MemRef,), 
            {
                "shape": shape, 
                "element_type": element_type,
            }
        )
    
    # Convenient alias
    get = class_factory

    def __init__(self, rep: OpView | Value) -> None:

        if isinstance(rep, OpView): rep = rep.result

        if (rep_type := type(rep.type)) is not MemRefType:
            raise TypeError(f"{rep_type} cannot be casted as a Memref")
        if not all([
            self.shape == tuple(rep.type.shape),
            lower_single(self.element_type) == rep.type.element_type
        ]):
            raise TypeError(f"Expected shape {'x'.join([str(sh) for sh in self.shape])}x{lower_single(self.element_type)}," 
                             "got OpView with shape {'x'.join([str(sh) for sh in rep.type.shape])}x{rep.type.element_type}")

        self.value = rep


    # https://docs.python.org/3/reference/datamodel.html#object.__getitem__
    def __getitem__(self, key: Index | AffineMapExpr) -> Any:
        match key:
            case Index():
                return self.element_type(
                    memref.LoadOp(lower_single(self), [lower_single(key)])
                )
            
            case tuple():
                return self.element_type(
                    memref.LoadOp(lower_single(self), lower_flatten(key))
                )
            
            case AffineMapExpr():
                key = key.lowered()
                return self.element_type(
                    affine.AffineLoadOp(
                        lower_single(self.element_type), 
                        lower_single(self),
                        indices=[*key.dims, *key.syms],
                        map=key.map))
            
            case _:
                raise TypeError(f"{type(key)} cannot be used to index a Memref")
    

    def __setitem__(self, key: Index | AffineMapExpr, value: Any) -> OpView:
        match key:
            case Index():
                return memref.StoreOp(lower_single(value), lower_single(self), [lower_single(key)])
            
            case tuple():
                return memref.StoreOp(
                    lower_single(value), lower_single(self), lower_flatten(key)
                )

            case AffineMapExpr():
                key = key.lowered()
                return affine.AffineStoreOp(
                    lower_single(value),
                    lower_single(self),
                    indices=[*key.dims, *key.syms],
                    map=key.map
                )

    @cache
    def lower(self) -> Tuple[Value]:
        return (self.value,)


    @classmethod
    def lower_class(cls) -> Tuple[mlir.Type]:
        if not all([cls.shape, cls.element_type]):
            e = TypeError("Attempted to lower Memref without defined dims or type")
            if (clsname := cls.__name__) != MemRef._default_subclass_name:
                e.add_note(f"Hint: class name is {clsname}")
            raise e
        
        return (MemRefType.get(list(cls.shape), lower_single(cls.element_type)),)
    

    @classmethod
    def ctype_type(cls) -> Tuple[Type]:
        return (
            POINTER(*cls.element_type.ctype_type()), 
            POINTER(*cls.element_type.ctype_type()), 
            c_uint64, c_uint64, c_uint64, c_uint64, c_uint64,)


    @classmethod
    def to_ctype(cls, pyval: float | int | bool):
        li = (cls.type_to_ctype(cls.element_type)[0] * reduce(lambda x,y: x * y, cls.shape))(*pyval)
        return (li, li, 0, *(cls.shape), *(cls.shape),)


# Convenient alias
MemRefFactory = MemRef.class_factory