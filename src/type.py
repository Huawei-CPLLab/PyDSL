from functools import cache, reduce
import typing
from typing import Any, Protocol, Tuple, runtime_checkable, Self
from enum import Enum, auto
import ctypes

import mlir.ir as mlir
from mlir.ir import *
from mlir.dialects import transform
import mlir.dialects.arith as arith

# To keep things simple for now, let's just implement __add__, __sub__, __mul__, __truediv__, __floordiv__, and __bool__
# List of other dunder functions and right-hand side variants used by numerical operators are supplied by both Python's and Mojo's documentation:
# https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types
# https://docs.modular.com/mojo/stdlib/builtin/int.html#int

# Differences from Mojo's struct-defined type:
# - Mojo just has MLIR's built-in index type as Int. We make low-level width and sign distinctions, and we always use signless int behind-the-scene.
# - Mojo does not rely on Python types. They are a first-class language. We however rely on Python types to initialize these classes
# - Mojo allows casting between types, such as passing an Int into an Index() to cast it into Index. We don't allow it for now for the sake of simplicity.
# - Mojo seems to have their own binding between MLIR and Mojo, via __mlir_type and __mlir_attr namespaces. We just use MLIR's own Python binding.
# - All dunder functions are inlined
# - An explicit lower function is needed to convert the class into an MLIR OpView object that can be manipulated by MLIR. Mojo simply uses a custom struct statement and assume all members of the struct are seen on the MLIR layer
# - TODO: not sure what Mojo's @register_passable means


@runtime_checkable
class Lowerable(Protocol):
    @cache
    def lower(self) -> Tuple[Value]:
        pass
    
    @classmethod
    def lower_class(cls) -> Tuple[mlir.Type]:
        pass


def lower(
        v: Lowerable | type | OpView | Value | mlir.Type
    ) -> Tuple[Value] | Tuple[mlir.Type]:
    """
    Convert a `Lowerable` type, type instance, and other MLIR objects into its lowest MLIR representation, as a tuple.

    This function is *not* idempotent.

    Specific behavior:
    - If `v` is a `Lowerable` type, a `mlir.ir.Type` is returned
    - If `v` is a `Lowerable` type instance, a `mlir.ir.Value` is returned
    - If `v` is an `mlir.ir.OpView` type instance, then its results (of type `mlir.ir.Value`) are returned
    - If `v` is already an `mlir.ir.Value` or `mlir.ir.Type`, `v` is returned enclosed in a tuple
    - If `v` is not any of the types above, `TypeError` will be excepted

    For example:
    - `lower(Index)` should be equivalent to `(IndexType.get(),)`
    - `lower(Index(5))` should be equivalent to `(ConstantOp(IndexType.get(), 5).results,)`
    - ```lower(UInt8(4).__add__(UInt8(5)))``` should be equivalent to ::

        tuple(AddIOp(
            ConstantOp(IntegerType.get_signless(8), 4), )
            ConstantOp(IntegerType.get_signless(8), 5)).results)
    """
    match v:
        case OpView():
            return tuple(v.results)
        case Value() | mlir.Type():
            return (v,)
        case type() if issubclass(v, Lowerable):
            # Lowerable class
            return v.lower_class()
        case _ if issubclass(type(v), Lowerable):
            # Lowerable class instance
            return v.lower()
        case _:
            raise TypeError(f"{v} is not Lowerable")


def lower_single(
        v: Lowerable | type | OpView | Value | mlir.Type
    ) -> Value | mlir.Type | Value | mlir.Type:
    """
    lower with the return value stripped of its tuple.
    Lowered output tuple must have exactly length of 1. Otherwise, `ValueError` is excepted.

    This function is idempotent.
    """

    res = lower(v)
    if len(res) != 1: raise ValueError(f"Lowering expected single element, got {res}")
    return res[0]

 
def lower_flatten(li):
    """
    Apply lower to each element of the list, then unpack the resulting tuples within the list.
    """
    # Uses map-reduce
    # Map: lower each element
    # Reduce: flatten the resulting list of tuples into a list of its constituents
    return reduce(lambda a, b: a + [*b], map(lower, li), [])


class Sign(Enum):
    SIGNED = auto()
    UNSIGNED = auto()
 

class Int:
    width: int = None
    sign: Sign = None
    value: Value

    def __init_subclass__(cls, /, width: int, sign: Sign=Sign.SIGNED, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls.width = width
        cls.sign = sign


    def __init__(self, rep: Any) -> None:
        # TODO: There is no good way to enforce that the OpView type passed in has the right sign.
        # This class is technically low-level enough that it's possible to construct the wrong sign with this function.
        # This is because MLIR by default uses signless for many dialects, and it's up to the language to enforce signs.
        # Users who never touch type implementation won't need to worry, but those who develop type classes can potentially
        # use the wrong sign when wrapping their MLIR OpView back into a language type.
        
        if not all([self.width, self.sign]):
            raise TypeError(f"Attempted to initialize {type(self)} without defined size or sign")
        
        if isinstance(rep, OpView): rep = rep.result

        match rep:
            case int():
                if self.sign == Sign.UNSIGNED and rep < 0:
                    raise ValueError(f"Expected positive value for signless Int, got {rep}")

                self.value = arith.ConstantOp(self.lower_class()[0], rep).result
            
            case Value():
                if (rep_type := type(rep.type)) is not IntegerType:
                    raise TypeError(f"{rep_type} cannot be casted as an Int")
                if (width := rep.type.width) != self.width:
                    raise TypeError(f"Int expected to be {self.width}, got {width}")
                if not rep.type.is_signless:
                    raise TypeError(f"Ops passed into Int must have signless result, but was signed or unsigned")

                self.value = rep

            case _ if callable(getattr(rep, 'Int', None)): # TODO: create a protocol called IntCastable. This protocol is broken right now
                self.value = rep.Int(type(self))

            case _:
                raise TypeError(f"{rep} cannot be casted as an Int")
        
    
    @cache
    def lower(self) -> Tuple[Value]:
        return (self.value,)


    @classmethod
    def lower_class(cls) -> Tuple[mlir.Type]:
        return (IntegerType.get_signless(cls.width),)
    

    def _signless_same_type_assertion(self, val) -> None:
        if not issubclass(type(val), Int):
            raise TypeError(f"{val} is not an Int")

        if val.width != self.width:
            raise TypeError(f"Int of size {self.width} and {val.width} are incompatible in operations")


    def _signful_same_type_assertion(self, val) -> None:
        self._signless_same_type_assertion(val)

        if val.sign != self.sign:
            raise TypeError(f"Int with different signs cannot be used in signful operations")
    
    # TODO: these arith operators should have more strict sign-unsigned checks
    # if either of the operation is signed, then the output should be signed
    def __add__(self, rhs: 'Int') -> 'Int':
        self._signless_same_type_assertion(rhs)
        return type(self)(arith.AddIOp(self.value, rhs.value))
    

    def __sub__(self, rhs: 'Int') -> 'Int':
        self._signless_same_type_assertion(rhs)
        return type(self)(arith.SubIOp(self.value, rhs.value))


    def __mul__(self, rhs: 'Int') -> 'Int':
        self._signless_same_type_assertion(rhs)
        return type(self)(arith.MulIOp(self.value, rhs.value))
    
    # TODO: __truediv__ cannot be implemented right now as it returns floating points 
    
    def __floordiv__(self, rhs: 'Int') -> 'Int':
        self._signful_same_type_assertion(rhs)
        # assertion ensures that self and rhs have the same sign
        op = arith.FloorDivSIOp if (self.sign == Sign.SIGNED) else arith.DivUIOp
        return type(self)(op(self.value, rhs.value))

    
    @classmethod
    def ctype_type(cls) -> Tuple[Type]:
        ctypes_map = {
            (Sign.SIGNED,   1   ): ctypes.c_bool,
            (Sign.SIGNED,   8   ): ctypes.c_int8,
            (Sign.SIGNED,   16  ): ctypes.c_int16,
            (Sign.SIGNED,   32  ): ctypes.c_int32,
            (Sign.SIGNED,   64  ): ctypes.c_int64,

            (Sign.UNSIGNED, 1   ): ctypes.c_bool,
            (Sign.UNSIGNED, 8   ): ctypes.c_uint8,
            (Sign.UNSIGNED, 16  ): ctypes.c_uint16,
            (Sign.UNSIGNED, 32  ): ctypes.c_uint32,
            (Sign.UNSIGNED, 64  ): ctypes.c_uint64,
        }
        
        if (key := (cls.sign, cls.width)) in ctypes_map:
            return (ctypes_map[key],)

        raise TypeError(f"{cls.__name__} does not have a corresponding ctype")


    @classmethod
    def to_ctype(cls, pyval: Any):
        try:
            pyval = int(pyval)
        except Exception as e:
            raise TypeError(f"{pyval} cannot be converted into an Int ctype") from e
        
        if (1 << cls.width) <= pyval:
            raise TypeError(f"{pyval} cannot fit into an Int of size {cls.width}")
        
        if (cls.sign is Sign.UNSIGNED and pyval < 0):
            raise ValueError(f"Expected positive pyval for signless Int, got {pyval}")

        return (pyval,)
    

    @classmethod
    def from_ctype(cls, cval: Any):
        return int(cval)
    

    def Int(self) -> Self:
        return self
    

class UInt8(Int, width=8, sign=Sign.UNSIGNED):
    pass

class UInt16(Int, width=16, sign=Sign.UNSIGNED):
    pass

class UInt32(Int, width=32, sign=Sign.UNSIGNED):
    pass

class UInt64(Int, width=64, sign=Sign.UNSIGNED):
    pass

class SInt8(Int, width=8, sign=Sign.SIGNED):
    pass

class SInt16(Int, width=16, sign=Sign.SIGNED):
    pass

class SInt32(Int, width=32, sign=Sign.SIGNED):
    pass

class SInt64(Int, width=64, sign=Sign.SIGNED):
    pass

# It's worth noting that Python treat bool as an integer, meaning that e.g. (1 + True) == 2
# To reflect this behavior, Bool inherits all integer operator overloading functions

# TODO: Bool currently does not accept anything except for Python value. It should also support ops returning i1
class Bool(Int, width=1, sign=Sign.UNSIGNED):

    def __init__(self, literal: bool) -> None:
        if type(literal) is not bool:
            raise TypeError(f"Literal {literal} is not a bool.")
        
        lit_as_bool = 1 if literal else 0

        self.value = arith.ConstantOp(IntegerType.get_signless(1), lit_as_bool).result
     

    def Bool(self) -> 'Bool':
        return self
    

    @classmethod
    def from_ctype(cls, cval: Any) -> bool:
        return bool(cval)
    
    
    @classmethod
    def to_ctype(cls, pyval: int | bool):
        try:
            pyval = bool(pyval)
        except Exception as e:
            raise TypeError(f"{pyval} cannot be converted into a {cls.__name__} ctype. Reason: {e}")

        return (pyval,)
    

    @classmethod
    def from_ctype(cls, cval: Any):
        return bool(cval)


class Float:
    width: int
    mlir_type: mlir.Type
    value: Value

    def __init_subclass__(cls, /, width: int, mlir_type: mlir.Type, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.width = width
        cls.mlir_type = mlir_type


    def __init__(self, rep: Any) -> None:
        if not all([self.width, self.mlir_type]):
            raise TypeError("Attempted to initialize Float without defined width or mlir_type")
        
        # TODO: Code duplication in many classes. Consider a superclass?
        if isinstance(rep, OpView): rep = rep.result

        match rep:
            case float() | int() | bool():
                rep = float(rep)
                self.value = arith.ConstantOp(
                                self.lower_class()[0], 
                                rep).result
            
            case Value():
                if (rep_type := type(rep.type)) is not self.mlir_type:
                    raise TypeError(f"{rep_type} cannot be casted as a {type(self).__name__}")

                self.value = rep

            case _ if callable(getattr(rep, 'Float', None)): # FIXME
                self.value = rep.Float()

            case _:
                raise TypeError(f"{rep} cannot be casted as a {type(self).__name__}")

    @cache
    def lower(self) -> Tuple[Value]:
        return (self.value,)
    
    @classmethod
    def lower_class(cls) -> Tuple[mlir.Type]:
        return (cls.mlir_type.get(),)
    

    def _same_type_assertion(self, val):
        if type(self) is not type(val): 
            raise TypeError(f"{type(self).__name__} cannot be added with {type(val).__name__}")


    def __add__(self, rhs: 'Float') -> 'Float':
        self._same_type_assertion(rhs)
        return type(self)(arith.AddFOp(self.value, rhs.value))
    

    def __sub__(self, rhs: 'Float') -> 'Float':
        self._same_type_assertion(rhs)
        return type(self)(arith.SubFOp(self.value, rhs.value))


    def __mul__(self, rhs: 'Float') -> 'Float':
        self._same_type_assertion(rhs)
        return type(self)(arith.MulFOp(self.value, rhs.value))
    

    def __truediv__(self, rhs: 'Float') -> 'Float':
        self._same_type_assertion(rhs)
        return type(self)(arith.DivFOp(self.value, rhs.value))
    
    # TODO: floordiv cannot be implemented so far. float -> int floor op needed.

    @classmethod
    def ctype_type(cls) -> Tuple[Type]:
        ctypes_map = {
            32: ctypes.c_float,
            64: ctypes.c_double,
            80: ctypes.c_longdouble,
        }
        
        if (key := cls.width) in ctypes_map:
            return (ctypes_map[key],)

        raise TypeError(f"{cls.__name__} does not have a corresponding ctype.")

    @classmethod
    def to_ctype(cls, pyval: float | int | bool):
        try:
            pyval = float(pyval)
        except Exception as e:
            raise TypeError(f"{pyval} cannot be converted into a {cls.__name__} ctype. Reason: {e}")

        return (pyval,)
    

    @classmethod
    def from_ctype(cls, cval: Any):
        return float(cval)


class F16(Float, width=16, mlir_type=F16Type):
    pass

class F32(Float, width=32, mlir_type=F32Type):
    pass

class F64(Float, width=64, mlir_type=F64Type):
    pass


# TODO: for now, you cannot do math on Index
class Index:

    value: OpView

    def __init__(self, rep: int | OpView | Value) -> None:

        if isinstance(rep, OpView): rep = rep.result

        match rep:
            case int():
                self.value = arith.ConstantOp(self.lower_class()[0], rep).result
            
            case Value():
                if (rep_type := type(rep.type)) is not IndexType:
                    raise TypeError(f"{rep_type} cannot be casted as an Index")

                self.value = rep
            
            case _ if callable(getattr(rep, 'Index', None)): # FIXME
                self.value = rep.Index()

            case _:
                raise TypeError(f"{rep} cannot be casted as an Index")
    
    @cache
    def lower(self) -> Tuple[Value]:
        return (self.value,)


    @classmethod
    def lower_class(cls) -> Tuple[mlir.Type]:
        return (IndexType.get(),)
    

    def Int(self, cls: type) -> Int:
        op = {
            Sign.SIGNED:    arith.index_cast,
            Sign.UNSIGNED:  arith.index_castui,
        }

        return op[cls.sign](IntegerType.get_signless(cls.width), self.value)
    

    def _same_type_assertion(self, val):
        if type(self) is not type(val): 
            raise TypeError(f"{type(self).__name__} cannot be added with {type(val).__name__}")
    

    def __add__(self, rhs: 'Index') -> 'Index':
        self._same_type_assertion(rhs)
        return type(self)(arith.AddIOp(self.value, rhs.value))
    

    def __sub__(self, rhs: 'Index') -> 'Index':
        self._same_type_assertion(rhs)
        return type(self)(arith.SubIOp(self.value, rhs.value))


    def __mul__(self, rhs: 'Index') -> 'Index':
        self._same_type_assertion(rhs)
        return type(self)(arith.MulIOp(self.value, rhs.value))
    

    def __truediv__(self, rhs: 'Index') -> 'Index':
        raise NotImplementedError() # TODO
    

    def __floordiv__(self, rhs: 'Index') -> 'Index':
        raise NotImplementedError() # TODO
            

    @classmethod
    def ctype_type(cls) -> Tuple[Type]:
        # TODO: this needs to be different depending on the platform. I'm not sure how to determine width of index from looking at the system info.
        
        return (ctypes.c_int64,)


    @classmethod
    def to_ctype(cls, pyval: float | int | bool):
        try:
            pyval = float(pyval)
        except Exception as e:
            raise TypeError(f"{pyval} cannot be converted into a {cls.__name__} ctype. Reason: {e}")

        return (pyval,)
    

    @classmethod
    def from_ctype(cls, cval: Any):
        return float(cval)


class AnyOp:

    def __init__(self, *_):
        raise NotImplementedError()

    @cache
    def lower(self):
        raise NotImplementedError()

    @classmethod
    def lower_class(cls) -> Tuple[mlir.Type]:
        return (transform.AnyOpType.get(),)