from __future__ import annotations

from abc import ABCMeta, abstractmethod
import ast
from typing import Any, List, TYPE_CHECKING
from enum import Enum, auto

from mlir._mlir_libs._mlir.ir import OpView
if TYPE_CHECKING:
    # This is for imports for type hinting purposes only and which can result in cyclic imports.
    from pydsl.compiler import ToMLIR

class Metafunction(ABCMeta):
    """
    A metaclass whose instances are callables that are evaluated during compile time and manipulates the MLIR output.
    By default, it cannot be called in a typical Python runtime.

    For all instances of Metafunction, and for all class methods annotated as @abstractmethod within the instance, 
    they must be implemented by their subclasses.
    """

    def __new__(cls, name, bases, attr):
        new_cls = super().__new__(cls, name, bases, attr)

        for base in bases:

            if not hasattr(base, "__abstractmethods__"):
                continue

            for abstract_method in base.__abstractmethods__:
                # check if the abstractmethod is overwritten
                if getattr(new_cls, abstract_method) is getattr(base, abstract_method):
                    raise TypeError(f"metafunction {new_cls.__name__} did not implement abstract method {abstract_method}, required by base {base}")

        return super().__new__(cls, name, bases, attr)


    def __call__(self, *_, **__):
        raise RuntimeError(f"{self.__name__} is a metafunction that cannot be called in regular Python runtime.")


class IteratorMetafunction(metaclass=Metafunction):
    """
    A metafunction that intends to behave like a Python iterator.
    
    Instead of implementing __next__, subclasses are required to generate the necessary MLIR
    for the various contexts where an iterator may be legally used in Python.
    """

    @abstractmethod
    def on_For(visitor: ToMLIR, node: ast.For):
        pass


    @abstractmethod
    def on_ListComp(visitor: ToMLIR, node: ast.ListComp):
        pass


class SubscriptingMetafunction(metaclass=Metafunction):
    """
    A metafunction that intends to be put into a Python subscript notation.

    The metafunction is responsible for transforming the full assignment statement into MLIR.
    """

    @abstractmethod
    def on_Store(visitor: ToMLIR, node: ast.Assign):
        pass


    @abstractmethod
    def on_Load(visitor: ToMLIR, node: ast.Subscript):
        pass


class CallingMetafunction(metaclass=Metafunction):
    """
    A metafunction that intends to be called as-is.

    The metafunction is responsible for transforming the Call node that called this function and its constituents.
    Note that if the context of the Call matches those of other Metafunction classes, they may be prioritized as the intended superclass and error may be thrown.
    """

    class ArgType(Enum):
        # This indicates that the argument is a literal or a variable name that is evaluated as a regular Python value
        # The scope of values accepted is very limited right now, only those parsable by ast.literal_eval can be used
        PYTHON = auto()

        # This indicates that the argument should be passed in as expressed in target language, i.e. MLIR OpViews
        # This accepts any Python expression that can be compiled by this compiler
        MLIR = auto()

        # This indicates that the argument should remain as a Python AST
        TREE = auto()
    

    @abstractmethod
    def argtypes() -> List[ArgType]:
        """
        This function must be overwritten to specify the type of the non-variable arguments that the function accepts
        """
        pass


    def varargtype() -> ArgType:
        """
        This function can be overwritten to specify the type of the variable argument
        """

        return None


    @abstractmethod
    def _on_Call(visitor: ast.NodeVisitor, args: List[Any]) -> Any:
        pass


    def _eval_python(arg: ast.AST) -> Any:
        try:
            return ast.literal_eval(arg)
        except Exception:
            raise ValueError(f"This expression cannot be evaluated as an ArgType.PYTHON: {ast.dump(arg)}")
            

    @classmethod
    def _parse_params(cls, visitor: ToMLIR, node: ast.Call) -> List[Any]:
        if cls.varargtype() is None and len(cls.argtypes()) != len(node.args):
            raise ValueError(f"{cls.__name__} expected {len(cls.argtypes())} arguments, got {len(node.args)}")
        
        # functionality for converting a single argument
        def convert(arg, argtype):
            match argtype:
                case cls.ArgType.PYTHON:
                    return cls._eval_python(arg)
                case cls.ArgType.MLIR:
                    return visitor.visit(arg)
                case cls.ArgType.TREE:
                    return arg
                case _:
                    raise TypeError(f"argtype() in {cls.__name__}, must return List[ArgType], got {argtype} as an element in the returned list")

        ret = []
        # evaluate regular args
        for arg, argtype in zip(node.args, cls.argtypes()):
            ret.append(convert(arg, argtype))

        # evaluate varargs
        if (varargtype := cls.varargtype()) is not None:
            for arg in node.args[len(cls.argtypes()):]:
                ret.append(convert(arg, varargtype))

        return ret


    @classmethod
    def on_Call(cls, visitor: ToMLIR, node: ast.Call) -> OpView:
        return cls._on_Call(visitor, cls._parse_params(visitor, node))