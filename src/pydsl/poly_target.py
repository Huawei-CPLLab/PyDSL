import ast
import ctypes
import dataclasses
import inspect
from inspect import BoundArguments
import logging
import re
import subprocess
import textwrap
import typing
from abc import ABC, abstractmethod
from ast import AST
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from ctypes import POINTER, Structure, cdll
from functools import cache, singledispatch
from logging import info, warning
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory, mkdtemp
from typing import (
    IO,
    Any,
    Optional,
    Protocol,
    Self,
    Type,
    Union,
    runtime_checkable,
)

from pydsl.compiler import CompilationError, Dialect, Module, Source, ToMLIR
from pydsl.func import Function
from pydsl.type import Tuple

import numpy as np

from pydsl.compiler import CompilationError, ToMLIR
from pydsl.memref import DYNAMIC, MemRef
from pydsl.protocols import ArgContainer
from pydsl.frontend import CTarget, SupportsCType, CompilationSetting

"""
NOTE: This feature is largely deprecated and is kept here to maintain
compatibility and as an example of how to implement a custom PyDSL lowering
pipeline. If you are actually trying to run anything with the PolyCTarget,
it will likely require a not insignificant amount of tweaking.
"""


@dataclasses.dataclass
class PolySetting(CompilationSetting):
    dataset: str = "DEFAULT_DATASET"
    """
    An argument specific to target_class=PolyCTarget.
    This determines the input dataset which the output Polybench program will
    accept.
    """


@runtime_checkable
class SupportsPolyCType(Protocol):
    """
    Special case for Poly CTypes. Poly's CType convention is inconsistent with
    the LLVM convention.

    If this protocol is not specified, the regular SupportsCType should be
    used instead
    """

    @classmethod
    def PolyCType(cls) -> CTypeTreeType:
        """
        Returns the class represented as a tuple tree of ctypes

        NOTE: Structures are not allowed. Represent them instead as a tuple.
        """
        ...

    @classmethod
    def to_PolyCType(cls: type, pyval: Any) -> CTypeTree:
        """
        Take a Python value and convert it to match the types of CType
        """
        ...

    @classmethod
    def from_PolyCType(cls: type, ct: CTypeTree) -> Any:
        """
        Take a tuple tree of ctypes value and convert it into a Python value
        """
        ...


class PolyCTarget(CTarget):
    """
    Subclasses CTarget. Basically the same behavior with a few exceptions.

    - Different compilation pipeline
    - Use Poly calling convention. If that's not defined, use the typical LLVM
    C calling convention.
    - Transform sequence is not ignored.
    """

    setting_type = PolySetting
    # This flag is different on mlir-affine-validator
    flag_print_all_passes = "-validator-print-after-all"

    @classmethod
    def type_to_CType(
        cls, typ: Type[SupportsPolyCType] | Type[SupportsCType]
    ) -> tuple[type]:
        if hasattr(typ, "PolyCType"):
            return typ.PolyCType()

        return typ.CType()

    @classmethod
    def val_to_CType(
        cls, typ: SupportsPolyCType | SupportsCType, val: Any
    ) -> tuple[type]:
        if hasattr(typ, "to_PolyCType"):
            return typ.to_PolyCType(val)

        return typ.to_CType(val)

    @classmethod
    def val_from_CType(
        cls, typ: SupportsPolyCType | SupportsCType, val: Any
    ) -> tuple[type]:
        match val:
            case ():
                # Outermost length is 0
                return None
            case (val_sub,):
                # Outermost length is 1
                if hasattr(typ, "from_PolyCType"):
                    return typ.from_PolyCType(val_sub)

                return typ.from_CType(val_sub)
            case _:
                raise ValueError("CType val must be a tuple of size 0 or 1")

    @cache
    def get_args_ctypes(self, f: Function) -> CTypeTreeType:
        sig = f.signature
        args_t = [sig.parameters[key].annotation for key in sig.parameters]

        if not all([
            isinstance(t, SupportsCType) or isinstance(t, SupportsPolyCType)
            for t in args_t
        ]):
            raise TypeError(
                f"argument types {f.return_type} of {f.name} cannot be "
                "converted into ctypes. Not all elements implement "
                "SupportsCType"
            )

        return tuple(self.type_to_CType(t) for t in args_t)

    @contextmanager
    def compile(self) -> Iterator[Module]:
        to_mlir = ToMLIR(
            self.settings.locals,
            catch_comp_error=self.settings.catch_comp_error,
        )

        try:
            with to_mlir.compile(
                self.src.src_ast, transform_seq=self.get_transform_source()
            ) as out:
                self.out = out
                yield out

        except CompilationError as ce:
            # Add source info and raise it further up the call stack
            ce.src = self.src
            raise ce

    def build(self) -> None:
        mlir = self.emit_mlir()
        mlir_file = NamedTemporaryFile(
            dir=self.binpath, suffix=".mlir", delete=False
        )
        with open(mlir_file.name, "w") as f:
            f.write(mlir)

        affine_validator_mlir_file = NamedTemporaryFile(
            dir=self.binpath, suffix=".mlir", delete=False
        )
        c_file = NamedTemporaryFile(
            dir=self.binpath, suffix=".c", delete=False
        )

        self.run_and_get_output([
            "mlir-affine-validator",
            mlir_file.name,
            "-no-thread-local=1",
            f"-output {affine_validator_mlir_file.name}",
            "--codegen-output",
            c_file.name,
        ])

        so_file = NamedTemporaryFile(
            dir=self.binpath, suffix=".so", delete=False
        )
        self.run_and_get_output([
            "clang",
            "-O3",
            "-DPOLYBENCH_TIME",
            "-DDATA_TYPE_IS_FLOAT",
            # "../utilities/polybench.c",
            f"-D{self.settings.dataset}",
            "-fopenmp",
            "-I/usr/lib/gcc/aarch64-linux-gnu/9/include/",
            "-lomp",
            "-fPIC",
            # "-I../utilities/",
            "-I.",
            c_file.name,
            "-shared",
            "-o",
            so_file.name,
        ])

        self._so = so_file

    @cache
    def load_function(self, f: Function):
        ret_struct: type[Structure] | type = CTypeTreeType_to_Structure(
            self.get_return_ctypes(f)
        )
        # All structs are passed by pointer
        if issubclass(ret_struct, Structure):
            ret_struct = POINTER(ret_struct)

        args_struct: list[type[Structure] | type] = [
            CTypeTreeType_to_Structure(t) for t in self.get_args_ctypes(f)
        ]
        # All structs are passed by pointer
        args_struct = [
            POINTER(t) if issubclass(t, Structure) else t for t in args_struct
        ]

        """
        Manage LLVM C-wrapper calling convention.

        When -llvm-request-c-wrappers gets passed or the function has the unit
        attribute `llvm.emit_c_interface` prior to -convert-func-to-llvm, the
        lowering process:
        - Will create another version of the function with the name
          prepended with `_mlir_ciface_`.
        - All types that are represented in a composite manner, such as MemRef
          or complex types, will be passed into the function through struct
          pointers.
        - If the return type of the function is "composite" in any way, such as
          -> (i32, i32) or -> memref<?x?xi16>, the wrapper function will have
          void return type. Instead of returning the return value directly, it
          writes the return value to the first argument passed into the
          function as a struct pointer.

        Example: If the return type is (i32, memref<?x?xi16>), then when it's
        lowered, the first argument will be expected to be a !llvm.ptr where
        the return type is written to as !llvm.struct<(i32, struct<(ptr, ptr,
        i64, array<2 x i64>, array<2 x i64>)>)>. The function itself is void.

        See more info here:
        https://mlir.llvm.org/docs/TargetLLVMIR/#c-compatible-wrapper-emission
        """

        loaded_so = cdll.LoadLibrary(self._so.name)
        so_f = getattr(loaded_so, f.name)  # TODO: this may throw an error

        if self.has_composite_return(f):
            so_f.restype = None  # void function
            args_struct.insert(0, ret_struct)  # instead write to first arg
        else:
            so_f.restype = ret_struct

        so_f.argtypes = tuple(args_struct)

        return so_f

    def call_function(self, fname: str, *args) -> Any:
        if self.has_composite_return(self.get_func(fname)):
            raise RuntimeError(
                f"PolyCTarget cannot call {fname} because it has "
                f"composite return type"
            )

        if not hasattr(self, "_so"):
            raise RuntimeError(
                f"function {fname} is called before it is compiled"
            )

        f = self.get_func(fname)
        sig = f.signature
        so_f = self.load_function(f)
        if not len(sig.parameters) == len(args):
            raise TypeError(
                f"{f.name} takes {len(sig.parameters)} positional "
                f"argument{'s' if len(sig.parameters) > 1 else ''} "
                f"but {len(args)} were given"
            )

        # This is a bit of a hack that should get cleaned up later
        # Kevin had some ideas about how to best do that
        mapped_args_ct = []

        for ct, param, a in zip(
            self.get_args_ctypes(f),
            sig.parameters.values(),
            args,
            strict=False,
        ):
            if issubclass(param.annotation, MemRef) and (
                DYNAMIC in param.annotation.shape
            ):
                # polybench requires dynamic memrefs to have their stries passed
                # in as arguments
                a: np.ndarray
                _, aligned_ptr, _, _, strides = (
                    param.annotation._ndarray_to_CType(a)
                )
                mapped_args_ct.append(((ctypes.c_void_p,), (aligned_ptr,)))
                for i in range(len(param.annotation.shape)):
                    mapped_args_ct.append(((ctypes.c_int,), (strides[i],)))
            else:
                mapped_args_ct.append((
                    ct,
                    self.val_to_CType(param.annotation, a),
                ))

        mapped_args = [CTypeTree_to_Structure(*i) for i in mapped_args_ct]

        if self.has_void_return(f):
            so_f(*mapped_args)  # Call function
            return None

        if not self.has_composite_return(f):
            retval = so_f(*mapped_args)  # Call function

            # This double tuple construct is essential!
            # - The outermost tuple tells us that the return is not void
            # - The inner tuple formats the returned element in the same
            #   way as a composite struct
            # See docstring of get_return_ctypes for detail
            retval_ct = ((retval,),)  # Call function

            # This is a necessary compensation as MLIR lowering confuses
            # single-element tuples with actual single elements, which
            # is why single-element tuples do not get considered by MLIR
            # as a composite return.
            # This confusion exists at the MLIR builtin language level and
            # there's nothing we can do about it.
            if issubclass(f.return_type, Tuple):
                retval_ct = (retval_ct,)

            return self.val_from_CType(f.return_type, retval_ct)

        # instantiate a structure return type, which by LLVM calling
        # convention is a void function that writes the return value in-place
        # to the first argument
        retval = CTypeTreeType_to_Structure(self.get_return_ctypes(f))()
        mapped_args.insert(0, retval)

        so_f(*mapped_args)  # Call function

        # the result is written in-place to retval. Convert it back to Python
        # CTypes
        retval_ct = CTypeTree_from_Structure(self.get_return_ctypes(f), retval)

        return self.val_from_CType(f.return_type, retval_ct)

    def get_supported_dialects(self) -> set[Dialect]:
        extra_supported = {Dialect.from_name("transform.validator")}
        return (
            super().get_supported_dialects(extra_supported) | extra_supported
        )
