import ast
import inspect
import subprocess             
from pathlib import Path
from tempfile import TemporaryDirectory, NamedTemporaryFile
import typing
from typing import IO, Any, Callable
import ctypes
from ctypes import cdll, POINTER, c_uint64
from functools import reduce, cache
from logging import warn

from pydsl.compiler import ToMLIR
import pydsl.type as dsltype

# Global for storing all temporary compiled files
bin = TemporaryDirectory(prefix='pydsl_bin_')
binpath = Path(bin.name)


flags = [
    "-convert-linalg-to-loops",
    "-lower-affine",
    "-convert-scf-to-cf",
    "-finalize-memref-to-llvm",
    "-convert-func-to-llvm",
    "-reconcile-unrealized-casts",
]

def log_stderr(cmds, result):
    if result.stderr: warn(f"""The following error is caused by this command: {cmds}.
Depending on the severity of the message, compilation may fail entirely.
{'*' * 20}
{result.stderr.decode('utf-8')}{'*' * 20}""")


def run_and_get_output(cmds):
    result = subprocess.run(cmds, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    log_stderr(cmds, result)
    return result.stdout.decode('utf-8') if result.stdout else None


def run_and_pipe_output(cmds, stdout: IO):
    result = subprocess.run(cmds, stdout=stdout, stderr=subprocess.PIPE)
    log_stderr(cmds, result)
    return result.stdout.decode('utf-8') if result.stdout else None


def check_cmd(cmd: str) -> None:
    pass # TODO something should be done to ensure that the command exists at all, and should use logging.info to inform which executable is being used


def mlir_passes(src: Path, flags, cmd='mlir-opt') -> NamedTemporaryFile:
    file = NamedTemporaryFile(dir=binpath, suffix='.mlir', delete=False)
    run_and_pipe_output([cmd, *flags, src], file)
    return file


def mlir_to_ll(src: Path, cmd='mlir-translate') -> NamedTemporaryFile:
    file = NamedTemporaryFile(dir=binpath, suffix='.ll', delete=False)
    run_and_get_output([cmd, '--mlir-to-llvmir', src, '-o', file.name])
    return file


def ll_to_so(src: Path, cmd='clang') -> NamedTemporaryFile:
    file = NamedTemporaryFile(dir=binpath, suffix='.so', delete=False)
    run_and_get_output([cmd, '-O3', '-shared', '-target', 'aarch64-unknown-linux-gnu', src, '-o', file.name])
    return file


def compose(funcs):
    def payload(x):
        y = x
        for f in funcs:
            y = f(y)
        
        return y

    return payload


class CompiledFunction:

    def __init__(
        self,
        f: Callable, 
        locals: dict[str, Any], 
        transform_seq: Callable[[Any], None] | None = None,
        auto_build: bool = True,
    ) -> None:
        """
        
        """

        # These variables must be immutable because emit_mlir relies on them and caches its result
        self._f = f 
        self._locals = locals
        self._transform_seq = transform_seq
        
        self._so = None
        self._loaded_so = None
        self._so_f = None

        if auto_build:
            self.build()

    @cache
    def emit_mlir(self) -> str:
        f_ast = ast.parse(inspect.getsource(self._f))
        transform_seq_ast = ast.parse(inspect.getsource(self._transform_seq)) \
                            if self._transform_seq is not None                \
                            else None

        to_mlir = ToMLIR(self._locals)
        return to_mlir.compile(f_ast, transform_seq=transform_seq_ast)
    

    def dump_mlir(self) -> None:
        print(self.emit_mlir())


    def get_function_signature(self) -> inspect.Signature:
        return inspect.signature(self._f)
    

    def load_so(self):
        self._loaded_so = cdll.LoadLibrary(self._so.name)
        self._so_f = getattr(self._loaded_so, self._f.__name__)
        sig = self.get_function_signature()
        self._so_f.restype = self.type_to_ctype(sig.return_annotation)[0]

        argtypes = [
            self.type_to_ctype(sig.parameters[key].annotation) for key in sig.parameters]
        
        # this operation flattens a list of tuples
        self._so_f.argtypes = sum(argtypes, ())


    # TODO: these conversions should be migrated to classes of each type via polymorphism rather than held by this class
    @classmethod
    def type_to_ctype(cls, typ) -> typing.Tuple[type]:
        return typ.ctype_type()
        match typ:
            case dsltype.i1 | dsltype.si1 | dsltype.ui1:
                return (ctypes.c_bool,)
            case dsltype.i8 | dsltype.si8:
                return (ctypes.c_int8,)
            case dsltype.i16 | dsltype.si16:
                return (ctypes.c_int16,)
            case dsltype.i32 | dsltype.si32:
                return (ctypes.c_int32,)
            case dsltype.ui8:
                return (ctypes.c_uint8,)
            case dsltype.ui16:
                return (ctypes.c_uint16,)
            case dsltype.ui32:
                return (ctypes.c_uint32,)
            case dsltype.f32:
                return (ctypes.c_float,)
            case dsltype.f64:
                return (ctypes.c_double,)
            case dsltype.index:
                return (ctypes.c_uint64,) # TODO: this is fixed to uint64
            case _ if type(typ) is dsltype.MemRefTypeTODO:
                return (
                    POINTER(*cls.type_to_ctype(typ.type)), 
                    POINTER(*cls.type_to_ctype(typ.type)), 
                    c_uint64, c_uint64, c_uint64, c_uint64, c_uint64,)
            case _:
                raise TypeError(f"{typ} is not supported as a function argument or return type")


    @classmethod
    def val_to_ctype(cls, typ, val: Any) -> typing.Tuple[type]:
        return typ.to_ctype(val)
        match arg:
            case dsltype.i1 | dsltype.si1 | dsltype.ui1:
                return (int(val),)
            case dsltype.i8 | dsltype.si8:
                return (int(val),)
            case dsltype.i16 | dsltype.si16:
                return (int(val),)
            case dsltype.i32 | dsltype.si32:
                return (int(val),)
            case dsltype.ui8:
                return (int(val),)
            case dsltype.ui16:
                return (int(val),)
            case dsltype.ui32:
                return (int(val),)
            case dsltype.f32:
                return (float(val),)
            case dsltype.f64:
                return (float(val),)
            case dsltype.index:
                return (int(val),)
            case _ if type(arg) is dsltype.MemRefTypeTODO:
                li = (cls.type_to_ctype(arg.type)[0] * reduce(lambda x,y: x * y, arg.dim))(*val)
                return (li, li, 0, *(arg.dim), *(arg.dim),)
            case _:
                raise TypeError(f"{arg} is not supported as a function argument or return type")


    def build(self) -> None:
        file = NamedTemporaryFile(dir=binpath, suffix='.mlir')
        mlir = self.emit_mlir()
        
        with open(file.name, 'w') as f:
            f.write(mlir)

        temp_file_to_path = lambda tempf: Path(tempf.name)

        self._so = compose([
            temp_file_to_path,
            lambda path: mlir_passes(path, flags),
            temp_file_to_path,
            mlir_to_ll,
            temp_file_to_path,
            ll_to_so,
        ])(file)

        self.load_so()


    def __call__(self, *args):
        # TODO: arguments need to be converted to ctype equivalents
        sig = self.get_function_signature()
        mapped_args = [
            self.val_to_ctype(sig.parameters[key].annotation, a) for key, a in zip(sig.parameters, args)]
        mapped_args = sum(mapped_args, ()) # flatten the list of tuples
        return self._so_f(*mapped_args)


def compile(
        f_locals: dict[str, Any],
        transform_seq: Callable[[Any], None] | None = None,
        dump_mlir: bool = False,
        auto_build: bool = True,
        ) -> Callable[..., CompiledFunction]:
    """
    Compile the function into MLIR and lower it to a temporary shared library object.

    The lowered function is a CompiledFunction object which may be called directly to run the respective function in the library.

    f_locals: a dictionary of local variables you want the function to have access to. Typically passing in Python's built-in `locals()` is sufficient.
    transform_seq: the function acting as the transform sequence that is intended to transform this function
    dump_mlir: whether or not to print out the MLIR output to standard output. This is helpful if you want to pipe the MLIR to your own toolchain
    """
    
    def compile_payload(f: Callable) -> CompiledFunction:
        cf = CompiledFunction(f, 
                                f_locals, 
                                transform_seq=transform_seq,
                                auto_build=auto_build)
        
        if dump_mlir:
            cf.dump_mlir()
        
        return cf
    
    return compile_payload