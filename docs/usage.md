# Introduction

This document outlines how to use PyDSL, and how it compares with Python.

PyDSL is a compiler that transforms a subset of Python down to MLIR, which can then be run by the user directly in a Python environment. As background, while the existing MLIR infrastructure is essential to our optimization stack, it does not yet provide a language that can describe MLIR program behaviors that also benefits end-developer productivity. As such, PyDSL aims to bridge this gap by providing a faithful Python-based syntax and programming style to writing MLIR programs without compromising the low-level and imperative aspects of MLIR.

# Basics

## Syntax and semantics

Syntax-wise, PyDSL is effectively a subset of Python.
It introduces no new syntax, meaning if PyDSL can parse it, Python can also parse it.

However, not everything written in PyDSL can be run in Python. For example, `range` is a function in Python, but it is a macro subclass in PyDSL.
As such, import statements like `from pydsl.scf import range` will shadow Python's built-in `range` and will cause any references to `range` to break.
In addition, PyDSL permits variables to be used within type hints, which allows users to include MLIR dynamic dimensions in their MemRef types.
This is not permitted in Python.

Conversely, not everything written in Python can be run in PyDSL, even if it's within the PyDSL syntax subset.
For example, PyDSL requires all arguments of a function to be type-hinted, but type-hinting is optional in Python.

Translation between Python and PyDSL code requires minor modification.

Some aspects of PyDSL's semantics deliberately stray from Python's for the purpose of aiding user productivity. For instance, a PyDSL function's return type can be optionally unhinted, but it denotates a None type (i.e. a void return) rather than an Any type.

## Limitations

Besides design-wise limitation on what PyDSL accepts, PyDSL is also in an early stage of development.
A lot of Python constructions cannot be compiled because they are simply not implemented,
such as nested functions and defining a variable inside a for loop and using it outside later.
PyDSL also currently has limited error messages provided for edge cases:
if you stray too far from examples shown in this documentation, you may come across cryptic MLIR errors.

PyDSL prioritizes supporting programs which make use of nested loops and explicit transformations,
and rarely make use of piecewise computations involving `if` statements.
Most of the features implemented in PyDSL and given as example below will reflect this fact.

## Compilation and running

The input program becomes transformed by PyDSL into MLIR, which is then lowered into potentially multiple targets, such as LLVM IR for C, or Poly C. Once compiled, user can directly run the compiled function by passing Python values into the function within the same program, and receive values returned by the program.

The compilation of PyDSL programs is performed by several components:
- The `ast.NodeVisitor` class from the Python standard library parses the input Python program into an AST and visits each node, returning each subtree translated as a partial MLIR program.
- MLIR Python Binding, which offers an API for building MLIR programs from Python.
- `mlir-opt`: the command-line tool that takes the output MLIR and lowers the higher-abstraction dialects down to the target desired by the user. The pipeline of passes is pre-defined and supports all dialects that PyDSL current supports.
- From here, compilation tools and pipeline may differ depending on the target.

As the NodeVisitor traverses the user program, it keeps track of local variables defined by `Assign` nodes. When it comes across a variable usage, it searches its internal stack for the variable name and attempt to resolve it in the same manner as Python. As explored in later sections, the variable stack can include variables defined outside of the program, allowing variables to be imported from other files.

The full compilation process is also described in graphical detail in the December MLIR Open Meeting slides: https://mlir.llvm.org/OpenMeetings/2023-12-21-PyDSL.pdf.


# Function

You can compile a function into an MLIR program by including the `@compile()` annotation.

Here is an example of how to compile and use a function:
```py
from pydsl.frontend import compile
from pydsl.scf import range
from pydsl.memref import MemRefFactory
from pydsl.type import F32, F64, Index


Memref64 = MemRefFactory((40, 40), F64)


@compile()
def hello(a: F32, b: F32) -> F32:
    d: F32 = 12.0
    l: Index = 5

    for i in range(l):
        e: F32 = 3.0
        f = e + d

    return (a / b) + d


retval = hello(25, 3)  # this now calls the compiled library
print(25 / 3 + 12)
print(retval)
```

Some notes:
- Return must always be the last statement in your function, unless the function is void.
- PyDSL has strict typing. All arguments, return value, and constant initiations must be type-hinted.
    - PyDSL accepts only its own types. This is to enforce the fact that you are using MLIR types and need to convert Python types into it when calling the function. Do not use int, float, etc. Instead, use `pydsl.type.F32`, etc.
- `@compile()` automatically pushes the variables defined prior to the decorator into the compiler so that the user can use these variables within the function. This is simply a dictionary of variables. The user can override this behavior by passing any dictionary mapping names to values into the first positional argument of `@compile()`.
    - This value-pushing behavior does its best to capture as much relevant contextual variable as possible. By default, this is `builtins | globals | locals`, where `|` is the dictionary union operation. However, this does not exhibit the same behavior as a regular Python function, which also graduate any used variables from outer nested functions into `locals`. This feature is missing from Python's metaprogramming API and there is no way to perfectly emulate Python's behavior.
- A lot of Python's built-in variables need to be shadowed with its PyDSL equivalent by importing them, such as `range`. This means that you can no longer use Python's own `range` in your Python script.
    - If you do not wish to overshadow Python's own built-in variables, you can name the import differently, e.g. `from pydsl.scf import range as srange` and use `srange` as `range` in your function
- After the function is defined, the compilation is automatically performed. When you call `hello`, you are actually running a compiled library.
    - If you don't want compilation to be automatically performed, call `@compile(locals(), auto_build=False)`
- PyDSL currently only supports compiling a single function at a time (with the exception of a transform function). This means you cannot reference a compiled function from another compiled function. Interop of compiled functions must be done by passing in from and returning values back to CPython.

You can see the MLIR output by setting `dump_mlir=True`
```py
@compile(locals(), dump_mlir=True)
def hello(a: F32, b: F32) -> F32:
    d: F32 = 12.0
    l: Index = 5

    for i in range(l):
        e: F32 = 3.0
        f = e + d

    return (a / b) + d
```

This example outputs
```
module {
  func.func public @hello(%arg0: f32, %arg1: f32) -> f32 {
    %cst = arith.constant 1.200000e+01 : f32
    %c5 = arith.constant 5 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %arg2 = %c0 to %c5 step %c1 {
      %cst_0 = arith.constant 3.000000e+00 : f32
      %2 = arith.addf %cst_0, %cst : f32
    }
    %0 = arith.divf %arg0, %arg1 : f32
    %1 = arith.addf %0, %cst : f32
    return %1 : f32
  }
}
```

# Module

All MLIR program has modules as the top-level construction. For situations where a single function is compiled, a module is implied to be wrapping the function and is automatically constructed by PyDSL. However, when a program needs to be defined as multiple functions or module-level definitions need to be created, the user would need to have full control over the definition of modules.

A module can be compiled from a Python class as seen below.

```py
@compile(body_func="my_body_func")
class CallTest:
    def my_body_func():
        a = 12

    def f1() -> UInt16:
        return f2()

    def f2() -> UInt16:
        return a

assert CallTest.f1() == 12
```

The same `pydsl.frontend.compile` decorator for decorating functions can be used to decorate classes. This decorator will compile all member methods within the class as MLIR functions, except for the function denoted by the body_func compilation setting argument which will instead be compiled as the module body. Caution must be exercised with defining variables in the module body, as most operators in MLIR must be defined in functions rather than module. For PyDSL specifically, initializing numbers without type hint is permitted in the module body (e.g. `a = 12` as seen above) to be used by all functions as it generates no MLIR. All member functions of the module can be called in similar ways as accessing a member method of a Python class (e.g. `CallTest.f1()` as seen above).

The compilation result is as below.

```mlir
module {
  func.func public @f1() -> i16 {
    %0 = call @f2() : () -> i16
    return %0 : i16
  }
  func.func public @f2() -> i16 {
    %c12_i16 = arith.constant 12 : i16
    return %c12_i16 : i16
  }
}
```

Note that `a = 12` is a virtual and delayed variable initialization in PyDSL due to a lack of concrete type hint. The associated `arith.constant` does not get defined until it is used in `@f2`. This relates to the way PyDSL currently conducts type inference.


# Typing system

Below are the common types that you can use in PyDSL that is currently implemented.

It is worth noting that the underlying MLIR type used for both signed and unsigned integers are lowered to MLIR signless integers. This is the convention employed by LLVM 2.0. The types are only distinguished by the sign of the operations they go through.

For example, if two `UInt8` goes through a floor division, they are lowered to signless `i8`s, but the `i8`s always go through `arith.FloorDivUIOp` rather than `arith.FloorDivSIOp`.

| PyDSL type                                                                | Underlying MLIR type        | Accepted Python type                                                                                                                                                                                      |
| ------------------------------------------------------------------------- | --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pydsl.type.UInt8`                                                        | `i8`                        | Anything that can be casted to ctypes.c_uint8                                                                                                                                                             |
| `pydsl.type.UInt16`                                                       | `i16`                       | Anything that can be casted to ctypes.c_uint16                                                                                                                                                            |
| `pydsl.type.UInt32`                                                       | `i32`                       | Anything that can be casted to ctypes.c_uint32                                                                                                                                                            |
| `pydsl.type.UInt64`                                                       | `i64`                       | Anything that can be casted to ctypes.c_uint64                                                                                                                                                            |
| `pydsl.type.SInt8`                                                        | `i8`                        | Anything that can be casted to ctypes.c_int8                                                                                                                                                              |
| `pydsl.type.SInt16`                                                       | `i16`                       | Anything that can be casted to ctypes.c_int16                                                                                                                                                             |
| `pydsl.type.SInt32`                                                       | `i32`                       | Anything that can be casted to ctypes.c_int32                                                                                                                                                             |
| `pydsl.type.SInt64`                                                       | `i64`                       | Anything that can be casted to ctypes.c_int64                                                                                                                                                             |
| `pydsl.type.F32`                                                          | `f32`                       | Anything that can be casted to ctypes.c_float                                                                                                                                                             |
| `pydsl.type.F64`                                                          | `f64`                       | Anything that can be casted to ctypes.c_double                                                                                                                                                            |
| `pydsl.type.Bool`                                                         | `i1`                        | Anything that can be casted to ctypes.c_bool                                                                                                                                                              |
| `pydsl.type.Index`                                                        | `index`                     | Anything that can be casted to ctypes.c_size_t                                                                                                                                                            |
| `pydsl.memref.MemRef` subclass with element type `T` and ranked shape `S` | `memref<SxT>`               | `numpy.ndarray` with such DType/shape or anything that implements `pydsl.memref.SupportsRMRD` protocol that points to a memory space with such DType/shape that persists across the compiled library call |
| `pydsl.type.Tuple`                                                        | Built-in tuple type `(…,…)` | *Tuple is not permitted as an argument type hint but is allowed as a return type hint.* When used as a return type hint, a built-in Python tuple is returned.                                             |
| `pydsl.type.AnyOp`                                                        | `!transform.any_op`         | *Python representation does not exist. This is interpreted at compile-time by MLIR*                                                                                                                       |


## Void return type

Void return is supported via `None`. Specifying return is optional.

A function without a return type hint is assumed to be a void return rather than `typing.Any`.

```py
def f() -> None:
    return

# OR

def f() -> None:
    pass
```

## Tuple return type

The user can return tuples by type hint with `Tuple` from `pydsl.type` module.

Currently, tuples can't be used for type hinting for variables. This is because MLIR does not have a tuple type.

```py
from pydsl.frontend import compile
from pydsl.scf import range
from pydsl.memref import MemRefFactory
from pydsl.type import F32, F64, Index, Tuple


Memref64 = MemRefFactory((40, 40), F64)


@compile(locals(), dump_mlir=True)
def tuple_example(a: F32, b: F32) -> Tuple[F32, Index]:
    d: F32 = 12.0
    l: Index = 5

    for i in range(l):
        e: F32 = 3.0
        f = e + d

    return (a / b) + d, l


a, b = tuple_example(25, 3)

print(f"a = {a} should be {25 / 3 + 12}")
print(f"b = {b} should be {5}")
```

This produces
```mlir
module {
  func.func public @tuple_example(%arg0: f32, %arg1: f32) -> (f32, index) {
    %cst = arith.constant 1.200000e+01 : f32
    %c5 = arith.constant 5 : index
    %0 = arith.divf %arg0, %arg1 : f32
    %1 = arith.addf %0, %cst : f32
    return %1, %c5 : f32, index
  }
}
```

## Typing and arithmetic

There are many ways to define a constant. You can define your constant by hinting a type (either through an annotated assign or through calling the type within an expression). You can also simply write a constant and leave out any typing information and have the compiler infer the type for you.

> Type inference on constants in PyDSL is performed by temporarily assigning an abstract Number type to the constant. Once the constant is used by a concrete MLIR operation, the abstract Number constant will be casted to a concrete `arith.const` operator with a concrete type.
> If you define a constant and never use it, the constant will not show up in the resulting MLIR.

```py
# Assign a new variable with type hinting via annotated assign syntax
a: I32 = 5

# Assign a new variable with type hinting via the type as a function
a = I32(5)

# Have the compiler evaluate the type of this constant lazily
a = 5
```

Using type as a function can help with situations where you need to quickly define a variable, such as in `range` which only accepts index types.

```py
for i in range(Index(5)):
    pass

# OR

for i in range(5):
    pass
```

Basic arithmetic can be performed in very similar way as Python. Integers only support `//`. Floats only support `/`. Modulo is not supported.

Casting can be done by putting the variable in the type as a function. Not all casting are supported yet.
```py
i: UInt32 = 5
f = F32(i)
```

Casting of Floats can be done to extend width, but width reduction is not supported yet.
```py
f: F32 = 5
fwide = F64(f)
```

When you perform operations on two different types, the type on the left-hand side will try to automatically cast the type on the right-hand side.
```py
@compile()
def operation_cast(a: F32, b: SInt32) -> F32:
    return a / b


retval = operation_cast(5.5, 5)
print(retval)  # This prints approx. 1.1

```

## Representing dependent types

Some types in PyDSL, such as `Tensor`, `MemRef`, and `Tuple`, are dependently-typed. This means that the definition of their type relies on instances and variables as type argument. For example, a `MemRef` on its own does not represent anything concrete in MLIR, but `MemRef[F32, 5, 5]` represents a 5x5 `float32` memory space. There are two ways to represent these types in PyDSL: through type factories in the Python metalanguage, or through indexing notation in PyDSL.

To use `MemRef` as an example, `MemRef[F32, 5, 5]` can be defined using `pydsl.memref.MemRefFactory`

```py
from pydsl.frontend import compile
from pydsl.memref import MemRefFactory
from pydsl.scf import range
from pydsl.type import UInt32

MemRef32 = MemRefFactory((5, 5), UInt32)

@compile(locals(), dump_mlir=True)
def hello(m: MemRef32) -> None:
    for i in range(5):
        m[i, i] = UInt32(i)
```

or by directly type-hinting using `MemRef[F32, 5, 5]`. Both methods emit the same MLIR output.

```py
from pydsl.frontend import compile
from pydsl.scf import range
from pydsl.memref import MemRef
from pydsl.type import UInt32

@compile(locals(), dump_mlir=True)
def hello(m: MemRef[UInt32, 5, 5]) -> None:
    for i in range(5):
        m[i, i] = UInt32(i)
```

Implementation-wise, `MemRefFactory` dynamically generates a new subclass of `MemRef` by populating the fields for element type, shape, and then calling the underlying MLIR Python binding function `MemRefType.get` to generate the MLIR equivalent type. Every time `MemRefFactory` is called, a new class is created and returned by the function. Internally, the type hint `MemRef[F32, 5, 5]` also makes use of `MemRefFactory` to generate new subtypes at compile time. Factory methods and type hints in other dependent types work in similar fashion.

## Type inference

PyDSL provides a preliminary type inference system to reduce the amount of times that users would need to explicitly specify a variable, improving the language ergonomics.
This is only applicable to numerical literals and relies entirely on PyDSL's type casting mechanism.

Whenever PyDSL comes across a numerical literal without any type hinting associated with it, PyDSL will associate it with a generic `Number` type that is not associated with any particular numerical type in MLIR. This `Number` type will temporarily hold the literal value in Python representation, and will either be casted into a concrete MLIR type (if it is used) or be eliminated (if it is not used). The specific concrete type depends on the operation performed.

Generally, the user should not need to concern with the usage of `Number` type. When adding a new Macro, however, the user needs to be aware that they may receive this type and that it exhibits unusual properties such as not being lowerable to an MLIR value. The typical strategy is to perform casting before using the input value for any other purpose.


# For loop

For loops are performed using `pydsl.scf.range`, `pydsl.affine.affine_range`, or any subclass of `IteratorMacro`. The syntax is roughly the same as Python, except the iterator is heavily restricted to specific use cases. These iterators currently only work in for loops.

For both `range` and `affine_range`, the rules are the same as Python.
- 1 argument means (exclusive end). Inclusively start at 0. Step 1.
- 2 arguments mean (inclusive start, exclusive end). Step 1.
- 3 arguments mean (inclusive start, exclusive end, step).

Like MLIR for loops, all arguments must be of Index type.

```py
for i in range(7): # this iterates 0, 1, 2, ..., 6
    pass

for i in range(3, 7): # this iterates 3, 4, 5, 6
    pass

for i in range(3, 7, 2): # this iterates 3, 5
    pass
```

# If-else statement

If-else statement in PyDSL is supported by the MLIR `scf` dialect. It is currently implemented on the language level and does not make use of the macro system. While resembling Python, it is currently very restrictive in its capabilities:
-	Return cannot be present in the if statement.

```py
@compile(globals(), dump_mlir=True)
def f(m: MemRefSingle, b: Bool):
    if b:
        m[Index(0)] = UInt32(5)
    else:
        m[Index(0)] = UInt32(10)
```


# Memory

There are two types in PyDSL for storing blocks of memory: `MemRef` and `Tensor`.

A `MemRef` is a reference to a specific location in memory.
Doing an operation on a `MemRef` causes it to be updated in-place.

A `Tensor` is a higher level abstraction and represents a tensor (high dimensional array) object.
A `Tensor` should behave quite similarly to a `numpy.ndarray`.

Internally, a PyDSL `Tensor` first gets lowered to an MLIR `tensor`, then an MLIR `memref`, while a PyDSL `MemRef` gets lowered directly to an MLIR `memref`.

If a PyDSL function has a `MemRef` or a `Tensor` as an argument, it expects to get a `numpy.ndarray` from the Python program.
Other types, such as `list` are not currently supported.

If the function argument is a `MemRef`, the `ndarray` that's passed in will be modified in-place.
If the function argument is a `Tensor`, it is not guaranteed whether it will be modified in-place or if a new copy will be made.

Thus, you should **not access an `ndarray` after it is passed to a PyDSL function as a `Tensor`**.

Here is a very basic example that accumulates the values of `range(3, 7, 2)`:

```py
import numpy as np
from pydsl.frontend import compile
from pydsl.memref import MemRefFactory
from pydsl.type import Index, UInt64
from pydsl.scf import range

MemRef64 = MemRefFactory((1,), UInt64)


@compile(dump_mlir=True)
def hello_memref(m: MemRef64) -> MemRef64:
    for i in range(3, 7, 2):
        m[0] = m[0] + i

    return m

n = np.asarray([0], dtype=np.uint64)

hello_memref(n)
print(n)  # [8]

```

Some notes:
- `MemRef64` is a type that is defined dynamically outside of the compiled function using `MemRefFactory`. It specifies a memory region of a single element requiring a `dtype` of `UInt64`.
- You **must** pass in a Numpy array with the correct type and dimension. The function will not accept any other iterables or arrays with the wrong type. We cannot cast your array for you as most Numpy casting requires a new copy of the array to be created, which your code would not have a reference of.

The alternative to using a NumPy array is to allocate memory directly in PyDSL. The library currently offers `pydsl.memref.alloc` (heap allocation) and `pydsl.memref.alloca` (stack allocation). Deallocation is supported via `pydsl.memref.dealloc`. The MemRef variables may be returned to the Python caller, which are automatically casted as a NumPy array.

```py
@compile()
def f() -> MemRef[UInt32, 1]:
    m_scalar = alloc(MemRef[UInt32, 1])
    m_scalar[0] = 1
    return m_scalar

assert (f() == np.asarray([1], dtype=np.uint32)).all()
```

## Defining the dimension of your array

MemRef supports arbitrary rank. With respect to MLIR's restriction, the element dtype of any MemRef must be one of the following:
- `IntegerType`,
- `F16Type`,
- `F32Type`,
- `F64Type`,
- `IndexType`,
- `ComplexType`

Make sure the first argument is always a tuple. Python has a particular way of writing a single-element tuple.
```py
# (1) is NOT a tuple. This will throw an error!
MemRef64 = MemRefFactory((1), UInt32)

# (1,) is a tuple.
MemRef64 = MemRefFactory((1,), UInt32)
```

When inlining the dimension as type hints, the element type comes before dimension, and dimension is written as comma-separated rather than as a tuple.

```py
# When using inline type hint, dimension comes after and tuple is not used.
@compile()
def f() -> MemRef[UInt32, 1, 2, 3]:
    ...
```

If you want dynamic-length, ranked MemRef, use the `DYNAMIC` constant.
```py
from pydsl.memref import DYNAMIC, MemRefFactory

MemrefF32 = MemRefFactory((DYNAMIC, DYNAMIC), F32) # this creates memref<?x?xf32>
```
- Note that `memref.dim` is currently not supported. The size of the array must be passed in alongside the MemRef into the function.

Unranked MemRefs (e.g. `memref<*xf32>`) are currently not supported.

## MemRef layout

`MemRef`s currently support the default layout and a strided layout.
See https://mlir.llvm.org/docs/Dialects/Builtin/#layout for a description of these layouts.
AFfine map layouts (other than the default identity map) are currently not supported.

To make a `MemRef` with a strided layout, specify the `offset` and `strides` parameters when calling `MemRefFactory`.
E.g.
```py
MemRefStrided = MemRefFactory((10, 20), F32, offset=0, strides=(20, 1))
```

`offset` and `strides` can also have `DYNAMIC` values.

The main usecase for declaring a `MemRef` type with a strided layout is probably if you want to return a `MemRef` that's the
result of a slice (`memref.subview`) operation, since those can result in a `MemRef` with a strided layout.

`Tensor`s do not explicitly have a layout, internally they always have `offset` and `strides` filled with all `DYNAMIC` values.

## Arrray indexing

`Tensor`s and `MemRef`s can be indexed by providing the indices in a single subscript operator, separated by commas.
For example:
```py
def f(m1: MemRef[F64, 10, 10]):
    m1[2, 3] = 50
```

Slicing of `Tensor` and `MemRef` is also supported, and they map to `tensor.extract_slice`, `tensor.insert_slice`, and `memref.subview` in MLIR.
Currently, the return type of an `extract_slice` and `subview` is always a `Tensor`/`MemRef` with fully dynamic dimensions, regardless of whether
it would be possible to statically determine the dimensions at compile time.
For `MemRef`, this means the `offset` and `strides` attributes are also dynamic.
`MemRef` provides `.get_fully_dynamic` to return a `MemRef` type with dynamic shape, offset, and strides, useful for specifying the return type of a
function which returns a subview of a `MemRef`.
For `Tensor`, typing `Tensor[F32, DYNAMIC, DYNAMIC]` or similar is not too verbose.

In the future, we will support casting `MemRef` and `Tensor` to different shapes, which should give more flexibility with specifying the types of
results of slicing.

If the number of indices is less than the rank of the `MemRef` or `Tensor`, the remaining dimensions are assumed to be `::`.

**PyDSL currently does not support negative indices!**
`arr[3:-2:-1]` is valid in Python, and it takes a slice from index `3` to index `len(arr)-2` and reverses it.
However, PyDSL does not parse negative indices in any special way, so `tensor[3:-2:-1]` will get lowered to MLIR exactly as if all indices were non-negative.
This will most likely result in undefined behaviour.

PyDSL currently does not do any bounds checking.
Invalid indexing/slicing compiles without a warning and produces undefined behaviour.

When copying a `Tensor` to a slice of another `Tensor` (e.g. `tensor1[1:6, 3:9:2] = tensor2`), the shape of the inserted tensor must match the shape of the slice.
E.g. in the above example, if `tensor1` has rank `2`, then `tensor2` must have shape `(5, 3)` at runtime.


# Affine programming

PyDSL has support for `affine.for`, `affine.load`, and `affine.store`.

Much like MLIR, these affine operations rely on affine maps for indexing, and affine maps have restrictions on its parameters. See [the MLIR affine dialect documentation](https://mlir.llvm.org/docs/Dialects/Affine/#polyhedral-structures) for details.

## Affine for loop

You can write affine for loops in roughly the same way as a regular for loop using `pydsl.affine.affine_range`. Instead of accepting up to 3 `Index` variables, it accepts up to 3 affine maps, which can be defined using `pydsl.affine.affine_map`. All variables within the affine map must be identifiable as either dimension or symbols. Constants are defined without any type specified.

```py
from pydsl.affine import (
    affine_map as am,
    affine_range as arange,
    symbol as S
)

# ...

for i in arange(am(S(v0)), am(S(v0) + 8), 2):
    pass
```

PyDSL will also try to infer that the arguments within the `affine_range` are affine maps, and that the variables within are symbols or dimensions by using surrounding context, so writing them is usually optional:
```py
from pydsl.affine import affine_range as arange

# ...

for i in arange(v0, v0 + 8, 2):
    pass
```

## Affine load/store

Affine loads/stores are done by simply indexing any MemRef with an affine map.
```py
from pydsl.affine import (
    affine_map as am,
    dimension as D,
    symbol as S,
)

A[am(D(i), S(v1) + 5)] = b
```

PyDSL will also try to infer all the dimensions and symbols if you don't write them by using the surrounding context. In addition, PyDSL assumes that all indexing that occurs within an affine for loop are implicitly affine loads/stores.

```py
for i in arange(am(S(v0)), am(S(v0) + 8), 2):
    A[am(D(i), S(v1) + 5)] = b

# is equivalent to

for i in arange(v0, v0 + 8, 2):
    A[i, v1 + 5] = b
```

## Affine map inference

As shown, when you write affine operations, PyDSL will try its best to infer whether an argument is an affine map, and infer whether variable uses in an affine map are dimensions or symbols. This does not always work, and in these cases you need to explicitly write out `affine_map`, `dimension`, and `symbol` to guide the compiler.

## Basic example

Here is a basic example of using the affine dialect.
```py
from pydsl.affine import affine_range as arange
from pydsl.frontend import compile
from pydsl.memref import MemRefFactory
from pydsl.type import F32, Index

n = 2200
m = 1800
MemRefF32NM = MemRefFactory((n, m), F32)


@compile(dump_mlir=True, auto_build=False)
def affine_example_implicit(
    v0: Index,
    v1: Index,
    A: MemRefF32NM,
) -> F32:
    b: F32 = 0.0
    for i in arange(v0, v0 + 8, 2):
        A[i, v1 + 5] = b

    return b

```

Here's what it produces

```
#map = affine_map<()[s0] -> (s0 + 8)>
module {
  func.func public @affine_example(%arg0: index, %arg1: index, %arg2: memref<2200x1800xf32>) -> f32 {
    %cst = arith.constant 0.000000e+00 : f32
    affine.for %arg3 = %arg0 to #map()[%arg0] step 2 {
      affine.store %cst, %arg2[%arg3, symbol(%arg1) + 5] : memref<2200x1800xf32>
    }
    return %cst : f32
  }
}
```

While that is what you would usually write, here's the explicit version of the code, for demonstration purposes.
```py
from pydsl.affine import (
    affine_map as am,
    affine_range as arange,
    dimension as D,
    symbol as S,
)
from pydsl.frontend import compile
from pydsl.memref import MemRefFactory
from pydsl.type import F32, Index

n = 2200
m = 1800
MemRefF32NM = MemRefFactory((n, m), F32)


@compile(locals(), dump_mlir=True, auto_build=False)
def affine_example_explicit(
    v0: Index,
    v1: Index,
    A: MemRefF32NM,
) -> F32:
    b: F32 = 0.0
    for i in arange(am(S(v0)), am(S(v0) + 8), 2):
        A[am(D(i), S(v1) + 5)] = b

    return b

```

# Metaprogramming

PyDSL currently provides two ways to perform metaprogramming:
- Use MLIR's transform dialect to modify your source code
    - This is more user-facing, and can be doen to fine-tune the performance of your payload code by annotating it with additional flags for MLIR
- Use the source-to-target macro system to extend the compiler
    - This requires understanding of how the compiler works, and is necessary if you wish to support more dialects

## Transformation

### Defining transform sequence

Like in MLIR, there are two functions you need to define:
- The payload function that will actually run
- The transform sequence function that performs transformations on the payload function

The transform sequence can be defined by creating a function with an `AnyOp` argument, then passing its name into the `@compile` decorator.
Note that `AnyOp` type lowers to `!transform.any_op` type.

```py
def transform_seq(targ: AnyOp):
    ...

@compile(locals(), transform_seq=transform_seq)
def fn(...) -> UInt32:
    ...
```

### Accepting operators as variable

Transformation operators usually require operators as its arguments, such as flagging operations with `tag`. Since there is no way to do that in standard Python, PyDSL has two non-invasive syntax for doing so:

**Defining with `decorate_next` call**

* By calling `decorate_next(...)` with one or more functions, the operator that comes *immediately after* the call will be passed into those functions as the first argument.
* Each function applies attributes to the operator.

In this example, the MLIR ForOp operation that is compiled from the `for` statement is passed into the `tag` function.
```py
decorate_next(tag("tile"))
for arg2 in arange(S(v0)): ...
another_operation # this is not passed in
```

You can think of it like this (Note that this is just pseudocode. You can't actually write this!):
```pseudocode
for_op = compile("for arg2 in arange(S(v0)): ...")
tag(for_op, "tile")
```

The syntax also allows multiple arguments:

```py
decorate_next(tag("tile"), int_attr("set", 2))
for arg2 in arange(S(v0)): ...
```

Here, both `tag("tile")` and `int_attr("set", 2)` are applied to the same `for` operator.

**Defining with `with` statement**
- You can also enclose and flag statements using Python's `with` statement, and setting its context to a transform operation. This has the benefit of allowing multiple operators to be passed into the transformation.

In this example, all operations within the `with` statement are passed into the function as a single syntax tree.
```py
from pydsl.frontend import compile
from pydsl.transform import recursively, tag
from pydsl.affine import (
    affine_map as am,
    affine_range as arange,
    symbol as S,
    dimension as D,
)
from pydsl.memref import MemRefFactory
from pydsl.type import F32, Index

n = 2200
m = 1800
MemRefF32NM = MemRefFactory((n, m), F32)


@compile(locals(), dump_mlir=True)
def affine_example(
    v0: Index,
    v1: Index,
    A: MemRefF32NM,
) -> F32:
    with recursively(tag("hello")):
        b: F32 = 0.0
        for i in arange(am(S(v0)), am(S(v0) + 8), 2):
            A[am(D(i), S(v1) + 5)] = b

    return b

```

This produces
```mlir
#map = affine_map<()[s0] -> (s0 + 8)>
module {
  func.func public @affine_example(%arg0: index, %arg1: index, %arg2: memref<2200x1800xf32>) -> f32 {
    %cst = arith.constant {hello} 0.000000e+00 : f32
    affine.for %arg3 = %arg0 to #map()[%arg0] step 2 {
      affine.store %cst, %arg2[%arg3, symbol(%arg1) + 5] {hello} : memref<2200x1800xf32>
    } {hello}
    return %cst : f32
  }
}
```

### `recursively`

Sometimes, you need a transform operator to be applied to every sub-operator in the same line. This is not possible with just string comment or `with` statements, as they only apply transformation to the top of the syntax tree.

`recursively` lets you do exactly that. It is a special macro that accepts a statement like every other operator, except it splices the overall statement into its constituent sub-operators, then passing each one through a user-defined function.

For example, you can write this to apply `int_attr` to every sub-operator of the overall store statement.
```py
with recursively(int_attr("set", 1)):
    s[j] = (s[j] + r[i]) * A[i, j]

with recursively(int_attr("set", 2)):
    q[i] = (q[i] + p[i]) * A[i, j]
```

The user-defined function, `int_attr("set", 1)`, is a standard Python statement that is interpreted during the compilation process (as PyDSL currently do not support lambda statements). It accepts the spliced-up MLIR operators one-by-one and tags each of them with the integer attribute `"set": 1`.

This is equivalent to writing each sub-operator in the statement on a separate line and introducing them to `int_attr` separately. If used correctly, this macro can help avoid a lot of tedious code duplications.
```py
decorate_next(int_attr("set", 1))
s_j = s[am(D(j))]
decorate_next(int_attr("set", 1))
r_i = r[am(D(i))]
decorate_next(int_attr("set", 1))
sum = s_j + r_i
decorate_next(int_attr("set", 1))
A_res = A[am(D(i), D(j))]
decorate_next(int_attr("set", 1))
mul = sum * A_res
decorate_next(int_attr("set", 1))
s[am(D(j))] = mul
decorate_next(int_attr("set", 2))
q_i = q[am(D(i))]
decorate_next(int_attr("set", 2))
p_i = p[am(D(i))]
decorate_next(int_attr("set", 2))
sum = q_i + p_i
decorate_next(int_attr("set", 2))
A_res = A[am(D(i), D(j))]
decorate_next(int_attr("set", 2))
mul = sum * A_res
decorate_next(int_attr("set", 2))
q[am(D(i))] = mul
```

## Supporting new dialects as macros

Another common way to do metaprogramming in PyDSL is to extend the compiler itself. The macro system is available to add additional functions that can be called in various contexts and transforms itself and surrounding context into raw MLIR.

To put generally, macros implement a method that transforms itself and its surrounding AST context into equivalent MLIR at compile-time by the visitor. They also implement a visitor callback protocol of some kind (such as `CompileTimeCallable` or `HandlesFor`), and the methods of the protocols define how they are converted into MLIR by the main visitor when the macro is traversed. The pairing of visitor callback protocols and macro is what enables users to use macros as if they are just normal Python syntax.

As an example, here's the `HandlesFor` protocol

```py
@runtime_checkable
class HandlesFor(Protocol):
    """
    A protocol where when its instances are inserted as the iterator of a for
    loop, they are responsible for transforming the entire for loop.
    """

    def on_For(visitor: "ToMLIRBase", node: ast.For) -> SubtreeOut:
        """
        Called when the implementer is inserted into a for loop.
        """
```

and the `IteratorMacro` subclass which stipulates the protocol interface.

```py
class IteratorMacro(Macro):
    @abstractmethod
    def on_For(visitor: ToMLIRBase, node: ast.For): ...

    @abstractmethod
    def on_ListComp(visitor: ToMLIRBase, node: ast.ListComp): ...

    @abstractmethod
    def _signature(*args: Any, **varargs: Any) -> None: ...
```

The protocol's callback functions usually take on a specific signature:
- The inputs always include the visitor itself and the AST node being traversed when this macro is called. The visitor is needed to access contextual information such as the statically-analyzed variable stack and to delegate subtree traversals back to it. The AST node is the input to the macro transformation.
- The return type is always SubtreeOut, a typing alias that represents any lowerable wrapper class around MLIR Python binding types. This is the output of the macro transformation.

It is impractical to give a comprehensive overview on every possible use case of macros as each macro exists for different purposes. New subclasses of `Macro` may also need to be defined for significantly different use cases. For more examples, it may be helpful to look at existing examples of macros in `transform.py` or the `on_Call` functions in `type.py` to motivate oneself of this design pattern.

Be careful when using this system, however. Creating a new macro requires understanding the internals of how the compiler works. The macro is responsible for accepting internal intermediate representation and converting it into MLIR via the MLIR Python binding.

### Raising and lowering `SubtreeOut`

When working with compiler code, a recurring theme is the concept of raising and lowering intermediate AST compilation outputs. This is necessary because PyDSL wraps almost all types in MLIR Python binding around a wrapper class to organize common operators and cache attributes on these types as member methods, making them easier to use and analyze.

This introduces an issue in the way that PyDSL interacts with MLIR Python binding, as the binding does not support any of the PyDSL wrapper classes. As such, every wrapper class includes a `lower` and `lower_class` method, which unwraps an instance and the unwrapped value’s MLIR type, respectively. The lowered value may be a tuple of MLIR values; this allows the wrapper class to make use of a collection of MLIR values to represent itself, like a register-passed struct.

```py
@runtime_checkable
class Lowerable(Protocol):
    def lower(self) -> tuple[Value]: ...

    @classmethod
    def lower_class(cls) -> tuple[mlir.Type]: ...

def lower(
    v: Lowerable | type | OpView | Value | mlir.Type,
) -> tuple[Value] | tuple[mlir.Type]:
    """
    Convert a `Lowerable` type, type instance, and other MLIR objects into its
    lowest MLIR representation, as a tuple.

    This function is *not* idempotent.
    """
    ...

def lower_single(
    v: Lowerable | type | OpView | Value | mlir.Type,
) -> Value | mlir.Type:
    """
    lower with the return value stripped of its tuple.
    Lowered output tuple must have length of exactly 1.
    """
    res = lower(v)
    if len(res) != 1:
        raise ValueError(f"lowering expected single element, got {res}")
    return res[0]

def lower_flatten(li: list[Lowerable]) -> list:
    """
    Apply lower to each element of the list, then unpack the resulting tuples
    within the list.
    """
    return reduce(lambda a, b: a + [*b], map(lower, li), [])
```

Lowering of values is facilitated by `lower`, `lower_single`, and `lower_flatten`. Given an input `v`, `lower` calls `v.lower` if `v` is an instance, or `v.lower_class` if `v` is a wrapper class, as such:
- `lower(Index)` should be equivalent to `(IndexType.get(),)`
- `lower(Index(5))` should be equivalent to `(ConstantOp(IndexType.get(), 5).results,)`

`lower_single` is a shorthand that strips the lowered result of its tuple, assuming the tuple wraps a single element.

`lower_flatten` is similar to lower except that it operates over a list of wrapper type values. In addition, it also flattens all the tuples within the list.

There also exists a notion of "raising" where a MLIR Python binding type is wrapped around a PyDSL class. This is usually done by simply passing them as the sole argument of a PyDSL type constructor.

Practically, these functions are often used when PyDSL receives a wrapper value and needs to pass it into an MLIR operator. As an example in the casting function `Int.Float`, both the integer being casted and the target type needs to be in their MLIR representation for `arith.uitofp`. `arith.sitofp` returns a MLIR type which needs to be raised back to a PyDSL type via `target_type`'s constructor (by simply passing it into `target_type(∙)`).

```py
match self.sign:
    case Sign.SIGNED:
        return target_type(
            arith.sitofp(lower_single(target_type), lower_single(self))
        )
    ...
```

### Creating new CallMacro

In a user-facing perspective, CallMacro can be thought of as a function that is called at runtime. However, behind the scene, CallMacro substitutes itself with MLIR representation during compile time whenever a call to it is visited within the AST.


#### Example via `CallMacro.generate`

Here is a basic example of a CallMacro that takes an MLIR node and attaches a
named integer attribute to it.

```py
@CallMacro.generate()
def int_attr(
    visitor: ToMLIRBase,
    mlir: Compiled,
    attr_name: Evaluated[str],
    value: Evaluated[int],
) -> SubtreeOut:
    target = get_operator(mlir)

    if type(attr_name) is not str:
        raise TypeError("Attribute name is not a string")
    target.attributes[attr_name] = IntegerAttr.get(IndexType.get(), value)

    return mlir
```

Here's what the code means:
- `@CallMacro.generate()` is a decorator that takes the `int_attr` function and
converts it into a subclass of `CallMacro`. We'll look at the internals of
`CallMacro` in the next example.
- The first argument that is passed into the macro is always the `ToMLIR`
visitor that is walking the AST. It contains important contextual information
you may need such as the live variables and the stack scopes when the function
is called.
- The arguments that come after are passed in by the user. **It is very
important to know that the type hinting here are mandatory**, as they indicate
how the arguments should be interpreted before it is passed into the function:
    - `Compiled` means that the argument should be visited and compiled by the
    `ToMLIR` visitor before it is passed in. The type is always a `SubtreeOut`
    which is either raw MLIR or wrapper classes that can lower down to raw MLIR.
    - `Evaluated[T]` means that the argument should be evaluated as a CPython
    expression. The expression is passed into an `eval()` function before
    passed in. This allows any arbitrary Python expression to be written, but
    usually this is for cases where you want the user to pass in string
    literals, numeric literals, or lambda functions.
        - The `T` type argument in `Evaluated[T]` hints the evaluated Python
        type that is passed into the function. It does not affect any runtime
        behavior and is used only by type checkers.
    - `Uncompiled` means the argument should be kept as an AST. This is useful
    for cases where you may want to delay the visiting process or want high
    degree of granularity in the way that the argument is compiled by your macro.
        - If you want to visit this AST, use `visitor.visit` where `visitor` is
        the `ToMLIR` instance passed in as the first argument of the function.
        - Refer to Python's `ast` module documentation on how to use AST nodes:
        https://docs.python.org/3/library/ast.html.
- The macro must return a `SubtreeOut`, which can be thought of as a partial
compilation result for a subtree of the program. This includes MLIR Python
binding objects such as Values or Operations.

You can also specify the `method_type` to create call macros that behave like
Python instance/class methods.
- See Python documentation of `pydsl.macro.MethodType`.
- See examples at `tests/e2e/test_macro.py`.
    - `test_method_instance` to `test_method_static`.

#### Example via subclassing CallMacro

You can also define a call macro by subclassing `CallMacro`, although using
`CallMacro.generate` is recommended when possible. Here's what
using a subclass would look like:

```py
class GetLoopMacro(CallMacro):
    @staticmethod
    def signature() -> inspect.Signature:
        def f(visitor: ToMLIRBase, target: Compiled, index: Evaluated): ...
        return inspect.signature(f)

    @staticmethod
    def __call__(visitor: ToMLIRBase, target: Compiled, index: Evaluated) -> OpView:
        if hasattr(target, "loops"):
            return target.loops[index]
        return target.operation.results[index]

get_loop = GetLoopMacro()
```

Conceptually, this is identical to definining macros via `CallMacro.generate`.
All subclasses are required to define 2 methods:
- `signature()` must return the signature of the macro function.
- `__call__` is the macro function itself.

Note that currently, for `CallMacro`s, we actually create an instance of the
class, while for `IteratorMacro`s, we still work with the class objects
directly. The main reason for using instances of `CallMacro`s is that we can
name the call function `__call__` instead of `__init__`.

### Creating new IteratorMacro

Not all macros in PyDSL are called at compile-time. Functions with complicated behaviors in Python are difficult to translate into MLIR, such as the built-in `range`. In a typical Python `for` loop iterating over a range of numbers, the `range` function is often used, returning an iterable `range` object in CPython that is consumed by the `for` statement. This behavior is not as pragmatic as directly using a `scf.ForOp` to iterate over a range of numbers in MLIR. This is where `IteratorMacro` comes in to effectively emulate the behavior of a function returning any kind of iterator.

Here is how `range` is actually defined in PyDSL by subclassing `IteratorMacro`:

```py
class range(IteratorMacro):
    def on_For(visitor: ToMLIR, node: ast.For) -> scf.ForOp:
        iter_arg = node.target
        iterator = node.iter

        start = arith.ConstantOp(IndexType.get(), 0)
        step = arith.ConstantOp(IndexType.get(), 1)

        args = [lower_single(visitor.visit(a)) for a in iterator.args]

        match len(args):
            case 1:
                (stop,) = args
            case 2:
                start, stop = args
            case 3:
                start, stop, step = args
            case _:
                raise ValueError(
                    f"range expects up to 3 arguments, got {len(args)}"
                )

        for_op = scf.ForOp(...)

        return for_op
```

As an `IteratorMacro`, `range` is required to implement its macro behavior whenever it is put in as the iterator of a `for` loop (in which case `on_For` is called). Mind that `range` not only substitutes itself, but also the entire `for` loop and the contents of its body.

Emphasis should be placed on the word "emulate". This does not reflect how CPython actually uses `range` whenever it is put into a `for` loop. Rather, we emulate that behavior as far as the final result is concerned, without regard as to the type of `range`, its callability, its return value, its performance, and whether it pushes a new frame into the call stack. All implementation details may differ. The benefit of defining `range` as such is simplicity of implementation and a clean MLIR output, but at the cost of generalizability.

# Autotuning

Like Mojo and Triton, PyDSL has support for autotuning. This means that it can select from a series of compiler options and parameters to find the configuration that provides the best performance.
The way that this is done is by providing a decorator `@autotune(...)` which allows the user to search over a space of configurations to find the one which provides the best runtime.

A configuration is represented by the class of type `Config`, which holds the parameters and options the compiler will use when compiling your function. This class has three fields.
- `Env`: A dictionary mapping variable names (represented as strings) to values. These values can be used anywhere in PyDSL, including the function body, or within the transformation sequence
- `Settings`: A dictionary mapping key-word arguments of the `@compile` decorator (represented as strings) to the values to be passed in during compilation
- `Args`: the data that will be passed to the function when evaluating its performance (represented as a list of values)

The argument to the autotune is an object of type `AutotuneConfigurations`, which is a very thin wrapper around a list of `Config`. Each config in the list is compiled/ran and performance is measured. After this is complete, the function is set to the fastest one measured.

Note that `AutotuneConfigurations` provides a number of utility functions to help users build lists of configurations. These include primitives to do cartesian products, zipping and concatenation. These make it easy to produce comprehensive search spaces.

# Inter-operability with Triton

PyDSL supports inter-operability with Triton, i.e. it is possible to call a Triton function from within a PyDSL function. This works by first compiling the Triton function to triton MLIR (ttir), converting it to the linalg dialect using the triton-adapter-opt tool, and then combining it with the MLIR generated by PyDSL.

## Requirements

in order to use the inter-op feature of PyDSL, you require a configured Triton environment, as well as the tool triton-adapter-opt from [triton-ascend](https://gitee.com/ascend/triton-ascend). 

### Building triton-ascend and adding triton-adapter-opt to your PATH:

#### Using pip:

For people who don't want to build all of triton-ascend, it is possible to extract only the triton-adapter-opt binary using the Python wheel for triton-ascend. Since this is designed for Python version 3.11 and we support Python 3.12, you can use the following commands to extract the triton-adapter-opt binary, add it to your PATH, and then remove the wheel. Make sure to execute this command wherever you want to have the triton-adapter-opt binary stored (such as the PyDSL root directory):
```bash
mkdir triton_bin
cd triton_bin
pip download triton-ascend --python-version=3.11 --no-deps
unzip triton_ascend*.whl -d extracted_wheel
cp extracted_wheel/triton/backends/huawei/triton-adapter-opt .
rm -rf extracted_wheel triton_ascend*.whl
cd ..
export PATH=$(pwd)/triton_bin:$PATH
```

#### Building from source:

Build instructions for building triton-ascend can be found here:  https://gitee.com/ascend/triton-ascend.
After building, you need to add the triton-adapter-opt tool to you path, i.e.
```bash
export PATH=$(find $ASCEND_DIRECTORY/build -name "triton-adapter-opt" | head -n 1):$PATH
```
changing $ASCEND_DIRECTORY to point to the location of your triton-ascend directory.

## Examples/Usage

The basic usage is to simply call an `@triton.jit` function from within a PyDSL function:

```python
@triton.jit
def kernel(a_ptr, ...): # signature
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)
    pid_z = tl.program_id(axis=2)
    # triton implmentation goes here

@compile()
def pydsl_func():
    # pydsl code goes here
    for i in arange(x_size):
        for j in arange(y_size):
            for k in arange(z_size):
	            kernel(a_ptr, ..., i, j, k) 
    # more pydsl code
```
Note how in addition to the parameters defined by the kernel, you can also provide 3 extra parameters (x, y, z), which define the grid index that you want to call the kernel with. These are `Index` arguments which need to be added for each axis that is defined within the Triton kernel. A more detailed example of how to use the interop feature can be found in [examples/triton_interop.py](/examples/triton_interop.py).

For calling a triton function with a constexpr parameter, a `Number` object must be passed in, i.e.
```python
@triton.jit
def kernel(... a: tl.constexpr, ...):
	# triton implementation goes here

@compile()
def pydsl_func():
	# pydsl code
	a: Number = 64
	kernel(..., a, ...)
    # or
    kernel(..., 64, ...)
```

## Limitations

- Since this feature uses triton-adapter-opt to convert the triton dialect to the linalg dialect, it inherits the limitations of Triton ascend. This includes the fact that not all ops are currently supported for all data types, and uint is currently not supported at all. Current support is detailed [here]([docs/sources/python-api/outline.md · Ascend/triton-ascend - Gitee](https://gitee.com/ascend/triton-ascend/blob/master/docs/sources/python-api/outline.md)) 

- PyDSL `Tensor` types are currently not supported; all arrays passed in must be `MemRef`s.

- `MemRef` shapes must always be 1-dimensional `DYNAMIC` arrays when they are passed into a Triton function.

- triton-adapter-opt only supports lowering a single function. Triton inlines function calls by default, but if a function is called that has `noinline=true` then the compilation will fail