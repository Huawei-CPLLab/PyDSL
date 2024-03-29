# PyDSL: A Python to MLIR Compiler

## Background

This project aims to provide an interface between Python and MLIR with the following design goals:
- Simple: The project should be easy to maintain with a thin layer of translation.
- Reliable: The project needs to have a healthy amount of maintenance and contributors, such that we can rely on the project in the long run. Alternatively, we need to have full control over the project.
- Pythonic: The syntax for as much of the API as possible should be legal Python, as well as concise as a precise language permits.
    - This is motivated by the fact that much of the machine learning community uses Python.
- Domain-specific: The aim is to use the language to write a specific class of difficult-to-optimize algorithms such as LU and Jacobi and exploit MLIR's tools to optimize them. 

Simplicity and being Pythonic are conflicting goals. Thus, we have to aim for a very strict subset of Python that enforces static typing and to disallow variables modified within a scope to be used outside of it.

This work was presented at the [Open MLIR Meeting](https://mlir.llvm.org/talks/) on December 21st, 2023. The video and slides can be found here:
https://www.youtube.com/watch?v=nmtHeRkl850

## Build Instructions

### 1. Clone PyDSL

```sh
git clone --recursive https://github.com/Huawei-PTLab/PyDSL.git
cd PyDSL
```

### 2. Install MLIR with Python Bindings

These install commands are based off the [MLIR Getting Started Guide](https://mlir.llvm.org/getting_started/) and the [MLIR Python Bindings Webpage](https://mlir.llvm.org/docs/Bindings/Python/).
If you already have a relatively recent version of MLIR built with Python Bindings, this step can be skipped.

```sh
mkdir build
cd build
cmake -G Ninja ../llvm-project/llvm \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_BUILD_EXAMPLES=ON \
    -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DPython3_EXECUTABLE="/usr/bin/python3"
ninja check-mlir
export PYTHONPATH=$(pwd)/tools/mlir/python_packages/mlir_core
```

These commands build MLIR with Python Binding, runs check MLIR to ensure the build was successful, and updates the `PYTHONPATH` variable to the mlir_core.

If your `python3` executable is located in another location (or if you are using conda), just change the `Python3_EXECUTABLE` flag to point to the correct location.

NOTE: This sets the `PYTHONPATH` for your current terminal session. Closing and reopening the terminal will require you to rerun the `export PYTHONPATH` command again.

### 3. Running an Example

The following is how to run a simple example from the test folder. From the root directory, type:
```sh
python3 test/simple.py
```
The expected output of this should look something like:
```mlir
module {
  func.func public @lu(%arg0: f64, %arg1: memref<40xf64>) -> index {
    %c40 = arith.constant 40 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c1_0 = arith.constant 1 : index
    scf.for %arg2 = %c0 to %c40 step %c1_0 {
      %1 = memref.load %arg1[%arg2] : memref<40xf64>
      %2 = arith.addf %1, %arg0 : f64
      memref.store %2, %arg1[%arg2] : memref<40xf64>
    }
    %0 = arith.addi %c1, %c40 : index
    return %0 : index
  }
}
```

There is also a version of this testcase that uses affine loops. Type:
```sh
python3 test/simple-affine.py
```
The expected output of this should look something like:
```mlir
#map = affine_map<(d0) -> (d0)>
module {
  func.func public @lu(%arg0: f64, %arg1: memref<40xf64>) -> index {
    %c40 = arith.constant 40 : index
    %c1 = arith.constant 1 : index
    affine.for %arg2 = 0 to #map(%c40) {
      %1 = affine.load %arg1[%arg2] : memref<40xf64>
      %2 = arith.addf %1, %arg0 : f64
      affine.store %2, %arg1[%arg2] : memref<40xf64>
    }
    %0 = arith.addi %c1, %c40 : index
    return %0 : index
  }
}
```

### Requirements

- Python 3.11 or above. anaconda or miniconda can be used to acquire this version by activating it as your environment.
    - You will also need `numpy` and `pybind11` installed.

## Known Issues

Some of the testcases are not working with the submodule version of LLVM. Working on supporting:
- test/heat-affine-transform.py
- test/lu-affine-transform.py
- test/seidal-affine-transform.py

## Using PyDSL

### Import PyDSL

In Python, you can only import Python files that are on your `sys.path`. This means that the file must be:
- Installed as a package.
    - This is not a viable option for now as there isn't the code in place to build the compiler as a package
- Be within a package folder right next to your file.
    - This is currently the easier option. Just place your Python file next to the `pydsl` folder.

- In your `PYTHONPATH` environment variable, i.e. add `path/to/pydsl` to `PYTHONPATH`.
    - NOTE: this will be broken if your ever move the folder around. Not recommended unless you are sure your file will stay in place.
- Modify the `sys.path` at runtime.

E.g. this is done for the files in the `test` folder to include `..` (relative to the location of the code file being run) before any `import` statement is made for the `pydsl`:
```py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / '..'))

from python_compiler.compiler import <whatever>
# etc.
```

### Boilerplate code

```py
# Types are stored in the type.py in python_compiler. Import what you need.
from pydsl.type import UInt32, F32, Index

# Always import this
from pydsl.frontend import compile

# Here is how you construct a memref that is ?x?xf32
memref_f32 = MemRefFactory((DYNAMIC, DYNAMIC), F32)


@compile(locals())
# Replace <type> with the type of the argument that you imported from python_compiler.type, same for <return_type>.
# All type hinting shown are MANDATORY! 
# Do not use any other Python argument features! No *args! No **kwargs!
def my_function(arg_name_0: <type>, arg_name_1: <type>) -> <return_type>:
    # Your Python code
```

This code will convert your `my_function` function into an MLIR function.

See examples in the `test` folder.

The `@compile(locals())` will grab all variables defined before the function, and convert the function being decorated into MLIR.
- this means that all type imports and type definitions must be done before and outside the function

## Features

### Arithmetic

Addition, subtraction, multiplication, division are supported for float-likes

Addition, subtraction are supported for integer-likes (such as index)

No integer multiplication/division
- This is simply because these operators are complicated in MLIR, and behavior differs for signed and unsigned integer.

No modulo

### For loop

Only `range` can be iterated.

E.g.
```py
# range can be:
# - range(upper_bound), which will iterate each integer between 0 (inclusive) to upper_bound (exclusive)
# - range(lower_bound, upper_bound), which will iterate each integer between lower_bound (inclusive) to upper_bound (exclusive)
# - range(lower_bound, upper_bound, step), which is equivalent to scf.for %iter_arg = %lower_bound to %upper_bound step %step { ... }
for i in range(5):
    # code

for i in range(0, 5):
    # code

for i in range(0, 5, 1):
    # code
```

### Affine
affine maps, ranges, dimensions, and symbols are supported in PyDSL.

```py
from pydsl.affine import            \
    affine_range as arange,         \
    affine_map as am,               \
    dimension as D,                 \
    symbol as S                     
```

```py
for i in arange(0, 5, 1):
    # code
for i in arange(0, D(a)):
    # code
for i in arange(S(t)):
    # code
for i in arange(D(a)):
    # code
# An example affinemap defined with dimension i to index over an array.
arg1[am(D(i))] = arg1[am(D(i))] + v0
```

### Memref

Define the size of your memref type outside of the function:
E.g. this is memref<40x40xf64>
```py
memref_f64 = MemRefFactory((40, 40), F64)
```

E.g. of using it
```py
# Suppose this is the function signature: def lu(v0: index, arg1: memref_f64) -> i32:
# so arg1 is a memref<40x40xf64>

for arg4 in range(arg3):
    # Load from arg1 to v4
    # NOTE: you must put in a tuple as the index: arg1[arg2, arg4]
    # Currying the index arg1[arg2][arg4] is not allowed!
    # arg2 and arg4 must be `index` type
    v4 = arg1[arg2, arg4] 

    # Do other stuff
    v5 = arg1[arg4, arg3]
    v6 = v4 * v5
    v7 = arg1[arg2, arg3]
    v8 = v7 - v6

    # Save to arg1 from v8
    arg1[arg2, arg3] = v8
```

### Transform Ops

Transforms within PyDSL are generated by defining a transform sequence function which is then passed into the compile flag. The function to be converted to MLIR can then be tagged with various flags to determing the location of the transform ops.

For example, see the following from `test/simple-transform.py`:
```python
from pydsl.transform import tag, loop_coalesce, match_tag as match

MemRefF64 = MemRefFactory((40,50,), F64)

def transform_seq(targ: AnyOp):
    loop_coalesce(match(targ, 'coalesce'))

@compile(locals(), transform_seq=transform_seq, dump_mlir=True, auto_build=False)
def simple_case(v0: F64, arg1: MemRefF64) -> Index:
    N: Index = 40
    M: Index = 50
    c1: Index = 1

    """@tag("coalesce")"""
    for i in range(N):
        for j in range(M):
            arg1[i, j] = arg1[i, j] + v0
    
    return c1 + N + M
```
Note that tags are placed in a docstring within the function iteslf and are read while converting to MLIR to determine tag locations. The resulting MLIR output for the transform sequence would look like this:
```mlir
module {
  func.func public @simple_case(%arg0: f64, %arg1: memref<40x50xf64>) -> index {
    %c40 = arith.constant 40 : index
    %c50 = arith.constant 50 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c1_0 = arith.constant 1 : index
    scf.for %arg2 = %c0 to %c40 step %c1_0 {
      %c0_1 = arith.constant 0 : index
      %c1_2 = arith.constant 1 : index
      scf.for %arg3 = %c0_1 to %c50 step %c1_2 {
        %2 = memref.load %arg1[%arg2, %arg3] : memref<40x50xf64>
        %3 = arith.addf %2, %arg0 : f64
        memref.store %3, %arg1[%arg2, %arg3] : memref<40x50xf64>
      }
    } {coalesce}
    %0 = arith.addi %c1, %c40 : index
    %1 = arith.addi %0, %c50 : index
    return %1 : index
  }
  transform.sequence  failures(propagate) {
  ^bb0(%arg0: !transform.any_op):
    %0 = transform.structured.match attributes {coalesce} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.loop.coalesce %0 : (!transform.any_op) -> !transform.op<"scf.for">
  }
}
```

## Restrictions

As mentioned, this is a very strict subset of Python, and many features are either not yet implemented or too non-trivial to implement. This section goes over common Python features you cannot use.

### Static typing only

When you declare a variable while assigning it a constant, it MUST be in this format:
```py
var_name: <type> = <constant, must be integer or float>
```
If you use float, you have to always write the decimals. I will look into doing auto casting some time down the line.

```py
my_float: f32 = 5.0 # NOT 5!
```

MLIR and the compiler will take care of rest of the type evaluation. No type hinting should be used if you are assigning a variable to another.

```py
var1 = var2
```

or if you are getting a value that is calling a MLIR operation behind the scene
```py
# var1 is actually being assigned the memref.LoadOp behind the scene. MLIR will take care of var1's type for you.
var1 = my_memref[12, 12]
```

For now, do not use a constant as-is in an expression, as the compiler cannot yet infer its type.
```py
# Don't
var1 = var2 + 5 

# Do
c5 = 5
var1 = var2 + c5
```

### Only for statements; No if/while statements

If and while statements are not yet implemented.

### No out-of-scope usage

If a variable enters a scope and is modified within that scope, you cannot use it once it leaves the scope. In our case, scope refers to a `for` loop.

E.g.
```py
a = 5 # a is defined outside of the scope

# this is our scope
for i in range(50):
    a = 10 # a is modified within the scope
    b = a + 1 # this is fine

b = a # NOT ALLOWED!
```

This is because the compiler currently do not support `scf.YieldOp` as supporting it requires variable usage analysis. A prototype visitor class has been successfully made, but for the sake of simplicity it is currently not part of the compiler. 

This does mean that a lot of code cannot be written using this compiler, but YieldOp is usually not needed for our domain-specific usage.

### Don't use variables outside a function, unless it's a type used for type hinting

The mechanisms for including and using outside variables are complicated and used only for specific purposes such as type hinting (types must be defined outside the function as type hinting is needed while the function itself is being defined). They are treated very differently from variables in your function.

### All functions must return something

Void type is currently not supported. You must return a variable that has the same type as the function's type hinting

### Only one function can be compiled for now

Do not call any function. Do not define any sub-functions within a function.

### If it looks fancy, it's probably not supported

No list comprehension, no dictionary, no tuple, no multiple return type, no list, et cetera ad nauseum. We want to keep things simple for now.
