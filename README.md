# Python-to-MLIR compiler

# Background

This project aims to provide an interface between Python and MLIR with the following design goals:
- Simple: The project should be easy to maintain with a thin layer of translation
- Reliable: The project needs to have a healthy amount of maintenance and contributors, such that we can rely on the project in the long run. Alternatively, we need to have full control over the project.
- Pythonic: The syntax for as much of the API as possible should be legal Python, as well as concise as a precise language permits.
    - This is motivated by the fact that much of the machine learning community uses Python
- Domain-specific: The aim is to use the language to write a specific class of difficult-to-optimize algorithms such as LU and Jacobi and exploit MLIR's tools to optimize them. 

Simplicity and being Pythonic are conflicting goals. Thus, we have to aim for a very strict subset of Python that enforces static typing, disallow variables modified within a scope to be used outside of it, 

We are motivated to create this project in-house because existing third-party tools do not meet the above requirements:
- Nelli/mlir-python-extras does not meet the simplicity requirement. As of 01/16/2024, it contains roughly 8000 lines of code. It has only one active contributor and there is no plan to integrate the project as part of the official MLIR upstream.

# Instruction for installation

## Prerequisite

Currently, the only dependency are:
- Python 3.12<=. You can easily use anaconda or miniconda to acquire this version and activate it as your environment.
- MLIR's Python binding. Refer to its wiki for how to install and use. By the time you finish installing it, the Python binding's package folder should be on your PYTHONPATH and you should be able to import and use `mlir`'s modules.

## Import the compiler

In Python, you can only import Python files that are on your `sys.path`. This means that the file must be:
- Installed as a package.
    - This is not a viable option for now as there isn't the code in place to build the compiler as a package
- Be within a package folder right next to your file.
    - This is what I would do for now because it's easy.

In our case, just place your Python file next to the `python_compiler` folder.

- In your `PYTHONPATH` environment variable. 
    - NOTE: this will be broken if your ever move the folder around. Not recommended unless you are sure your file will stay in place.
- Modify the `sys.path` at runtime.

E.g. this is done for the files in the `examples` folder to include `..` (relative to the location of the code file being run) before any `import` statement is made for the `python_compiler`:
```py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / '..'))

from python_compiler.compiler import <whatever>
# etc.
```

# Boilerplate code

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

See examples in the `examples` folder.

The `@compile(locals())` will grab all variables defined before the function, and convert the function being decorated into MLIR.
- this means that all type imports and type definitions must be done before and outside the function

# Features

## Arithmetic

Addition, subtraction, multiplication, division are supported for float-likes

Addition, subtraction are supported for integer-likes (such as index)

No integer multiplication/division
- This is simply because these operators are complicated in MLIR, and behavior differs for signed and unsigned integer. I might add it soon if I have time.

No modulo

## For loop

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

## Memref

Define the size of your memref type outside of the function:
E.g. this is memref<40x40xf64>
```py
memref_f64 = MemRefFactory((40, 40), F64)
```

E.g. of using it
```py
# suppose this is the function signature: def lu(v0: index, arg1: memref_f64) -> i32:
# so arg1 is a memref<40x40xf64>

for arg4 in range(arg3):
    # Load from arg1 to v4
    # NOTE: you must put in a tuple as the index: arg1[arg2, arg4]
    # Currying the index arg1[arg2][arg4] is not allowed!
    # arg2 and arg4 must be `index` type
    v4 = arg1[arg2, arg4] 

    # do other stuff
    v5 = arg1[arg4, arg3]
    v6 = v4 * v5
    v7 = arg1[arg2, arg3]
    v8 = v7 - v6

    # Save to arg1 from v8
    arg1[arg2, arg3] = v8
```

# Other restriction

As mentioned, this is a very strict subset of Python, and many features are either not yet implemented or too non-trivial to implement. This section goes over common Python features you cannot use.

## Static typing only

When you declare a variable while assigning it a constant, it MUST be in this format:
```py
var_name: <type> = <constant, must be integer or float>
```
If you use float, you have to always write the decimals. I will look into doing auto casting some time down the line.

```py
my_float: f32 = 5.0 # NOT 5!
```

MLIR and the compiler will take care of rest of the type evaluation. No type hinting should be used if you are assigning a variable to another

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
var1 = var2 = c5
```

## Only for statements; No if/while statements

This is simply not yet implemented.

## No out-of-scope usage

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

## Don't use variables outside a function, unless it's a type used for type hinting

The mechanisms for including and using outside variables are complicated and used only for specific purposes such as type hinting (types must be defined outside the function as type hinting is needed while the function itself is being defined). They are treated very differently from variables in your function.

## All functions must return something

Void type is currently not supported. You must return a variable that has the same type as the function's type hinting

## Only one function can be compiled for now

Do not call any function. Do not define any sub-functions within a function.

## If it looks fancy, it's probably not supported

No list comprehension, no dictionary, no tuple, no multiple return type, no list, et cetera ad nauseum. We want to keep things simple for now.