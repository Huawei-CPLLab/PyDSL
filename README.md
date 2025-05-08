# PyDSL: A Python to MLIR Compiler

This project aims to provide an interface between Python and MLIR with the following design goals:
- **Simple**. The project should be easy to maintain with a thin layer of translation.
- **Pythonic**. The syntax for PyDSL should be as close to legal Python as possible, while maintaining the precision of MLIR.

Simplicity and being Pythonic are conflicting goals. In order to maintain simplicity, we have to aim for a very strict subset of Python that benefits domain-specific applications such as affine programming with little if-else statements. As such, certain programs will be easy to write while others will be nearly infeasible.

# Presentations
PyDSL has been presented at the following venues:

- [Open MLIR Meeting](https://mlir.llvm.org/talks/) on December 21st, 2023: 📊 [**Slides**](https://mlir.llvm.org/OpenMeetings/2023-12-21-PyDSL.pdf) | 🎞️ [**Video**](https://www.youtube.com/watch?v=nmtHeRkl850)
- [2024 LLVM Developers' Meeting](https://llvm.swoogo.com/2024devmtg): 📊 [**Slides**](https://github.com/Huawei-CPLLab/PyDSL/blob/main/PyDSL%20-%20LLVM%20Conference%202024.pdf) | 🎞️ [**Video**](https://www.youtube.com/watch?v=iYLxgTRe8TU)  
- 2025 January PyDSL 2.0 Announcement: 🎞️[**Video**](https://youtu.be/b6Q-KgqCSoU)

# Usage

Refer to the user documentation here: [docs/usage.md](docs/usage.md).

## PyDSL at a glance

```py
import numpy as np
from pydsl.frontend import compile
from pydsl.affine import affine_range as arange
from pydsl.memref import DYNAMIC, MemRefFactory
from pydsl.type import Index, UInt64

MemRef64 = MemRefFactory((DYNAMIC, DYNAMIC), UInt64)


@compile(dump_mlir=True)
def my_function(size: Index, m: MemRef64) -> MemRef64:
    o = size // 2

    for i in arange(size):
        m[1, i] = o
        m[i, i] = i + o

    return m


arr = np.zeros((8, 8), dtype=np.uint64)

print(my_function(8, arr))

```

This code will convert your `my_function` function into an MLIR function:
```mlir
module {
  func.func public @my_function(%arg0: index, %arg1: memref<?x?xi64>) -> memref<?x?xi64> {
    %c2 = arith.constant 2 : index
    %0 = index.floordivs %arg0, %c2
    affine.for %arg2 = 0 to %arg0 {
      %1 = arith.index_castui %0 : index to i64
      affine.store %1, %arg1[1, %arg2] : memref<?x?xi64>
      %2 = index.add %arg2, %0
      %3 = arith.index_castui %2 : index to i64
      affine.store %3, %arg1[%arg2, %arg2] : memref<?x?xi64>
    }
    return %arg1 : memref<?x?xi64>
  }
}
```

The function will return the following array:
```py
(array([[ 4,  0,  0,  0,  0,  0,  0,  0],
       [ 4,  5,  4,  4,  4,  4,  4,  4],
       [ 0,  0,  6,  0,  0,  0,  0,  0],
       [ 0,  0,  0,  7,  0,  0,  0,  0],
       [ 0,  0,  0,  0,  8,  0,  0,  0],
       [ 0,  0,  0,  0,  0,  9,  0,  0],
       [ 0,  0,  0,  0,  0,  0, 10,  0],
       [ 0,  0,  0,  0,  0,  0,  0, 11]], dtype=uint64),)
```

See more examples in the `examples` or `tests/e2e` folder.

# Instruction for installation

Follow through all of the subsections below to install and use PyDSL.

## Prerequisites

Before performing any installation, you need:
- An environment with Python 3.11 or greater. All requirements can be installed via `pip install -r requirements.txt`
- MLIR built with Python binding or a clone of `llvm-project`
  - a `llvm-project` git submodule is provided in this repo for your convenience.

## Installing MLIR Python binding

If you already have a relatively recent version of MLIR built with Python bindings, this step can be skipped.

The [official documentation](https://mlir.llvm.org/docs/Bindings/Python/#building) has the most definitive instruction on installing the binding. A git submodule of `llvm-project` is also available for your convenience.

Here we will summarize the process for a basic setup:

Once this git repository is cloned, go to the `llvm-project` submodule and run these commands to build MLIR with Python binding:

```sh
mkdir build
cd build
cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_BUILD_EXAMPLES=ON \
    -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DPython3_EXECUTABLE="$(which python3)" # If you are using something like a virtual environment, make sure it is active
ninja check-mlir
export PYTHONPATH=$(pwd)/tools/mlir/python_packages/mlir_core
```

> 📌 **NOTES**:
>- If your `python3` executable is located in another location (or if you are using conda), change the `Python3_EXECUTABLE` flag to point to the correct location. If you are not sure, use the result of `which python3`.
>- This sets the `PYTHONPATH` for your current terminal session. Closing and reopening the terminal will require you to rerun the `export PYTHONPATH` command again.

To confirm you installed the binding properly, the Python binding's package folder should be on your `PYTHONPATH` and you should be able to import and use the binding by importing `mlir` in the Python REPL.

## Installing PyDSL

There are 2 methods to install PyDSL:
- **Quick editable install**: This method lets you edit the source code of PyDSL as you are using it. Ideal for PyDSL development.
- **Building a wheel**: This method packages PyDSL into a standalone wheel that is suitable for deployment.

## Quick editable install

If you don't want to build PyDSL and want to develop PyDSL as you are using it, you can just perform an editable package install:
1. Go to the root of this project (same folder as `pyproject.toml`)
2. Run `pip install -e .`
3. Run `pip list` and check that it is indeed an editable package:

```
Package    Version Editable project location
---------- ------- ------------------------------------------
...
pydsl      0.0.1   <your PyDSL location>
...
```

`Editable project location` should be a field for the entry on `pydsl`. It should be pointing to this project folder.

## Building a wheel

This project uses Hatch as its project manager. Refer to hatch's documentation on installing it: https://hatch.pypa.io/1.12/install/#pip.

If you are running any of the commands below for the first time, or if you are left in a dirty environment state, run

```sh
hatch env prune
```

To generate wheel, run

```sh
hatch build
```

The wheel will be located in `dist/`.

To remove all the generated artifacts, run

```sh
hatch clean
```

# Development

While not mandatory, it is recommended that you use Hatch (the project manager of PyDSL) to aid development. Refer to Hatch's documentation on installing it: https://hatch.pypa.io/1.12/install/#pip. Hatch will automatically install and use the right Python enviornment for various development actions.

## Testing

To test the code, make sure to do a build, prune your environment, then run `hatch test`
```sh
hatch build
hatch env prune
hatch test
```

If you do not have Hatch, you can also install `pytest` and run `python3 -m pytest .` at the top of this project. Be wary of what Python version and packages `pytest` is using for its tests!

## Formatting

To format the code, run
```sh
hatch fmt -f
```
> ⚠️ **WARNING:**
> DO NOT RUN `hatch fmt` WITHOUT THE `-f` FLAG! This will cause unsafe changes which breaks any test or examples that uses docstring directives.

If you do not have Hatch, you can also install and use Python formatters such as `black` or `ruff`.

## Polymorphous

PyDSL is capable of activating polyhedral code generation.  The polyhedral source code is [open-sourced](https://github.com/kaitingwang/llvm-project/commit/9697b2a8c21c419b9d1a23760e4565b512889a3f).
