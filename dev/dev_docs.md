The purpose of this document is to give you a high-level overview of the
internals of PyDSL, and some tips for developing the compiler.
This document assumes you have already read some of the user-facing
documentation of PyDSL, and are thus familiar with how it works from a user's
perspective.

# Installation

[build.md](/dev/build.md) has a condensed version of all the steps needed for
installation.

# Organizational structure of source files

`frontend.py` is the file in which PyDSL interacts with non-PyDSL and
non-MLIR Python Binding things.
It defines different targets, it has code for running shell commands for
lowering from MLIR binding types to an object file, it has code for interacting
with the surrounding (non-PyDSL) Python, like the `@compile` decorator, etc.

`compiler.py` mostly has code for parsing the AST.
The main important part of `compiler.py` is the `ToMLIR` class, which is a
subclass of `ast.NodeVisitor`, and it is the object responsible for parsing the
AST.
If you need to add new *syntax* to PyDSL, you will most likely do it by
modifying `ToMLIR` and adding a new `visit_{CLASS_NAME}` method (see the
documentation of the built-in `ast` module).
Note that ideally, the visitor should not do much logic other than parsing the
AST.
It should instead call functions defined in other files for each specific
operation.
For example, when the visitor encounters a return statement, it doesn't do
too much with it directly; instead, it calls the `on_Return` method of the
current scope stack.

`protocols.py` defines different [protocols](https://typing.python.org/en/latest/spec/protocol.html)
and some functions closely related to these protocols.
A protocol in Python is similar to an Interface in Java, and it describes
methods that classes implementing the protocol should have.
Classes are considered to implement a protocol even if they don't explicitly
subclass it (maybe you have to put `@runtime_checkable` in the correct places
for this, not sure).
For example, `isinstance(tensor, CompileTimeSliceable)` returns true when
`tensor` is a `pydsl.tensor.Tensor` object, because `Tensor` implements
`on_getitem` and `on_setitem`.
The `on_Return` method mentioned above works similarly, there is a `Returnable`
protocol defined in `protocols.py`.

The above are the most general files used in PyDSL.
The rest of the files generally implement a specific MLIR dialect or a more
specific feature of PyDSL.

# General control flow of PyDSL

`frontend.py` creates an MLIR context, any MLIR Python objects created within
this context will be added to it.
So for operations with side effects but no return value, you don't have to
worry about storing them anywhere, since they are added to the context (which
is used to generate the output MLIR) automatically.

The visitor goes through the AST and converts every subtree to a PyDSL
representation.
It also has a scope stack and keeps track of what values are assigned to each
of the variables.
Generally, each variable defined by the user should correspond to some PyDSL
object (some type defined by PyDSL), never an MLIR binding object directly,
and rarely a normal built-in Python type.

## PyDSL types

PyDSL has many types whose purpose is to be a wrapper around an MLIR binding
type.
Examples of such types include `Int`, `Float`, `Index`, `Bool`, `MemRef`,
`Tensor`, `Tuple`, etc.

The way these types generally work is as follows.
They store information about the type of the object (e.g. bitwidth for `Int`,
shape and element_type for `Tensor`) and a variable called `value`, which is a
reference to the MLIR Python binding object.
This value will usually be something like "what operation in MLIR outputs this
value".

The `__init__` method of such a PyDSL type will usually expect an MLIR Python
binding object as its only argument.
Casting between PyDSL types has limited support.
`type.py` defines some things related to casting with `on_Call` and
`Supportable`, which works okay for now but might be improved in the future.
These allow you to do casting using the normal Python casting syntax, but it
will actually call a method in the class you are casting *from*, not the
`__init__` method of the class you are casting to directly.
`Tensor` and `MemRef` do not currently support casting.

You should use PyDSL types instead of MLIR types whenever possible.
- You can call `lower_single` and `lower` for converting to an MLIR type.
- For most MLIR operations, you should only convert to MLIR types when you are
passing in the arguments to the operation, not earlier.

# Writing test cases

Most of our tests are in `tests/e2e`.
- Specifically, this folder has "end-to-end" test cases, which means we write
PyDSL, lower it all the way down to a binary, then execute the binary to see if
it gives the correct result.

There is also `tests/unit` for testing smaller parts of PyDSL that can be
tested without going through the entire compilation pipeline, but we currently
don't really use this.

General tips for writing test cases:
- You should write tests for any new features you implement.
- You should test edge cases as well.
- You should test cases that are supposed to throw an error using
`failed_from` and `compilation_failed_from`, defined in `tests/helper.py`.
- Any function whose name starts with `test_` will be run by pytest when
you call `hatch test` in the root directory of the project.
- You should also have an `if __name__ == "__main__":` block at the end of each
test file which calls each test function (see any `tests/e2e/test_XYZ.py` file
for an example).
    - This allows you to call a single file using `python test_XYZ.py`.
    - Calling an individual file is more convenient for debugging, since
    it only prints the error message from the first test that fails.
    - This also makes it possible to comment out all but a specific run
    statement to run only a single test function.
- For generating arrays as test data, you should prefer to use `multi_arange`
from `helper.py`. You can do some operations on the output of this to make it
be in the correct range (e.g. see `test_linalg.py`).
    - Small arrays, like hardcoding a 2x2 is also okay.
    - You can also use other methods, like numpy random with a fixed seed if
    necessary.
    - `multi_arange` is generally preferrable to random, because it is
    easier to debug when something breaks, because when you print the
    arrays, it's easier to try to guess what went wrong when the input
    is highly structured like `multi_arange`.

# Formatting

You can run `hatch fmt -f --check` to check files for formatting issues and
`hatch fmt -f` to automatically fix them.
In the CI, we enforce that `hatch fmt -f --check` returns no unformatted files.

You can also run `hatch fmt --check` (no `-f`) to run the linter as well.
Note that this will show unsafe fixes as well, so you should not run it without
the `--check` flag, since it would break examples/testcases using docstring
directives.
We do not enforce that the linter passes in CI.
We generally recommend running `hatch fmt --check [filename]` on any files
you are working on (after you are mostly done), and manually applying the fixes
it suggests that you think are useful (some of them might not be the most
useful for our project, and maybe we could add them to the ignore list in the
future instead).

In the future, it might be a good idea to enable (at least part of) the linter
in CI: see https://github.com/Huawei-CPLLab/PyDSL/issues/43.

The current formatter is not always strict about line-width of comments and
docstrings.
Thus, you should make your IDE display the max line-width used by the project
(specified in `pyproject.toml`, currently 79).
In VS Code, this can be done by adding `"editor.rulers": [79],` to
`settings.json`.

# Pull requests

- You can make draft pull requests if you want someone to look at your code and
give feedback before you are finished.
- Pull requests should be rebased by the author before merging.
- At least one person other than the author should approve a pull request
before merging.

# Errors and type safety

- PyDSL can be unsafe at runtime.
    - If something can only be checked at runtime, it's ok to not check it and
    get undefined behaviour.
    - This is because PyDSL is supposed to be a moderately thin wrapper around
    MLIR.
    - Example: we don't do bounds checking for array (memref/tensor) accesses.
- PyDSL should catch any errors that can be caught at compile time.
    - Ideally, we should not let any errors go to the Python MLIR binding
    level.
    - Example: in `linalg.reduce`, we check that both arguments are both tensor
    or both memref, and we check that after removing the dimensions specified
    by the user, the non-dynamic dimensions match.
    - Our existing code does not always do this. This mostly is because Python
    is weakly typed which doesn't play nicely with this.
    - It is most important that reasonable errors are caught with a nice
    message, ones a user is likely to make.

# Compiler code (Python) vs. user code (PyDSL)

The same syntax might do slightly different things if it's called from PyDSL
vs. if it's called by the compiler directly.
- For example, for a function-like object (`Function`, `CallMacro`,
`InlineFunction`, class of PyDSL type like `Int`, etc.) calling it from PyDSL
calls its `on_Call` method, but calling it from compiler code directly calls
its  `__call__` method (which is `__init__` for a class).
- The required arguments and the exact behaviour can thus be slightly
different.
    - E.g. when calling a `CallMacro` or `InlineFunction` from compiler code
    (like from another `CallMacro`), you must manually pass in `visitor` as the
    first argument.
- Sometimes calling a function-like object from compiler code and user code is
basically the same, which is nice.
    - Casting of scalar types works directly in both user and compiler code:
    `UInt32(UInt16(123))` is valid in both PyDSL and Python.
        - Maybe the behaviour for casting within the compiler will become
        a bit different depending on how
        https://github.com/Huawei-CPLLab/PyDSL/issues/19 is resolved.
- When writing something similar that you want to be able to call from both
PyDSL code and compiler code, the main way to decide what should go in
`on_Call` and what should go in `__call__` is based on what logic needs to be
done only when called from PyDSL, and what logic needs to be done for both
being called from PyDSL and Python.

# Circular import

Circular import errors can come up, we don't currently know a good way to fix
them.
- You can try to restructure your function definitions so that no circular
imports are necessary (by moving the function to another file).
- If that is not possible, you can try importing inside a class, inside  a
function, or not at the top of the file (very ugly); or try `import xyz`
instead of `from xyz import abc`.

# Building a wheel

`hatch build` can be used to build a wheel and also a zip archive of the
project.
This can take a minute or two and there are not always proper "progress"
messages.
It is unclear how well these work, considering our dependency on MLIR
Python bindings, which can not easily be packaged into a wheel by default.
See https://github.com/Huawei-CPLLab/PyDSL/issues/68.

`hatch env prune` can be used to clean/remove your hatch environment, but we
haven't used this much during development.
