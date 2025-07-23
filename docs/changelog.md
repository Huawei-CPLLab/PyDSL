# 2025 July 23: `ndarray` Linking

`ndarray`s returned from a function now store a pointer to the root of
`ndarrays`s passed to the function if they overlap in memory to prevent the
memory from being deallocated.
It should now be possible to pass `ndarray`s to a function without having to
keep a reference to them!

Internally, we assume that for any `ndarray`, you can repeatedly take its
`.base` to eventually get an `ndarray` that owns its memory, has C contiguous
memory, and the returned `ndarray` is fully contained in a single such root
array.
If you know a not stupid way this can be false, please let us know.

# 2025 July 21: CallMacro Improvements

Rewrote some parts of the `CallMacro` system.
Several of these changes are not backwards compatible if you implemented your
own `CallMacro`s.

- `CallMacro`s can now be called from other `CallMacro`s.
  - Replaced internal `_on_Call` function with `__call__`.
  - Call macros are now represented by instances of subclasses of `CallMacro`,
  not subclasses directly.
- Support making `CallMacro`s that behave like Python instance/class methods
  using the `method_type` attribute, which can be specified in `CallMacro.generate`.
  - Removed `is_member` attribute, which had very similar but much more
  restricted functionality.

# December 2024 Update

For details of these changes, see `docs/usage.md`.

## Autotune
- New autotune feature, which allows you to find the program parameters that maximize the performance of your program.
  - See `examples/autotune/` for examples of using autotune.

## Compilation and syntax
- Classes can now be compiled into MLIR modules.
  - Apply `@compile` to a class. All methods of the class become functions of the MLIR module.
- Types with new type arguments can now be inlined directly within a PyDSL. E.g. `MemRef[F32, 5]`.
- New `Dialect` class.

## Frontend
- Significant refactor to the way that a frontend runner interprets a PyDSL function.
  - `MemRef` can now be returned directly from a `Function`.
  - `Tuple` can now be returned with `MemRef` as elements.
- Each target now has a method for rejecting programs. Specifically, targets will now reject dialects that they do not recognize in order to minimize possibilities of errors or segmentation fault from `mlir-opt` during the lowering stage.

## Function
- New `Function` and `TransformSequence` class.
- Logic for compiling a `Function` and a `TransformSequence` has been unified under abstract class `FunctionLike`.

## Transform
- Added `cse` and `outline_loop` to `pydsl.transform`.
- Tuple unpacking is now supported when assigning tuple to multiple variables. E.g. `fun, call = outline_loop(...)`.
- `AnyOp` graduated from a dummy type hint to a full PyDSL type.

## Other improvements
- Open-sourced the E2E tests which runs through the entire compilation pipeline and checks the result for 107 PyDSL programs.
- More docstrings are added to methods and fields in the source.
- Updated documentation (`docs/usage.md`).
