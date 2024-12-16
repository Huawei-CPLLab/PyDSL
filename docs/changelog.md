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

## Other improvements:
- Open-sourced the E2E tests which runs through the entire compilation pipeline and checks the result for 107 PyDSL programs.
- More docstrings are added to methods and fields in the source.
- Updated documentation (`docs/usage.md`).