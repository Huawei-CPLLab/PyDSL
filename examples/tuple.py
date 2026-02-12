from pydsl.frontend import compile
from pydsl.type import F32, Index, Tuple


@compile(dump_mlir=True)
def tuple_example(a: F32, b: F32) -> Tuple[F32, Index]:
    d: F32 = 12.0
    l: Index = 5

    return (a / b) + d, l


a, b = tuple_example(25, 3)

print(f"a = {a} should be {25 / 3 + 12}")
print(f"b = {b} should be {5}")
