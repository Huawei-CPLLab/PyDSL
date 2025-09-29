from typing import Tuple
import mlir.ir as mlir
from pydsl.protocols import Lowerable


class Attribute(Lowerable):
    @property
    def attr_name(self) -> str: ...

    def lower(self) -> Tuple[mlir.Attribute, ...]: ...
