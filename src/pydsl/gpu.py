from enum import Enum

from mlir.ir import AttrBuilder
import mlir.dialects.gpu as gpu

from pydsl.memref import MemorySpace


class GPU_AddrSpace(MemorySpace, Enum):
    Global = gpu.AddressSpace.Global
    Workgroup = gpu.AddressSpace.Workgroup
    Private = gpu.AddressSpace.Private

    def lower(self):
        return (
            AttrBuilder.get("GPU_AddressSpaceAttr")(self.value, context=None),
        )
