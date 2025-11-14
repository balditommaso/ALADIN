

from dory.Hardware_targets.PULP.Common.Tiler.tiler import Tiler_PULP
from ..Ne16_HW_node import Ne16_HW_node
from .tiler_conv2d_ne16 import Tiler_Conv2D_Ne16


class Tiler_GAP9(Tiler_PULP):
    def __init__(self, HW_node, previous_HW_node, code_reserved_space, double_buffering=2):
        super().__init__(HW_node, previous_HW_node, code_reserved_space, double_buffering)

    def get_tiling(self, level):
        if 'Conv' in self.HW_node.name and isinstance(self.HW_node, Ne16_HW_node):
            return Tiler_Conv2D_Ne16(self).get_tiling(level)
        return super().get_tiling(level)
