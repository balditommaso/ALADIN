from copy import deepcopy
from qonnx.core.modelwrapper import ModelWrapper
from dory.Frontend_frameworks.QONNX.transformations.base import BaseTrasformation
from onnx import helper



class RenameTensorsSequentially(BaseTrasformation):
    
    def __init__(self, verbose=False):
        super().__init__(verbose)
    
    def apply(self, model: ModelWrapper):
        graph = model.graph

        rename_map = {}
        counter = 0

        # Go through nodes in order — this keeps graph execution order
        for node in graph.node:
            # Rename ALL inputs (not just input[0])
            for i in range(min(2, len(node.input))):        # HACK to save bias tag
                in_name = node.input[i]
                if in_name not in rename_map:
                    rename_map[in_name] = str(counter)
                    counter += 1
                
                node.input[i] = rename_map[in_name] 

            # Rename ALL outputs (not just output[0])
            for i in range(len(node.output)):
                out_name = node.output[i]
                if out_name not in rename_map:
                    rename_map[out_name] = str(counter)
                    counter += 1
                node.output[i] = rename_map[out_name]

        # Update value_info, inputs, outputs, and initializers
        for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
            if vi.name in rename_map:
                vi.name = rename_map[vi.name]

        for init in graph.initializer:
            if init.name in rename_map:
                init.name = rename_map[init.name]
                
        return (model, False)
