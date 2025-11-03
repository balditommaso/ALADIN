import numpy as np
from onnx import helper, TensorProto, NodeProto
from qonnx.core.modelwrapper import ModelWrapper
from dory.Frontend_frameworks.QONNX.transformations.base import BaseTrasformation
from typing import *


def find_quant_producer(model: ModelWrapper, tensor_name: str, depth: int = 5) -> NodeProto:
    while (depth > 0):
        node = model.find_producer(tensor_name)
        if node is None:
            break
        
        if node.op_type in ["Quant", "Trunc"]:
            return node
        
        tensor_name = node.input[0]
        depth -= 1
    
    raise ValueError("Quant Node not found!")


class RecordOutScale(BaseTrasformation):
    """
    Record the out scale in the op. node
    """
    
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        
    
    def apply(self, model: ModelWrapper) -> Tuple[ModelWrapper, bool]:
        graph = model.graph
        # iter_graph = deepcopy(graph)
        for node in graph.node:
            # check operation which could have static parameters
            if node.op_type == "Quant" or len(node.input) < 2:
                continue
            
            if len(node.input) == 2:
                # out scale is given by the product of the scale 
                in_quant = find_quant_producer(model, node.input[0])
                w_quant = find_quant_producer(model, node.input[1])
                
                in_scale_index = 1 if in_quant.op_type == "Quant" else 4
                in_scale = model.get_initializer(in_quant.input[in_scale_index])
                w_scale_index = 1 if w_quant.op_type == "Quant" else 4
                w_scale = model.get_initializer(w_quant.input[w_scale_index])
                
                out_scale = np.multiply(in_scale, w_scale)
            elif len(node.input) == 3:
                # extract the out scale from the bias
                b_quant = find_quant_producer(model, node.input[2])
                out_scale_index = 1 if b_quant.op_type == "Quant" else 4
                out_scale = model.get_initializer(b_quant.input[out_scale_index])
            else:
                self.warning_message(f"{node.name} has more than 3 inputs, not handled yet.")
                continue
            
            
            attr = helper.make_attribute(
                "out_scale",
                helper.make_tensor(
                    name=model.make_new_valueinfo_name(),
                    data_type=TensorProto.FLOAT,
                    dims=out_scale.shape,
                    vals=out_scale.flatten().astype(float)
                )
            )
            
            node.attribute.append(attr)
        
        return (model, False)
            

   