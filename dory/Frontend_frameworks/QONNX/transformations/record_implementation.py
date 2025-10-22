import json
import numpy as np
from onnx import helper, TensorProto, NodeProto
from qonnx.core.modelwrapper import ModelWrapper
from dory.Frontend_frameworks.QONNX.transformations.base import BaseTrasformation
from dory.Frontend_frameworks.QONNX.transformations.record_out_scale import find_quant_producer
from typing import *



class RecordImplementation(BaseTrasformation):
    """
    Record the implementation details of each node
    """
    
    def __init__(self, path_implementation: str, verbose: bool = False):
        self.config = None
        with open(path_implementation, "r") as f:
            self.config = json.load(f)
            
        super().__init__(verbose)
        
    # TODO: double check if it works fine with multiple branches
    def apply(self, model: ModelWrapper) -> Tuple[ModelWrapper, bool]:
        conv_idx = 1
        relu_idx = 1
        avgpool_idx = 1
        fc_idx = 1
        graph = model.graph
        for node in graph.node:
            if node.op_type == "Conv":
                if f"conv{conv_idx}" not in self.config:
                    self.warning_message(f"{node.name} implementation info not found! ({conv_idx})")
                    continue
                impl = self.config[f"conv{conv_idx}"]["implementation"]
                conv_idx += 1
            elif node.op_type == "Relu":
                if f"relu{relu_idx}" not in self.config:
                    self.warning_message(f"{node.name} implementation info not found! ({relu_idx})")
                    continue
                impl = self.config[f"relu{relu_idx}"]["implementation"]
                relu_idx += 1
            elif node.op_type in ["Gemm", "MatMul"]:
                if f"fc{fc_idx}" not in self.config:
                    self.warning_message(f"{node.name} implementation info not found! ({fc_idx})")
                    continue
                impl = self.config[f"fc{fc_idx}"]["implementation"]
                fc_idx += 1
            elif node.op_type == "AveragePool":
                if f"avgpool{avgpool_idx}" not in self.config:
                    self.warning_message(f"{node.name} implementation info not found! ({avgpool_idx})")
                    continue
                impl = self.config[f"avgpool{avgpool_idx}"]["implementation"]
                avgpool_idx += 1
            else:
                continue
            
            attr = helper.make_attribute("implementation", impl)
            node.attribute.append(attr)
        
        return (model, False)
            

   