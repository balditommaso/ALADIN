import json
import numpy as np
from onnx import helper, TensorProto, NodeProto
from qonnx.core.modelwrapper import ModelWrapper
from dory.Frontend_frameworks.QONNX.transformations.base import BaseTrasformation
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
        

    def apply(self, model: ModelWrapper) -> Tuple[ModelWrapper, bool]:
        graph = model.graph
        
        for node in graph.node:
            if node.name not in self.config:
                self.warning_message(f"Implementation info not found for {node.name}")
                continue
            
            impl = self.config[node.name]["implementation"]
            self.info_message(f"{node.name} -> {impl};")
            attr = helper.make_attribute("implementation", impl)
            node.attribute.append(attr)
            
        return model, False
            
    # def apply(self, model: ModelWrapper) -> Tuple[ModelWrapper, bool]:
    #     conv_idx = 0
    #     relu_idx = 0
    #     avgpool_idx = 0
    #     fc_idx = 0
        
    #     conv_confs = [v for v in self.config.values() if v.get("type") == "Conv2d"]
    #     relu_confs = [v for v in self.config.values() if v.get("type") == "ReLU"]
    #     fc_confs = [v for v in self.config.values() if v.get("type") in ["Linear", "Gemm", "MatMul"]]
    #     avgpool_confs = [v for v in self.config.values() if v.get("type") in ["AvgPool2d", "AveragePool"]]
        
    #     graph = model.graph
    #     for node in graph.node:
    #         if node.op_type == "Conv":
    #             if conv_idx >= len(conv_confs):
    #                 self.warning_message(f"{node.name} implementation info not found! ({conv_idx})")
    #                 continue
    #             impl =conv_confs[conv_idx].get("implementation", None)
    #             conv_idx += 1
    #         elif node.op_type == "Relu":
    #             if relu_idx >= len(relu_confs):
    #                 self.warning_message(f"{node.name} implementation info not found! ({relu_idx})")
    #                 continue
    #             impl = relu_confs[relu_idx].get("implementation", None)
    #             relu_idx += 1
    #         elif node.op_type in ["Gemm", "MatMul"]:
    #             if fc_idx >= len(fc_confs):
    #                 self.warning_message(f"{node.name} implementation info not found! ({fc_idx})")
    #                 continue
    #             impl = fc_confs[fc_idx].get("implementation", None)
    #             fc_idx += 1
    #         elif node.op_type == "AveragePool":
    #             if avgpool_idx >= len(avgpool_confs):
    #                 self.warning_message(f"{node.name} implementation info not found! ({avgpool_idx})")
    #                 continue
    #             impl = avgpool_confs[avgpool_idx].get("implementation", None)
    #             avgpool_idx += 1
    #         else:
    #             continue
            
    #         if impl is None:
    #             self.warning_message(f"No implementation found for {node.name}")
    #         else:
    #             attr = helper.make_attribute("implementation", impl)
    #             node.attribute.append(attr)
        
    #     return (model, False)
            

   