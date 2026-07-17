from onnx import helper, TensorProto
from qonnx.core.modelwrapper import ModelWrapper
from dory.Frontend_frameworks.QONNX.transformations.base import BaseTrasformation
from qonnx.util.basic import get_by_name
from typing import *



class DoryConfigParser(BaseTrasformation):
    """
    Generate the config file espected by DORY for the parsing
    and edit it based on the QONNX model informations
    
    """
    
    def __init__(self, config: Dict[str, Any], code_size: int = 150000, verbose: bool = False):
        super().__init__(verbose)
        self.config = config
        self.code_size = code_size
    
    
    def apply(self, model: ModelWrapper) -> Tuple[ModelWrapper, bool]:
        graph = model.graph
        # defaults
        self.config["BNRelu_bits"] = 32
        self.config["code reserved space"] = self.code_size
        
        # we handle only models with just one input
        self.config["n_inputs"] = 1
        input_quant = model.find_consumer("global_in")
        if input_quant.op_type != "Quant":  
            self.error_message(f"Missing input quantization!.", ValueError)
            
        self.config["input_bits"] = int(model.get_initializer(input_quant.input[3]))
        self.config["input_signed"] = bool(get_by_name(input_quant.attribute, "signed").i)
        
        quant_node = input_quant
        old_input_name = graph.input[0].name

        quant_input_name = quant_node.input[0]
        quant_output_name = quant_node.output[0]

        assert old_input_name == quant_input_name

        for node in graph.node:
            if node == quant_node:
                continue
            for i, inp in enumerate(node.input):
                if inp == quant_output_name:
                    node.input[i] = quant_input_name

        graph.node.remove(quant_node)
        
        return (model, False)
            
            
            
            
            