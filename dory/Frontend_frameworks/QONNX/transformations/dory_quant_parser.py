import numpy as np
from copy import deepcopy
from onnx import helper, numpy_helper, TensorProto
from qonnx.core.modelwrapper import ModelWrapper, NodeProto
from dory.Frontend_frameworks.QONNX.transformations.base import BaseTrasformation
from dory.Frontend_frameworks.QONNX.transformations.fold_static_quant import *
from qonnx.util.basic import get_by_name
from typing import *

MAX_DEPTH = 5
            
        

def get_prev_out_scale(mw: ModelWrapper, node: NodeProto) -> np.array:
    curr_node = deepcopy(node)
    
    for _ in range(MAX_DEPTH):
        buffer = get_by_name(curr_node.attribute, "out_scale")
        if buffer is not None:
            return numpy_helper.to_array(buffer.t)
        
        curr_node = mw.find_producer(curr_node.input[0])
        if curr_node is None:
            return np.array(1)
        
    raise ValueError(f"Output scale of the previous {node.name} not found!")


def replace_quant(mw: ModelWrapper, node: NodeProto, delta: int = 2**16) -> str:           
    prev_node = mw.find_producer(node.input[0])

    if prev_node is not None and prev_node.op_type == "Relu":
        relu_input = prev_node.input[0]
        node.input[0] = relu_input
        mw.graph.node.remove(prev_node)
        
    out_scale = get_prev_out_scale(mw, node)
    quant_scale = mw.get_initializer(node.input[1])
    rounding_mode = get_by_name(node.attribute, "rounding_mode").s.decode("utf-8")
    round_fx = resolve_rounding_mode(rounding_mode)
    
    out_shape = mw.get_tensor_shape(node.input[0])
    C_out = out_shape[1] if len(out_shape) > 1 else out_shape[0]
    
    M = round_fx(out_scale / quant_scale * delta).astype(np.float32)
    M = numpy_helper.from_array(M, mw.make_new_valueinfo_name())
    mw.graph.initializer.append(M)
    
    out_mul_tensor = helper.make_tensor_value_info(
        mw.make_new_valueinfo_name(),
        TensorProto.FLOAT,
        mw.get_tensor_shape(node.input[0])
    )
    mw.graph.value_info.append(out_mul_tensor)
    
    mul_node = helper.make_node(
        "Mul",
        [node.input[0], M.name],
        [out_mul_tensor.name]
    )
    
    mw.graph.node.append(mul_node)
    div_input_name = out_mul_tensor.name
    # add node only for asymmetric quantization
    zeropt = mw.get_initializer(node.input[2])
    if zeropt is not None and not np.all(zeropt == 0.):
        Z = round_fx(zeropt * delta * out_scale)
        if np.isscalar(Z) or np.size(Z) == 1:
            Z = np.full((C_out, 1, 1), float(Z), dtype=np.float32)
        else:
            Z = np.reshape(Z, (C_out, 1, 1)).astype(np.float32)
                    
        Z = numpy_helper.from_array(Z, mw.make_new_valueinfo_name())
        mw.graph.initializer.append(Z)
            
        out_add_tensor = helper.make_tensor_value_info(
            mw.make_new_valueinfo_name(),
            TensorProto.FLOAT,
            mw.get_tensor_shape(node.input[0])
        )
        mw.graph.value_info.append(out_add_tensor)
        
        add_node = helper.make_node(
            "Add",
            [out_mul_tensor.name, Z.name],
            [out_add_tensor.name]
        )
        mw.graph.node.append(add_node)
        div_input_name = out_add_tensor.name
                
    # div node to remove the scale
    D = np.array(delta, dtype=np.float32)
    D = numpy_helper.from_array(D, mw.make_new_valueinfo_name())
    mw.graph.initializer.append(D)
                
    out_div_tensor = helper.make_tensor_value_info(
        mw.make_new_valueinfo_name(),
        TensorProto.FLOAT,
        mw.get_tensor_shape(node.input[0])
    )
    mw.graph.value_info.append(out_div_tensor)
    
    div_node = helper.make_node(
        "Div",
        [div_input_name, D.name],
        [out_div_tensor.name]
    )
    mw.graph.node.append(div_node)
    
    # clip to apply the relu activation
    out_clip_tensor = helper.make_tensor_value_info(
        mw.make_new_valueinfo_name(),
        TensorProto.FLOAT,
        mw.get_tensor_shape(node.input[0])
    )
    
    clip_node = helper.make_node(
        "Clip",
        [out_div_tensor.name],
        [out_clip_tensor.name]
    )
        
    out_bit_width = int(mw.get_initializer(node.input[3]))
    is_narrow = bool(get_by_name(node.attribute, "narrow").i)
    signed = bool(get_by_name(node.attribute, "signed").i)
    
    clip_min = min_int(signed, is_narrow, out_bit_width)
    clip_max = max_int(signed, is_narrow, out_bit_width)
    
    bit_width_attr = helper.make_attribute(
        "out_bits", out_bit_width
    )
    clip_min_attr = helper.make_attribute(
        "min", clip_min
    )
    clip_max_attr = helper.make_attribute(
        "max", clip_max
    )
    clip_node.attribute.extend([bit_width_attr, clip_min_attr, clip_max_attr])
    mw.graph.node.append(clip_node)
    # reconnecte the chain
    next_node = mw.find_consumer(node.output[0])
    if next_node is not None:
        next_node.input[0] = out_clip_tensor.name
    else:
        mw.graph.output[0].name = out_clip_tensor.name

    mw.graph.node.remove(node)
        


class DoryQuantParser(BaseTrasformation):
    def __init__(self, delta: int, verbose: bool = False):
        super().__init__(verbose)
        self.delta = delta

    def apply(self, model: ModelWrapper) -> Tuple[ModelWrapper, bool]:
        iter_model = deepcopy(model)

        quant_nodes = [n for n in iter_model.graph.node if n.op_type == "Quant"]
        for node in quant_nodes:
            replace_quant(iter_model, node, self.delta)

        return (iter_model, False)