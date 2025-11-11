import os 
import json
import csv
import argparse
import numpy as np
from qonnx.util.basic import get_by_name
from argparse import RawTextHelpFormatter
from typing import *
from plot import *


def args_parse() -> Dict:
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    
    parser.add_argument('--config_files', nargs="+", type=str, help='Path to the JSON file that specifies the ONNX file of the network and other information.')
    parser.add_argument('--result_file', type=str, default="profile.csv", help='Name of the final CSV file.')
    parser.add_argument('--prefix', type=str, default='', help='Prefix for the naming')

    args = parser.parse_args()
    return args


def profile_convolution(
    input_shape: List[int], 
    output_shape: List[int], 
    weight_shape: List[int],
    input_bitwidth: int, 
    weight_bitwidth: int, 
    output_bitwidth: int, 
    padding: List[int],
    stride: List[int],
    implementation: str,
    *args, **kwargs
) -> Tuple[int, int, int]:
    """
    Profile a convolution operation using im2col transformation.

    Returns:
        macs: total number of multiply-accumulate operations
        total_mem: total memory footprint in bytes
    """
    C_in, H_in, W_in = input_shape
    C_out, _, K_h, K_w = weight_shape
    p_t, p_b, p_l, p_r = padding
    s1, s2 = stride

    H_out = (H_in + p_t + p_b - K_h) // s1 + 1
    W_out = (W_in + p_l + p_r - K_w) // s2 + 1

    # im2col matrix dimensions: [C_in * K_h * K_w, H_out * W_out]
    im2col_elems = C_in * K_h * K_w * H_out * W_out
    im2col_mem = im2col_elems * input_bitwidth // 8

    # standard buffers
    input_mem = np.prod(input_shape) * input_bitwidth // 8
    weight_mem = np.prod(weight_shape) * weight_bitwidth // 8
    # bias_mem = C_out * output_bitwidth // 8
    out_mem = np.prod(output_shape) * output_bitwidth // 8

    total_mem = input_mem + weight_mem + out_mem + im2col_mem
    if "bias_shape" in kwargs:
        bias_mem = np.prod(kwargs["bias_shape"]) * output_bitwidth // 8
        total_mem += bias_mem
    # LUT-based implementation adds additional lookup table storage
    
    macs = C_out * H_out * W_out * C_in * K_h * K_w
    bops = macs * (1 + input_bitwidth + weight_bitwidth + output_bitwidth)
    if implementation == "lut":
        lut_dim = 2 ** (input_bitwidth + weight_bitwidth) * output_bitwidth // 8
        total_mem += lut_dim
        macs = 0

    return macs, total_mem, bops
    
    
def profile_gemm(
    input_shape: List[int], 
    output_shape: List[int], 
    weight_shape: List[int],
    input_bitwidth: int, 
    weight_bitwidth: int, 
    output_bitwidth: int, 
    implementation: str,
    *args, **kwargs
) -> Tuple[int, int, int]:
    input_mem = np.prod(input_shape) * input_bitwidth // 8
    weight_mem = np.prod(weight_shape) * weight_bitwidth // 8
    out_mem = np.prod(output_shape) * output_bitwidth // 8
    
    total_mem = input_mem + weight_mem + out_mem
    if "bias_shape" in kwargs:
        bias_mem = np.prod(kwargs["bias_shape"]) * output_bitwidth // 8
        total_mem += bias_mem
        
    macs = weight_shape[0] * weight_shape[1]
    bops = macs * (1 + input_bitwidth + output_bitwidth + weight_bitwidth)
    if implementation == "lut":
        lut_dim = 2 ** (input_bitwidth + weight_bitwidth) * output_bitwidth // 8
        total_mem += lut_dim
        macs = 0
    
    return macs, total_mem, bops
    

def profile_relu(input_shape: List[int], input_bitwidth: int, *args, **kwargs) -> Tuple[int, int, int]:
    input_mem = output_mem = np.prod(input_shape) * input_bitwidth
    total_mem = input_mem + output_mem
    macs = 0
    bops = np.prod(input_shape) * (input_bitwidth + 1)
    
    return macs, total_mem, bops 


def profile_quant(input_shape: List[int], input_bitwidth: int, output_bitwidth: int, implementation: str, *args, **kwargs) -> Tuple[int, int, int]:
    channel_wise = kwargs.get("channel-wise", False)
    scale = 1 if not channel_wise else input_shape[0]
    # zp = 1 if not channel_wise else input_shape[0]
    
    input_mem = np.prod(input_shape) * input_bitwidth
    output_mem = np.prod(input_shape) * output_bitwidth
    
    if implementation == "thresholds":
        num_bins = (2 ** output_bitwidth) - 1
        bops = (np.log2(num_bins) * input_bitwidth) * np.prod(input_shape)
        param_mem = num_bins * input_bitwidth * scale
    else:
        bops = (2 * input_bitwidth + 5) + np.prod(input_shape)
        param_mem = scale * 32
        
    total_mem = input_mem + output_mem + param_mem
    
    return 0, total_mem, bops


def profile_avgpool(input_shape: List[int], output_shape: List[int], input_bitwidth: int, *args, **kwargs) -> Tuple[int, int, int]:
    _, H_in, W_in = input_shape
    _, H_out, W_out = output_shape
    
    K_h = H_in // H_out
    K_w = W_in // W_out
    
    input_mem = output_mem = np.prod(input_shape) * input_bitwidth
    total_mem = input_mem + output_mem
    
    bops = np.prod(input_shape) * K_h * K_w * input_bitwidth
    
    return 0, total_mem, bops
    

def profile_maxpool(input_shape: List[int], output_shape: List[int], input_bitwidth: int, *args, **kwargs) -> Tuple[int, int, int]:
    _, H_in, W_in = input_shape
    _, H_out, W_out = output_shape
    
    K_h = H_in // H_out
    K_w = W_in // W_out
    
    input_mem = output_mem = np.prod(input_shape) * input_bitwidth
    total_mem = input_mem + output_mem
    
    bops = np.prod(input_shape) * K_h * K_w * input_bitwidth
    
    return 0, total_mem, bops


def main(args: Dict) -> None:
    
    csv_path = os.path.join("./analysis/", args.result_file)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Case", "Layers", "MACs", "Memory", "BOPs"])
        for idx, cfg_file in enumerate(args.config_files, start=1):
            case_name = f"Case {idx}"
            
            with open(cfg_file, "r") as f:
                config = json.load(f)
            
            layer_idx = 0
            for layer, info in config.items():
                if info["type"] == "Conv":
                    macs, memory, bops = profile_convolution(**info)
                elif info["type"] in ["Gemm", "Matmul"]:
                    macs, memory, bops = profile_gemm(**info)
                elif info["type"] in ["Trunc", "Quant"]:
                    macs, memory, bops = profile_quant(**info)
                elif info["type"] == "Relu":
                    macs, memory, bops = profile_relu(**info)
                elif info["type"] in ["GlobalAveragePool", "AveragePool"]:
                    macs, memory, bops = profile_avgpool(**info)
                elif info["type"] in ["GlobalMaxPool", "MaxPool"]:
                    macs, memory, bops = profile_maxpool(**info)
                else:
                    continue
                
                layer_name = f"{info['type']}_{layer_idx}"
                layer_idx += 1
                layer_name = layer_name.replace("GlobalAveragePool", "avgpool")
                writer.writerow([case_name, layer_name, macs, memory, bops])
            
    # plot stuff
    plot_macs(csv_path, f"./image/{args.prefix}_macs.png", compare_by="Case")
    plot_memory(csv_path, f"./image/{args.prefix}_memory.png", compare_by="Case")
    plot_bops(csv_path, f"./image/{args.prefix}_bops.png", compare_by="Case")
    

if __name__ == "__main__":
    args = args_parse()
    main(args)