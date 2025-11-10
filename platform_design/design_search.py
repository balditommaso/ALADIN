import argparse
import subprocess
import csv
import json
import os
from argparse import RawTextHelpFormatter
from network_generate import network_generate
from typing import *
from plot import *

NUM_CORES = [8, 4, 2]
L1_MEM = [16000, 32000, 64000,]
L2_MEM = [128000, 256000, 512000]


def arg_parser() -> Dict:
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    
    parser.add_argument('config_file', type=str, help='Path to the JSON file that specifies the ONNX file of the network and other information.')
    parser.add_argument('result_file', type=str, default="./perf_analysis.csv", help='Name of the final CSV file.')

    args = parser.parse_args()
    return args


def sum_componenets(tile: Dict) -> int:
    return sum((
        tile["weight_memory"],
        tile["bias_memory"],
        tile["constants_memory"],
        tile["input_activation_memory"],
        tile["output_activation_memory"],
        tile.get("lut_memory", 0),
    ))


def main(args: Dict) -> None:
    
    csv_path = os.path.join("./platform_design/", args.result_file)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer_name", "MACs", "num_cycles", "MAC_per_cycle", "num_cores", "L1_mem", "L2_mem", "L1_tiling", "L2_tiling"])
        
        for L1_mem in L1_MEM:
            for L2_mem in L2_MEM:
                layers_tiling_info = None
                try:
                    tmp_exit = os._exit
                    os._exit = lambda code=0: (_ for _ in ()).throw(SystemExit(code))
                    network_generate(
                        "QONNX",
                        "PULP.PULP_gvsoc",
                        args.config_file,
                        L1_capacity=L1_mem,
                        L2_capacity=L2_mem,
                        appdir="./application",
                        perf_layer="Yes",
                        optional="mixed-sw",
                        prefix=""
                    )
                except SystemExit as e:
                    print("Not enough resources to schedule the inference!")
                    continue
                finally:
                    os._exit = tmp_exit
                
                with open("./logs/HW_related/json_files/06_DORY_HW_tiled_graph.json", "r") as f:
                        layers_tiling_info = json.load(f)["graph"]
                
                                    
                for n_core in NUM_CORES:
                    # generate the C code
                    print(100 * "*")
                    print(f"Config under analysis: {n_core} cores | L1 {L1_mem//1000} kB | L2 {L2_mem//1000} kB")
                    print(100 * "*")
                    # run the C code
                    cmd = ['make', '-C', './application', 'clean', 'all', 'run', 'platform=gvsoc', f'CORE={n_core}']
                    try:
                        proc = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=720)
                    except subprocess.CalledProcessError as e:
                        raise RuntimeError(f"Build or run failed (exit {e.returncode}):\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}")
                    except subprocess.TimeoutExpired as e:
                        raise TimeoutError(f"Test timed out after 720s.\nPartial STDOUT:\n{e.output}\nSTDERR:\n{e.stderr}")

                    # extract info and store on a CSV
                    idx = 0
                    for line in proc.stdout.splitlines():
                        if line.startswith("PERF_LOG"):
                            info = line.strip().split(",")
                            if len(info) == 6:
                                _, layer_name, macs, cycles, perf, cores = info
                                tiling_info = layers_tiling_info[idx]["Tiling_parameters"]
                                idx += 1
                                writer.writerow([
                                    layer_name,
                                    int(macs),
                                    int(cycles),
                                    float(perf),
                                    int(cores),
                                    L1_mem,
                                    L2_mem,
                                    sum_componenets(tiling_info["L1"]),
                                    sum_componenets(tiling_info["L2"])
                                ])
                                
    
    plot_performance(csv_path, "./performance.png")                            

if __name__ == "__main__":
    args = arg_parser()
    main(args)
    