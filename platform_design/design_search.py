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
L1_MEM = [64000]
L2_MEM = [256000, 320000, 512000]


def arg_parser() -> Dict:
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    
    parser.add_argument('--config_files', nargs="+", type=str, help='Path to the JSON file that specifies the ONNX file of the network and other information.')
    parser.add_argument('--result_file', type=str, default="./perf_analysis.csv", help='Name of the final CSV file.')
    parser.add_argument('--prefix', type=str, default='', help='Prefix for the naming')
    parser.add_argument('--only_plot', action='store_true', help='Only plotting without grid-search')
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
    
    csv_path = os.path.join("./platform_design/analysis", args.result_file)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not args.only_plot:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Case", "layer_name", "MACs", "num_cycles", "MAC_per_cycle", "num_cores", "L1_mem", "L2_mem", "L1_tiling", "L2_tiling"])
            for file_idx, cfg_file in enumerate(args.config_files, start=1):
                case_name = f"Case {file_idx}"
                
                for L1_mem in L1_MEM:
                    for L2_mem in L2_MEM:
                        layers_tiling_info = None
                        try:
                            tmp_exit = os._exit
                            os._exit = lambda code=0: (_ for _ in ()).throw(SystemExit(code))
                            network_generate(
                                "QONNX",
                                "PULP.PULP_gvsoc",
                                cfg_file,
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
                                            case_name,
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
    
    os.makedirs("./platform_design/image/", exist_ok=True)                               
    for idx in range(1, len(args.config_files)):
        case_name = f"Case {idx}"
        for L1_mem in L1_MEM:
            for L2_mem in L2_MEM:
                for n_core in NUM_CORES:
                    # plot_performance(csv_path, f"./platform_design/image/{args.prefix}_{case_name[-1]}_performance.png") 
                    plot_performance(csv_path, f"./platform_design/image/{args.prefix}_{case_name[-1]}_cycle_by_core.png", "num_cores")  
                    plot_performance(csv_path, f"./platform_design/image/{args.prefix}_{case_name[-1]}_cycle_by_L2.png", "L2_mem") 
                    plot_memory(csv_path, f"./platform_design/image/{args.prefix}_{case_name[-1]}_memory.png", "L2_mem")  
                    
            
        plot_metric_comparison(
            file_path=csv_path,
            metric="num_cycles",
            cores=8,
            l1=64,
            l2=512,
            group_by="layer_name",
            compare_by="Case",
            dst_path=f"./platform_design/image/{args.prefix}_num_cycles_case_comparison.png"
        )           
        plot_metric_comparison(
            file_path=csv_path,
            metric="L1_tiling",
            cores=8,
            l1=64,
            l2=512,
            group_by="layer_name",
            compare_by="Case",
            dst_path=f"./platform_design/image/{args.prefix}_L1_tilling_case_comparison.png"
        )   
        plot_metric_comparison(
            file_path=csv_path,
            metric="L2_tiling",
            cores=8,
            l1=64,
            l2=512,
            group_by="layer_name",
            compare_by="Case",
            dst_path=f"./platform_design/image/{args.prefix}_L2_tiling_case_comparison.png"
        )                
                         

if __name__ == "__main__":
    args = arg_parser()
    main(args)
    