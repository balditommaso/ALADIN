import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


def plot_performance(file_path, dst_path):
    df = pd.read_csv(file_path)
    df['config'] = df.apply(lambda x: f"Cores:{x['num_cores']}_L1:{x['L1_mem']}_L2:{x['L2_mem']}", axis=1)
    plt.figure(figsize=(12, 6))
    for config, group in df.groupby('config'):
        plt.plot(group['layer_name'], group['num_cycles'], marker='o', label=f"{config}")
    plt.xticks(rotation=90)
    plt.xlabel("Layer")
    plt.ylabel("Number of Cycles")
    plt.title("Layer-wise Performance Comparison by Configuration")
    plt.legend()
    plt.tight_layout()
    plt.savefig("performance_cycles.png")
    plt.close()  

# # Load CSV
# df = pd.read_csv('your_file.csv')

# # Create a configuration column to group by
# df['config'] = df.apply(lambda x: f"Cores:{x['num_cores']}_L1:{x['L1_mem']}_L2:{x['L2_mem']}", axis=1)



# # --- Plot 1b: MACs per layer ---
# plt.figure(figsize=(12, 6))
# for config, group in df.groupby('config'):
#     plt.plot(group['layer_name'], group['MACs'], marker='o', label=f"{config}")
# plt.xticks(rotation=90)
# plt.xlabel("Layer")
# plt.ylabel("MACs")
# plt.title("Layer-wise MACs Comparison by Configuration")
# plt.legend()
# plt.tight_layout()
# plt.savefig("performance_MACs.png")
# plt.close()

# # --- Plot 2: Tiling comparison ---
# plt.figure(figsize=(12, 6))
# for config, group in df.groupby('config'):
#     plt.plot(group['layer_name'], group['L1_tiling'], marker='o', label=f"L1 - {config}")
#     plt.plot(group['layer_name'], group['L2_tiling'], marker='x', linestyle='--', label=f"L2 - {config}")
# plt.xticks(rotation=90)
# plt.xlabel("Layer")
# plt.ylabel("Tiling Size")
# plt.title("Layer-wise L1 and L2 Tiling Comparison by Configuration")
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.savefig("tiling_comparison.png")
# plt.close()
