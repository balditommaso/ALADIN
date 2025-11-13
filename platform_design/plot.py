import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Uses the same COLOR_PALETTE from your previous code
COLOR_PALETTE = [
    "#1a80bb",  # strong blue
    "#ea801c",  # vivid orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
]


def _fmt_config_label(cfg_val, compare_by=None):
    # nicely format config labels when compare_by is a mem column
    if compare_by == 'L1_mem' or compare_by == 'L2_mem':
        return f"{int(cfg_val)//1000} kB"
    return str(cfg_val)

def plot_performance(file_path, dst_path, compare_by=None, fill_missing=np.nan, max_configs=None):
    """
    compare_by: None or one of ['num_cores', 'L1_mem', 'L2_mem']
    fill_missing: np.nan or 0 depending if you want gaps or zeros for missing combos
    max_configs: optionally limit number of configs plotted (useful for wide data)
    """
    df = pd.read_csv(file_path)

    # original full config label (for compare_by is None)
    df['config'] = df.apply(lambda x: f"Cores: {x['num_cores']} L1: {x['L1_mem']//1000} kB L2: {x['L2_mem']//1000} kB", axis=1)

    # keep original layer ordering as it appears in file (preserves expected x order)
    layers = pd.Categorical(df['layer_name'], categories=pd.unique(df['layer_name']), ordered=True)

    if compare_by in ['num_cores', 'L1_mem', 'L2_mem']:
        # group by layer and compare_by, compute mean
        df_grouped = df.groupby(['layer_name', compare_by], as_index=False)['num_cycles'].mean()
        # pivot so columns are the compare_by values and rows are layers
        pivot = df_grouped.pivot(index='layer_name', columns=compare_by, values='num_cycles')

        # reindex pivot to original layer order
        pivot = pivot.reindex(index=pd.unique(df['layer_name']))

        # optionally limit configs
        cols = list(pivot.columns)
        if max_configs:
            cols = cols[:max_configs]
            pivot = pivot[cols]

        # prepare labels for legend
        col_labels = [_fmt_config_label(c, compare_by) for c in pivot.columns]

        # plotting
        n_configs = len(pivot.columns)
        x = np.arange(len(pivot.index))
        width = 0.8 / max(1, n_configs)

        plt.figure(figsize=(max(10, len(x)*0.25), 6))
        for i, col in enumerate(pivot.columns):
            y = pivot[col].values
            # replace missing values with fill_missing for plotting if needed
            y_plot = np.where(np.isnan(y) & np.isnan(fill_missing), np.nan, np.nan_to_num(y, nan=fill_missing))
            plt.bar(x + i*width, y_plot, width=width, label=col_labels[i])

        plt.xticks(x + width*(n_configs-1)/2, pivot.index, rotation=90)
        plt.xlabel("Layer")
        plt.ylabel("Number of Cycles (log scale)")
        plt.yscale("log")
        plt.title(f"Layer-wise Performance Comparison by {compare_by}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(dst_path)
        plt.close()

    else:
        # original behavior: each full config (string) plotted per layer (no averaging)
        # create pivot where columns are full config strings (keeps layer alignment)
        pivot = df.pivot(index='layer_name', columns='config', values='num_cycles')
        # preserve file order of layers
        pivot = pivot.reindex(index=pd.unique(df['layer_name']))

        cols = list(pivot.columns)
        if max_configs:
            cols = cols[:max_configs]
            pivot = pivot[cols]

        n_configs = len(pivot.columns)
        x = np.arange(len(pivot.index))
        width = 0.8 / max(1, n_configs)

        plt.figure(figsize=(max(10, len(x)*0.25), 6))
        for i, col in enumerate(pivot.columns):
            y = pivot[col].values
            y_plot = np.where(np.isnan(y) & np.isnan(fill_missing), np.nan, np.nan_to_num(y, nan=fill_missing))
            plt.bar(x + i*width, y_plot, width=width, label=col)

        plt.xticks(x + width*(n_configs-1)/2, pivot.index, rotation=90)
        plt.xlabel("Layer")
        plt.ylabel("Number of Cycles")
        plt.title("Layer-wise Performance Comparison by Configuration")
        plt.legend()
        plt.tight_layout()
        plt.savefig(dst_path)
        plt.close()


def plot_memory(file_path, dst_path, compare_by=None, fill_missing=np.nan, max_configs=None):
    """
    Plot L1_tiling and L2_tiling (in kB) per layer.
    compare_by: None or one of ['num_cores', 'L1_mem', 'L2_mem']
    fill_missing: np.nan or 0 depending if you want gaps or zeros for missing combos
    max_configs: optionally limit number of configs plotted
    """
    df = pd.read_csv(file_path)
    df['config'] = df.apply(lambda x: f"Cores: {x['num_cores']} L1: {x['L1_mem']//1000} kB L2: {x['L2_mem']//1000} kB", axis=1)

    # convert tiling memory to kB
    df['L1_tiling_kB'] = df['L1_tiling'] / 1000.0
    df['L2_tiling_kB'] = df['L2_tiling'] / 1000.0

    layers = pd.Categorical(df['layer_name'], categories=pd.unique(df['layer_name']), ordered=True)

    # helper to make plots for both L1 and L2
    def _plot_one(metric_col, title_suffix):
        if compare_by in ['num_cores', 'L1_mem', 'L2_mem']:
            df_grouped = df.groupby(['layer_name', compare_by], as_index=False)[metric_col].mean()
            pivot = df_grouped.pivot(index='layer_name', columns=compare_by, values=metric_col)
            pivot = pivot.reindex(index=pd.unique(df['layer_name']))

            cols = list(pivot.columns)
            if max_configs:
                cols = cols[:max_configs]
                pivot = pivot[cols]

            col_labels = [_fmt_config_label(c, compare_by) for c in pivot.columns]
            n_configs = len(pivot.columns)
            x = np.arange(len(pivot.index))
            width = 0.8 / max(1, n_configs)

            plt.figure(figsize=(max(10, len(x)*0.25), 6))
            for i, col in enumerate(pivot.columns):
                y = pivot[col].values
                y_plot = np.where(np.isnan(y) & np.isnan(fill_missing), np.nan, np.nan_to_num(y, nan=fill_missing))
                plt.bar(x + i*width, y_plot, width=width, label=col_labels[i])

            plt.xticks(x + width*(n_configs-1)/2, pivot.index, rotation=90)
            plt.xlabel("Layer")
            plt.ylabel("Memory (kB)")
            plt.title(f"{title_suffix} by {compare_by}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(dst_path.replace(".png", f"_{title_suffix.replace(' ', '_')}.png"))
            plt.close()

        else:
            pivot = df.pivot(index='layer_name', columns='config', values=metric_col)
            pivot = pivot.reindex(index=pd.unique(df['layer_name']))

            cols = list(pivot.columns)
            if max_configs:
                cols = cols[:max_configs]
                pivot = pivot[cols]

            n_configs = len(pivot.columns)
            x = np.arange(len(pivot.index))
            width = 0.8 / max(1, n_configs)

            plt.figure(figsize=(max(10, len(x)*0.25), 6))
            for i, col in enumerate(pivot.columns):
                y = pivot[col].values
                y_plot = np.where(np.isnan(y) & np.isnan(fill_missing), np.nan, np.nan_to_num(y, nan=fill_missing))
                plt.bar(x + i*width, y_plot, width=width, label=col)

            plt.xticks(x + width*(n_configs-1)/2, pivot.index, rotation=90)
            plt.xlabel("Layer")
            plt.ylabel("Memory (kB)")
            plt.title(f"{title_suffix} by Configuration")
            plt.legend()
            plt.tight_layout()
            plt.savefig(dst_path.replace(".png", f"_{title_suffix.replace(' ', '_')}.png"))
            plt.close()

    # make both plots
    _plot_one('L1_tiling_kB', 'L1 Tiling')
    _plot_one('L2_tiling_kB', 'L2 Tiling')
    
    
    


def plot_metric_comparison(file_path, metric, group_by="Config", cores=8, l1=64, l2=512, compare_by="Case",
                           dst_path="plot.png", fill_missing=np.nan, max_groups=None):
    """
    Compare metric values (e.g. MACs, BOPs, Memory_kB) across cases for fixed configurations.
    
    Parameters:
        file_path: CSV path with columns ['Layers', metric, group_by, compare_by]
        metric: 'MACs', 'BOPs', 'Memory', etc.
        group_by: column to fix (e.g. 'Config', 'Layer', 'Block')
        compare_by: column to compare (e.g. 'Case')
        fill_missing: np.nan or 0 — how to handle missing entries
        max_groups: limit number of groups plotted (useful for large models)
        dst_path: output file for saved plot
    """
    df = pd.read_csv(file_path)
    df = df[(df["L1_mem"] == (l1*1000)) & (df["L2_mem"] == (l2*1000)) & (df["num_cores"] == cores)]

    df['L1_tiling'] = df['L1_tiling'] / 1000.0
    df['L2_tiling'] = df['L2_tiling'] / 1000.0
    # Build pivot (rows = group_by, columns = compare_by)
    pivot = df.pivot_table(index=group_by, columns=compare_by, values=metric, aggfunc='mean')

    # Maintain original order
    pivot = pivot.reindex(index=pd.unique(df[group_by]))

    # Optionally limit number of groups
    if max_groups:
        pivot = pivot.iloc[:max_groups]

    cases = list(pivot.columns)
    x = np.arange(len(pivot.index))
    width = 0.8 / max(1, len(cases))
    colors = [COLOR_PALETTE[i % len(COLOR_PALETTE)] for i in range(len(cases))]

    plt.style.use("default")
    plt.figure(figsize=(max(10, len(x) * 0.25), 6))
    for i, case in enumerate(cases):
        y = pivot[case].values
        y_plot = np.where(np.isnan(y) & np.isnan(fill_missing), np.nan,
                          np.nan_to_num(y, nan=fill_missing))
        plt.bar(x + i * width, y_plot, width=width, label=str(case), color=colors[i])

    plt.xticks(x + width * (len(cases) - 1) / 2, pivot.index, rotation=90)
    metric = metric.replace("_", " ")
    plt.title(f"{metric} Comparison Across Cases by Layers")
    if "cycle" in metric:
        plt.yscale("log")
        metric += " (log scale)"
    plt.ylabel(metric if "tiling" not in metric else f"{metric} [kB]")
    plt.legend(title=compare_by)
    plt.tight_layout()
    plt.savefig(dst_path)
    plt.close()