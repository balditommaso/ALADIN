import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === COLOR PALETTE ===
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

# === FONT SIZES ===
FONTS = {
    "title": 18,
    "label": 16,
    "ticks": 14,
    "legend": 14,
}

def _fmt_config_label(cfg_val, compare_by=None):
    """Nicely format config labels when compare_by is a memory column."""
    if compare_by == 'L1_mem' or compare_by == 'L2_mem':
        return f"{int(cfg_val)//1000} kB"
    return str(cfg_val)


def plot_performance(file_path, dst_path, compare_by=None, fill_missing=np.nan, max_configs=None):
    df = pd.read_csv(file_path)
    df['config'] = df.apply(lambda x: f"Cores: {x['num_cores']} L1: {x['L1_mem']//1000} kB L2: {x['L2_mem']//1000} kB", axis=1)
    layers = pd.Categorical(df['layer_name'], categories=pd.unique(df['layer_name']), ordered=True)

    if compare_by in ['num_cores', 'L1_mem', 'L2_mem']:
        df_grouped = df.groupby(['layer_name', compare_by], as_index=False)['num_cycles'].mean()
        pivot = df_grouped.pivot(index='layer_name', columns=compare_by, values='num_cycles').reindex(index=pd.unique(df['layer_name']))

        if max_configs:
            pivot = pivot.iloc[:, :max_configs]

        col_labels = [_fmt_config_label(c, compare_by) for c in pivot.columns]
        n_configs = len(pivot.columns)
        x = np.arange(len(pivot.index))
        width = 0.8 / max(1, n_configs)

        fig, ax = plt.subplots(figsize=(max(10, len(x)*0.25), 6))
        for i, col in enumerate(pivot.columns):
            y = pivot[col].values
            y_plot = np.where(np.isnan(y) & np.isnan(fill_missing), np.nan, np.nan_to_num(y, nan=fill_missing))
            ax.bar(x + i*width, y_plot, width=width, label=col_labels[i], color=COLOR_PALETTE[i % len(COLOR_PALETTE)])

        ax.set_xticks(x + width*(n_configs-1)/2)
        ax.set_xticklabels(pivot.index, rotation=90, fontsize=FONTS["ticks"])
        ax.set_xlabel("Layer", fontsize=FONTS["label"])
        ax.set_ylabel("Number of Cycles (log scale)", fontsize=FONTS["label"])
        # ax.set_yscale("log")
        ax.set_title(f"Layer-wise Performance Comparison by {compare_by}", fontsize=FONTS["title"])
        ax.legend(fontsize=FONTS["legend"], title_fontsize=FONTS["legend"])
        plt.tight_layout()
        plt.savefig(dst_path)
        plt.close(fig)

    else:
        pivot = df.pivot(index='layer_name', columns='config', values='num_cycles').reindex(index=pd.unique(df['layer_name']))
        if max_configs:
            pivot = pivot.iloc[:, :max_configs]

        n_configs = len(pivot.columns)
        x = np.arange(len(pivot.index))
        width = 0.8 / max(1, n_configs)

        fig, ax = plt.subplots(figsize=(max(10, len(x)*0.25), 6))
        for i, col in enumerate(pivot.columns):
            y = pivot[col].values
            y_plot = np.where(np.isnan(y) & np.isnan(fill_missing), np.nan, np.nan_to_num(y, nan=fill_missing))
            ax.bar(x + i*width, y_plot, width=width, label=col, color=COLOR_PALETTE[i % len(COLOR_PALETTE)])

        ax.set_xticks(x + width*(n_configs-1)/2)
        ax.set_xticklabels(pivot.index, rotation=90, fontsize=FONTS["ticks"])
        ax.set_xlabel("Layer", fontsize=FONTS["label"])
        ax.set_ylabel("Number of Cycles", fontsize=FONTS["label"])
        ax.set_title("Layer-wise Performance Comparison by Configuration", fontsize=FONTS["title"])
        ax.legend(fontsize=FONTS["legend"], title_fontsize=FONTS["legend"])
        plt.tight_layout()
        plt.savefig(dst_path)
        plt.close(fig)


def plot_memory(file_path, dst_path, compare_by=None, fill_missing=np.nan, max_configs=None):
    df = pd.read_csv(file_path)
    df['config'] = df.apply(lambda x: f"Cores: {x['num_cores']} L1: {x['L1_mem']//1000} kB L2: {x['L2_mem']//1000} kB", axis=1)
    df['L1_tiling_kB'] = df['L1_tiling'] / 1000.0
    df['L2_tiling_kB'] = df['L2_tiling'] / 1000.0
    layers = pd.Categorical(df['layer_name'], categories=pd.unique(df['layer_name']), ordered=True)

    def _plot_one(metric_col, title_suffix):
        if compare_by in ['num_cores', 'L1_mem', 'L2_mem']:
            df_grouped = df.groupby(['layer_name', compare_by], as_index=False)[metric_col].mean()
            pivot = df_grouped.pivot(index='layer_name', columns=compare_by, values=metric_col).reindex(index=pd.unique(df['layer_name']))
            if max_configs:
                pivot = pivot.iloc[:, :max_configs]

            col_labels = [_fmt_config_label(c, compare_by) for c in pivot.columns]
            n_configs = len(pivot.columns)
            x = np.arange(len(pivot.index))
            width = 0.8 / max(1, n_configs)

            fig, ax = plt.subplots(figsize=(max(10, len(x)*0.25), 6))
            for i, col in enumerate(pivot.columns):
                y = pivot[col].values
                y_plot = np.where(np.isnan(y) & np.isnan(fill_missing), np.nan, np.nan_to_num(y, nan=fill_missing))
                ax.bar(x + i*width, y_plot, width=width, label=col_labels[i], color=COLOR_PALETTE[i % len(COLOR_PALETTE)])

            ax.set_xticks(x + width*(n_configs-1)/2)
            ax.set_xticklabels(pivot.index, rotation=90, fontsize=FONTS["ticks"])
            ax.set_xlabel("Layer", fontsize=FONTS["label"])
            ax.set_ylabel("Memory (kB)", fontsize=FONTS["label"])
            ax.set_title(f"{title_suffix} by {compare_by}", fontsize=FONTS["title"])
            ax.legend(fontsize=FONTS["legend"], title_fontsize=FONTS["legend"])
            plt.tight_layout()
            plt.savefig(dst_path.replace(".png", f"_{title_suffix.replace(' ', '_')}.png"))
            plt.close(fig)

        else:
            pivot = df.pivot(index='layer_name', columns='config', values=metric_col).reindex(index=pd.unique(df['layer_name']))
            if max_configs:
                pivot = pivot.iloc[:, :max_configs]

            n_configs = len(pivot.columns)
            x = np.arange(len(pivot.index))
            width = 0.8 / max(1, n_configs)

            fig, ax = plt.subplots(figsize=(max(10, len(x)*0.25), 6))
            for i, col in enumerate(pivot.columns):
                y = pivot[col].values
                y_plot = np.where(np.isnan(y) & np.isnan(fill_missing), np.nan, np.nan_to_num(y, nan=fill_missing))
                ax.bar(x + i*width, y_plot, width=width, label=col, color=COLOR_PALETTE[i % len(COLOR_PALETTE)])

            ax.set_xticks(x + width*(n_configs-1)/2)
            ax.set_xticklabels(pivot.index, rotation=90, fontsize=FONTS["ticks"])
            ax.set_xlabel("Layer", fontsize=FONTS["label"])
            ax.set_ylabel("Memory (kB)", fontsize=FONTS["label"])
            ax.set_title(f"{title_suffix} by Configuration", fontsize=FONTS["title"])
            ax.legend(fontsize=FONTS["legend"], title_fontsize=FONTS["legend"])
            plt.tight_layout()
            plt.savefig(dst_path.replace(".png", f"_{title_suffix.replace(' ', '_')}.png"))
            plt.close(fig)

    _plot_one('L1_tiling_kB', 'L1 Tiling')
    _plot_one('L2_tiling_kB', 'L2 Tiling')


def plot_metric_comparison(file_path, metric, group_by="Config", cores=8, l1=64, l2=512, compare_by="Case",
                           dst_path="plot.png", fill_missing=np.nan, max_groups=None):
    df = pd.read_csv(file_path)
    df = df[(df["L1_mem"] == (l1*1000)) & (df["L2_mem"] == (l2*1000)) & (df["num_cores"] == cores)]
    df['L1_tiling'] = df['L1_tiling'] / 1000.0
    df['L2_tiling'] = df['L2_tiling'] / 1000.0

    pivot = df.pivot_table(index=group_by, columns=compare_by, values=metric, aggfunc='mean').reindex(index=pd.unique(df[group_by]))
    if max_groups:
        pivot = pivot.iloc[:max_groups]

    cases = list(pivot.columns)
    x = np.arange(len(pivot.index))
    width = 0.8 / max(1, len(cases))
    colors = [COLOR_PALETTE[i % len(COLOR_PALETTE)] for i in range(len(cases))]

    fig, ax = plt.subplots(figsize=(max(10, len(x) * 0.25), 6))
    for i, case in enumerate(cases):
        y = pivot[case].values
        y_plot = np.where(np.isnan(y) & np.isnan(fill_missing), np.nan, np.nan_to_num(y, nan=fill_missing))
        ax.bar(x + i * width, y_plot, width=width, label=str(case), color=colors[i])

    ax.set_xticks(x + width * (len(cases) - 1) / 2)
    ax.set_xticklabels(pivot.index, rotation=90, fontsize=FONTS["ticks"])
    metric_label = metric.replace("_", " ")
    ax.set_title(f"{metric_label} Comparison Across Cases by Layers", fontsize=FONTS["title"])
    if "cycle" in metric_label.lower():
        ax.set_yscale("log")
        metric_label += " (log scale)"
    ax.set_ylabel(metric_label if "tiling" not in metric_label else f"{metric_label} [kB]", fontsize=FONTS["label"])
    ax.legend(title=compare_by, fontsize=FONTS["legend"], title_fontsize=FONTS["legend"])
    plt.tight_layout()
    plt.savefig(dst_path)
    plt.close(fig)
