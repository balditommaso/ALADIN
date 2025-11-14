import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === GLOBAL HIGH-CONTRAST PALETTE ===
# Works well on light/dark backgrounds, colorblind safe
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

figsize = (12, 4)

FONTS = {
    "title": 18,
    "label": 16,
    "ticks": 14,
    "legend": 14,
}



def _align_cases(df, value_col, compare_by):
    """Align all cases by the CSV order of Layers, fill missing with 0."""
    layers = list(dict.fromkeys(df["Layers"]))
    cases = sorted(df[compare_by].unique())

    aligned = []
    for c in cases:
        subset = df[df[compare_by] == c][["Layers", value_col]]
        subset = subset.set_index("Layers").reindex(layers, fill_value=0).reset_index()
        subset["Case"] = c
        aligned.append(subset)

    return layers, cases, aligned


def _get_colors(n):
    """Cycle through the palette for n cases."""
    return [COLOR_PALETTE[i % len(COLOR_PALETTE)] for i in range(n)]


def plot_macs(file_path, dst_path, compare_by="Case"):
    df = pd.read_csv(file_path)
    df = df[df["MACs"] != 0]
    df["Layers"] = df["Layers"].str.replace(r"^GlobalAveragePool_", "avgpool_", regex=True)

    plt.style.use("default")

    if compare_by and compare_by in df.columns:
        layers, cases, aligned = _align_cases(df, "MACs", compare_by)
        x = np.arange(len(layers))
        width = 0.8 / len(cases)
        colors = _get_colors(len(cases))

        fig, ax = plt.subplots(figsize=figsize)
        for i, subset in enumerate(aligned):
            ax.bar(
                x + i * width,
                subset["MACs"],
                width=width,
                label=str(subset["Case"].iloc[0]),
                color=colors[i],
            )

        ax.set_xticks(x + width * (len(cases) - 1) / 2)
        ax.set_xticklabels(layers, rotation=90, fontsize=FONTS["ticks"])
        ax.set_ylabel("MACs", fontsize=FONTS["label"])
        ax.set_title("MACs per Layer (comparison)", fontsize=FONTS["title"])
        ax.legend(title=compare_by, fontsize=FONTS["legend"], title_fontsize=FONTS["legend"])
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(df["Layers"], df["MACs"], color=COLOR_PALETTE[0])
        ax.set_xticklabels(df["Layers"], rotation=90, fontsize=FONTS["ticks"])
        ax.set_ylabel("MACs", fontsize=FONTS["label"])
        ax.set_title("MACs per Layer", fontsize=FONTS["title"])

    plt.tight_layout()
    plt.savefig(dst_path)
    plt.close(fig)



def plot_bops(file_path, dst_path, compare_by="Case"):
    df = pd.read_csv(file_path)
    df = df[df["BOPs"] != 0]
    df = df[~df["Layers"].str.contains("Relu", case=False, na=False)]
    df["Layers"] = df["Layers"].str.replace(r"^GlobalAveragePool_", "avgpool_", regex=True)

    plt.style.use("default")

    if compare_by and compare_by in df.columns:
        layers, cases, aligned = _align_cases(df, "BOPs", compare_by)
        x = np.arange(len(layers))
        width = 0.8 / len(cases)
        colors = _get_colors(len(cases))

        fig, ax = plt.subplots(figsize=figsize)
        for i, subset in enumerate(aligned):
            ax.bar(
                x + i * width,
                subset["BOPs"],
                width=width,
                label=str(subset["Case"].iloc[0]),
                color=colors[i],
            )

        ax.set_xticks(x + width * (len(cases) - 1) / 2)
        ax.set_xticklabels(layers, rotation=90, fontsize=FONTS["ticks"])
        ax.set_yscale("log")
        ax.set_ylabel("BOPs (log scale)", fontsize=FONTS["label"])
        ax.set_title("BOPs per Layer (comparison)", fontsize=FONTS["title"])
        ax.legend(title=compare_by, fontsize=FONTS["legend"], title_fontsize=FONTS["legend"])
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(df["Layers"], df["BOPs"], color=COLOR_PALETTE[0])
        ax.set_xticklabels(df["Layers"], rotation=90, fontsize=FONTS["ticks"])
        ax.set_yscale("log")
        ax.set_ylabel("BOPs", fontsize=FONTS["label"])
        ax.set_title("BOPs per Layer", fontsize=FONTS["title"])

    plt.tight_layout()
    plt.savefig(dst_path)
    plt.close(fig)


def plot_memory(file_path, dst_path, compare_by="Case"):
    df = pd.read_csv(file_path)
    df = df[df["Memory"] != 0]
    df = df[~df["Layers"].str.contains("Relu", case=False, na=False)]
    df["Layers"] = df["Layers"].str.replace(r"^GlobalAveragePool_", "avgpool_", regex=True)
    df["Memory_kB"] = df["Memory"] / 1000.0

    plt.style.use("default")

    if compare_by and compare_by in df.columns:
        layers, cases, aligned = _align_cases(df, "Memory_kB", compare_by)
        x = np.arange(len(layers))
        width = 0.8 / len(cases)
        colors = _get_colors(len(cases))

        fig, ax = plt.subplots(figsize=figsize)
        for i, subset in enumerate(aligned):
            ax.bar(
                x + i * width,
                subset["Memory_kB"],
                width=width,
                label=str(subset["Case"].iloc[0]),
                color=colors[i],
            )

        ax.set_xticks(x + width * (len(cases) - 1) / 2)
        ax.set_xticklabels(layers, rotation=90, fontsize=FONTS["ticks"])
        ax.set_ylabel("Memory [kB]", fontsize=FONTS["label"])
        ax.set_title("Memory Footprint per Layer (comparison)", fontsize=FONTS["title"])
        ax.legend(title=compare_by, fontsize=FONTS["legend"], title_fontsize=FONTS["legend"])
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(df["Layers"], df["Memory_kB"], color=COLOR_PALETTE[0])
        ax.set_xticklabels(df["Layers"], rotation=90, fontsize=FONTS["ticks"])
        ax.set_ylabel("Memory [kB]", fontsize=FONTS["label"])
        ax.set_title("Memory Footprint per Layer", fontsize=FONTS["title"])

    plt.tight_layout()
    plt.savefig(dst_path)
    plt.close(fig)

