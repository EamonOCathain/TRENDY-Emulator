#!/usr/bin/env python3
"""
make_heatmap.py
---------------
Read a table (metrics in header row, variables in first column) and render a
colour-coded heatmap. Cells are auto-sized to be roughly square, column labels
are shown on top at 45°.

Usage:
  python make_heatmap.py --input scores.csv --output ilamb_heatmap.png
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def main():
    ap = argparse.ArgumentParser(description="Create a heatmap from metric table.")
    ap.add_argument("--input", required=True, help="Path to CSV/TSV file.")
    ap.add_argument("--output", default="heatmap.png", help="Output image (png/pdf/svg).")
    ap.add_argument("--sep", default=None,
                    help="Column separator. If omitted, inferred by extension "
                         "(.tsv/.txt -> tab, else comma).")
    ap.add_argument("--index-col", type=int, default=0,
                    help="Index column (default: 0 = first column).")
    ap.add_argument("--title", default="Benchmark Scores (Higher = Better)",
                    help="Figure title.")
    ap.add_argument("--vmin", type=float, default=0.3, help="Lower bound for colour scale.")
    ap.add_argument("--vmax", type=float, default=1.0, help="Upper bound for colour scale.")
    ap.add_argument("--cmap", default="RdYlGn", help="Matplotlib/Seaborn colormap name.")
    ap.add_argument("--annot", action="store_true", help="Show numeric values in cells.")
    ap.add_argument("--fmt", default=".3f", help="Number format for annotations.")
    ap.add_argument("--dpi", type=int, default=300, help="DPI for raster outputs (PNG).")
    ap.add_argument("--na", default="", help="String treated as NA (e.g., 'NA').")
    args = ap.parse_args()

    in_path = Path(args.input)
    if args.sep is None:
        sep = "\t" if in_path.suffix.lower() in {".tsv", ".tab", ".txt"} else ","
    else:
        sep = args.sep

    # Load table, set index to first column (variable names)
    df = pd.read_csv(in_path, sep=sep, na_values=[args.na] if args.na else None)
    df = df.set_index(df.columns[args.index_col])
    df = df.apply(pd.to_numeric, errors="coerce")

    # --- Determine colormap ---
    cmap_name = args.cmap.lower()
    if cmap_name in {"rdwhgn", "greenwhitered", "gwr"}:
        cmap = LinearSegmentedColormap.from_list("RdWhGn", ["red", "white", "green"])
    elif cmap_name in {"rdwhgn_r", "redwhitegreen"}:
        cmap = LinearSegmentedColormap.from_list("RdWhGn_r", ["green", "white", "red"])
    else:
        cmap = args.cmap

    # --- Auto-size for ~square cells (shorter width), with reasonable caps ---
    n_rows, n_cols = df.shape
    cell = 0.55  # inches per cell
    fig_w = max(5.0, min(12.0, n_cols * cell))
    fig_h = max(5.0, min(12.0, n_rows * cell))

    plt.figure(figsize=(fig_w, fig_h))
    ax = sns.heatmap(
        df,
        cmap=cmap,
        vmin=args.vmin,
        vmax=args.vmax,
        annot=args.annot,
        fmt=args.fmt if args.annot else "",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Performance Score"},
        square=False
    )
    ax.set_title(args.title)
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Put column labels on TOP, rotated 45°, and align to cell centers
    ax.tick_params(axis="x", bottom=False, labelbottom=False, top=True, labeltop=True)
    ax.set_xticks(np.arange(len(df.columns)) + 0.5)
    ax.set_xticklabels(df.columns, rotation=45, ha="right")

    # Keep row labels horizontal and centered
    ax.set_yticks(np.arange(len(df.index)) + 0.5)
    ax.set_yticklabels(df.index, rotation=0)

    plt.tight_layout()

    # Save
    suffix = Path(args.output).suffix.lower()
    dpi = args.dpi if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff"} else None
    plt.savefig(args.output, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {args.output}")

if __name__ == "__main__":
    main()