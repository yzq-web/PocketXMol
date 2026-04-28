import argparse
import math
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def remove_worst_fraction(df, score_col, frac, larger_is_worse=True):
    """
    Remove the worst `frac` fraction by `score_col`.
    - larger_is_worse=True: larger value is worse (e.g. FoldX energy / Rosetta dG).
    - larger_is_worse=False: smaller value is worse (e.g. interface SASA).
    NaN is always treated as worst.
    """
    if score_col not in df.columns:
        raise ValueError(f"Column '{score_col}' not found. Available: {list(df.columns)}")
    if not (0 <= frac < 1):
        raise ValueError(f"frac must be in [0, 1), got {frac}")

    n_before = len(df)
    if n_before == 0:
        return df.copy(), n_before, n_before, 0

    n_remove = math.floor(n_before * frac)
    if n_remove <= 0:
        return df.copy(), n_before, n_before, 0

    series = pd.to_numeric(df[score_col], errors="coerce")
    sort_key = series.fillna(np.inf if larger_is_worse else -np.inf)
    # Put worst samples first, then drop the first n_remove.
    # larger_is_worse=True  -> descending (large to small)
    # larger_is_worse=False -> ascending  (small to large)
    order = sort_key.sort_values(ascending=not larger_is_worse).index
    keep_index = order[n_remove:]
    kept = df.loc[keep_index].copy()

    return kept, n_before, len(kept), n_remove


def _safe_name(name):
    return "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in str(name))


def plot_histograms(
    df,
    left_col,
    right_col,
    output_dir,
    bins=50,
    suffix="",
    file_prefix="foldx_hist",
    title_prefix="FoldX",
):
    """
    双列直方图；默认文件名/标题面向 FoldX，Rosetta 侧传入
    `file_prefix="interface_hist"`, `title_prefix="InterfaceAnalyzer"`。
    """
    for col in (left_col, right_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found for plotting. Available: {list(df.columns)}")

    os.makedirs(output_dir, exist_ok=True)
    hist_path = os.path.join(
        output_dir,
        f"{file_prefix}_{_safe_name(left_col)}_{_safe_name(right_col)}{suffix}.png",
    )

    left_values = pd.to_numeric(df[left_col], errors="coerce").dropna()
    right_values = pd.to_numeric(df[right_col], errors="coerce").dropna()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    title_left = f"{title_prefix}: {left_col}" if title_prefix else str(left_col)
    title_right = f"{title_prefix}: {right_col}" if title_prefix else str(right_col)

    axes[0].hist(
        left_values,
        bins=bins,
        color="#4C72B0",
        alpha=0.85,
        edgecolor="black",
    )
    axes[0].set_title(title_left, fontsize=18)
    axes[0].set_xlabel('', fontsize=16) # left_col
    axes[0].set_ylabel("Count", fontsize=16)
    axes[0].tick_params(axis='x', labelsize=16)
    axes[0].tick_params(axis='y', labelsize=16)

    axes[1].hist(
        right_values,
        bins=bins,
        color="#55A868",
        alpha=0.85,
        edgecolor="black",
    )
    axes[1].set_title(title_right, fontsize=18)
    axes[1].set_xlabel('', fontsize=16) # right_col
    axes[1].set_ylabel("Count", fontsize=16)
    axes[1].tick_params(axis='x', labelsize=16)
    axes[1].tick_params(axis='y', labelsize=16)

    fig.tight_layout()
    fig.savefig(hist_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return hist_path


def main():
    parser = argparse.ArgumentParser(
        description="Filter FoldX results: remove worst 30%% by binding energy, then worst 30%% by stability."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to foldx.csv exported from evaluate/foldx/foldx_pipeline.py",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="",
        help="Output path for filtered CSV. Default: <input_dir>/foldx_filter.csv",
    )
    parser.add_argument(
        "--binding_col",
        type=str,
        default="energy",
        help="Column used for binding-energy filtering. Higher is worse.",
    )
    parser.add_argument(
        "--stability_col",
        type=str,
        default="stability",
        help="Column used for stability filtering. Higher is worse.",
    )
    parser.add_argument(
        "--remove_frac",
        type=float,
        default=0.30,
        help="Fraction removed at each filtering step (default: 0.30).",
    )
    parser.add_argument(
        "--plot_bins",
        type=int,
        default=50,
        help="Number of bins used in histogram plotting (default: 50).",
    )
    args = parser.parse_args()

    input_csv = args.input_csv
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input csv not found: {input_csv}")

    output_csv = args.output_csv
    if output_csv == "":
        output_csv = os.path.join(os.path.dirname(input_csv), "foldx_filter.csv")

    df = pd.read_csv(input_csv)
    input_dir = os.path.dirname(input_csv)
    output_dir = os.path.dirname(output_csv)
    os.makedirs(output_dir, exist_ok=True)

    print(f"[Step 0] Loaded samples: {len(df)}")
    hist_path = plot_histograms(
        df,
        left_col=args.binding_col,
        right_col=args.stability_col,
        output_dir=output_dir,
        bins=args.plot_bins,
    )
    print(f"[Plot] Saved histogram to: {hist_path}")

    # Step 1: remove worst 30% by binding energy
    df_step1, n0, n1, r1 = remove_worst_fraction(
        df, score_col=args.binding_col, frac=args.remove_frac, larger_is_worse=True
    )
    print(
        f"[Step 1] Binding filter by '{args.binding_col}': "
        f"before={n0}, removed={r1}, after={n1}"
    )

    # Step 2: remove worst 30% by stability
    df_step2, n1b, n2, r2 = remove_worst_fraction(
        df_step1, score_col=args.stability_col, frac=args.remove_frac, larger_is_worse=True
    )
    print(
        f"[Step 2] Stability filter by '{args.stability_col}': "
        f"before={n1b}, removed={r2}, after={n2}"
    )

    # Keep a stable, human-friendly order
    # if "filename" in df_step2.columns:
    #     df_step2 = df_step2.sort_values("filename").reset_index(drop=True)
    # else:
    #     df_step2 = df_step2.reset_index(drop=True)
    df_step2 = df_step2.reset_index(drop=True)
    hist_path_filtered = plot_histograms(
        df_step2,
        left_col=args.binding_col,
        right_col=args.stability_col,
        output_dir=output_dir,
        bins=args.plot_bins,
        suffix="_filtered",
    )
    print(f"[Plot] Saved filtered histogram to: {hist_path_filtered}")

    df_step2.to_csv(output_csv, index=False)
    print(f"[Done] Saved filtered CSV to: {output_csv}")


if __name__ == "__main__":
    main()
