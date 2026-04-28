import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

try:
    from .filter_foldx import remove_worst_fraction
except ImportError:
    from filter_foldx import remove_worst_fraction


def _normalize_name(x: str) -> str:
    if not isinstance(x, str):
        return ""
    y = x.strip()
    if y.endswith(".pdb"):
        y = y[:-4]
    return y


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _safe_name(name):
    return "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in str(name))


def plot_three_histograms(
    df: pd.DataFrame,
    output_dir: str,
    bins: int = 50,
    suffix: str = "",
    file_prefix: str = "flexpepdock_hist",
):
    cols = ["refine_score", "score_only", "min_only"]
    for col in cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found for plotting. Available: {list(df.columns)}")

    os.makedirs(output_dir, exist_ok=True)
    hist_path = os.path.join(
        output_dir,
        f"{file_prefix}_{_safe_name(cols[0])}_{_safe_name(cols[1])}_{_safe_name(cols[2])}{suffix}.png",
    )

    values = [pd.to_numeric(df[col], errors="coerce").dropna() for col in cols]
    colors = ["#4C72B0", "#55A868", "#C44E52"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for i, col in enumerate(cols):
        axes[i].hist(
            values[i],
            bins=bins,
            color=colors[i],
            alpha=0.85,
            edgecolor="black",
        )
        axes[i].set_title(f"FlexPepDock: {col}", fontsize=18)
        axes[i].set_xlabel('', fontsize=16) # col
        axes[i].set_ylabel("Count", fontsize=16)
        axes[i].tick_params(axis='x', labelsize=16)
        axes[i].tick_params(axis='y', labelsize=16)

    fig.tight_layout()
    fig.savefig(hist_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return hist_path


def load_and_merge_scores(rosetta_dir: str) -> pd.DataFrame:
    refine_path = os.path.join(rosetta_dir, 'refine_score.csv')
    rescore_path = os.path.join(rosetta_dir, 'score_only.csv')
    min_path = os.path.join(rosetta_dir, 'min_only.csv')

    for p in [refine_path, rescore_path, min_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"File not found: {p}")

    df_refine = pd.read_csv(refine_path)
    df_rescore = pd.read_csv(rescore_path)
    df_min = pd.read_csv(min_path)

    required_cols = {"in_pdb", "total_score"}
    for name, df in [
        ('refine_score', df_refine),
        ('score_only', df_rescore),
        ('min_only', df_min),
    ]:
        miss = required_cols.difference(df.columns)
        if miss:
            raise ValueError(f"{name} missing columns: {sorted(miss)}")

    df_refine = (
        df_refine[["in_pdb", "total_score"]]
        .drop_duplicates(subset=["in_pdb"], keep="first")
        .rename(columns={"total_score": "refine_score"})
    )
    df_rescore = (
        df_rescore[["in_pdb", "total_score"]]
        .drop_duplicates(subset=["in_pdb"], keep="first")
        .rename(columns={"total_score": "score_only"}) # rescoring_score
    )
    df_min = (
        df_min[["in_pdb", "total_score"]]
        .drop_duplicates(subset=["in_pdb"], keep="first")
        .rename(columns={"total_score": "min_only"}) # minimization_score
    )

    merged = (
        df_refine.merge(df_rescore, on="in_pdb", how="inner")
        .merge(df_min, on="in_pdb", how="inner")
        .copy()
    )
    merged["in_pdb"] = merged["in_pdb"].map(_normalize_name)
    merged = merged.drop_duplicates(subset=["in_pdb"], keep="first").reset_index(drop=True)
    return merged


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Filter FlexPepDock outputs: remove worst 30% refine_score globally, "
            "then remove samples with min_score > 100 or rescore_score > 1000."
        )
    )
    parser.add_argument(
        "--rosetta_dir",
        type=str,
        required=True,
        help="Directory containing refine_score.csv, score_only.csv and min_only.csv.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="",
        help="Filtered output csv path.",
    )
    # parser.add_argument(
    #     "--summary_csv",
    #     type=str,
    #     default="",
    #     help="Optional merged-all-samples csv path before filtering.",
    # )
    parser.add_argument(
        "--pre_filter_csv",
        type=str,
        default="",
        help="Optional csv path with 'filename' column. Keep only matched samples by in_pdb/filename.",
    )
    parser.add_argument(
        "--remove_frac",
        type=float,
        default=0.30,
        help="Fraction removed by refinement score globally (default: 0.30).",
    )
    parser.add_argument(
        "--min_threshold",
        type=float,
        default=100.0,
        help="Remove samples with min_only > this threshold (default: 100).",
    )
    parser.add_argument(
        "--rescore_threshold",
        type=float,
        default=1000.0,
        help="Remove samples with score_only > this threshold (default: 1000).",
    )
    parser.add_argument(
        "--plot_bins",
        type=int,
        default=50,
        help="Number of bins used in histogram plotting (default: 50).",
    )
    args = parser.parse_args()

    rosetta_dir = args.rosetta_dir
    if not os.path.isdir(rosetta_dir):
        raise NotADirectoryError(f"rosetta_dir not found: {rosetta_dir}")

    output_dir = os.path.dirname(args.output_csv)
    os.makedirs(output_dir, exist_ok=True)
    # output_csv = os.path.join(output_dir, "flexpepdock_filter.csv")
    # summary_csv = args.summary_csv
    summary_csv = os.path.join(rosetta_dir, "summary_score.csv")

    df_all = load_and_merge_scores(rosetta_dir=rosetta_dir)
    print(f"[Step 0] merged samples: {len(df_all)}")

    for c in ["refine_score", "score_only", "min_only"]:
        df_all[c] = _to_numeric(df_all[c])

    if args.pre_filter_csv:
        if not os.path.exists(args.pre_filter_csv):
            raise FileNotFoundError(f"pre_filter_csv not found: {args.pre_filter_csv}")
        df_pre = pd.read_csv(args.pre_filter_csv)
        if "filename" not in df_pre.columns:
            raise ValueError(f"'filename' column not found in {args.pre_filter_csv}")

        keep_names = set(df_pre["filename"].map(_normalize_name))
        n_before_pre = len(df_all)
        df_all = df_all[df_all["in_pdb"].map(_normalize_name).isin(keep_names)].copy()
        n_after_pre = len(df_all)
        print(
            f"[Step 0.5] keep pre_filter_csv filenames: "
            f"before={n_before_pre}, removed={n_before_pre - n_after_pre}, after={n_after_pre}"
        )

    if summary_csv:
        df_all.to_csv(summary_csv, index=False)
        print(f"[Step 0] saved merged summary: {summary_csv}")

    hist_path = plot_three_histograms(
        df_all,
        output_dir=output_dir,
        bins=args.plot_bins,
    )
    print(f"[Plot] saved histogram (before filter): {hist_path}")

    df_step1, n0, n1, r1 = remove_worst_fraction(
        df_all,
        score_col="refine_score",
        frac=args.remove_frac,
        larger_is_worse=True,
    )
    df_step1 = df_step1.reset_index(drop=True)
    print(
        f"[Step 1] remove worst {args.remove_frac:.0%} refine_score globally: "
        f"before={n0}, removed={r1}, after={n1}"
    )

    keep_mask = (df_step1["min_only"] <= args.min_threshold) & (
        df_step1["score_only"] <= args.rescore_threshold
    )
    df_final = df_step1[keep_mask].reset_index(drop=True)
    n2 = len(df_final)
    print(
        f"[Step 2] threshold filter (min_only <= {args.min_threshold}, "
        f"score_only <= {args.rescore_threshold}): "
        f"before={n1}, removed={n1 - n2}, after={n2}"
    )

    hist_path_filtered = plot_three_histograms(
        df_final,
        output_dir=output_dir,
        bins=args.plot_bins,
        suffix="_filtered",
    )
    print(f"[Plot] saved histogram (after filter): {hist_path_filtered}")

    df_final.to_csv(args.output_csv, index=False)
    print(f"[Done] saved filtered csv: {args.output_csv}")


if __name__ == "__main__":
    main()
