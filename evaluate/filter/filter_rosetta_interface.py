import argparse
import json
import os
import pickle
from glob import glob
from tqdm import tqdm

import numpy as np
import pandas as pd

try:
    from .filter_foldx import plot_histograms, remove_worst_fraction
except ImportError:
    from filter_foldx import plot_histograms, remove_worst_fraction


def normalize_filename(name: str) -> str:
    """Normalize file identifiers across csv/pkl names."""
    if not isinstance(name, str):
        return ""
    x = name.strip()
    if x.endswith(".pdb"):
        x = x[:-4]
    if x.endswith("_score"):
        x = x[: -len("_score")]
    return x


def to_scalar_or_json(value):
    """Keep scalar values; serialize complex objects for csv output."""
    if isinstance(value, (str, int, float, bool, np.integer, np.floating)) or value is None:
        return value
    return json.dumps(value, ensure_ascii=True, sort_keys=True) # list, dict, ... -> string


def build_interface_csv(interface_dir: str, output_csv: str) -> pd.DataFrame:
    pkl_files = sorted(glob(os.path.join(interface_dir, "*.pkl")))
    if len(pkl_files) == 0:
        raise FileNotFoundError(f"No .pkl files found in: {interface_dir}")

    rows = []
    for path in tqdm(pkl_files):
        base = os.path.basename(path)
        # e.g. xxx_complex_score.pkl -> xxx_complex
        filename = normalize_filename(base.replace(".pkl", ""))

        with open(path, "rb") as f:
            data = pickle.load(f)
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict in {path}, got {type(data)}")

        row = {"filename": filename}
        for k, v in data.items():
            row[k] = to_scalar_or_json(v)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    return df


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build interface_analyzer.csv from pkl files, keep samples in foldx_filter.csv, "
            "then filter worst 30% dG and worst 10% interface SASA."
        )
    )
    parser.add_argument(
        "--interface_dir",
        type=str,
        required=True,
        help="Directory containing InterfaceAnalyzer pkl files.",
    )
    parser.add_argument(
        "--pre_filter_csv",
        type=str,
        default="",
        help="Optional csv path with 'filename' column.",
    )
    parser.add_argument(
        "--interface_csv",
        type=str,
        default="",
        help="Output csv path built from pkl files. Default: <interface_dir>/interface_analyzer.csv",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="",
        help="Filtered output csv path. Default: <interface_dir>/interface_analyzer_filter.csv",
    )
    parser.add_argument(
        "--energy_col",
        type=str,
        default="",
        help="RosettdG_separateda energy-change column. Larger is worse.",
    )
    parser.add_argument(
        "--sasa_col",
        type=str,
        default="dSASA_int",
        help="Interface SASA column. Smaller is worse.",
    )
    parser.add_argument(
        "--energy_remove_frac",
        type=float,
        default=0.30,
        help="Fraction removed by energy step (default: 0.30).",
    )
    parser.add_argument(
        "--sasa_remove_frac",
        type=float,
        default=0.10,
        help="Fraction removed by SASA step (default: 0.10).",
    )
    parser.add_argument(
        "--plot_bins",
        type=int,
        default=50,
        help="Number of bins used in histogram plotting (default: 50).",
    )
    args = parser.parse_args()

    interface_dir = args.interface_dir
    if not os.path.isdir(interface_dir):
        raise NotADirectoryError(f"interface_dir not found: {interface_dir}")
    # if not os.path.exists(args.pre_filter_csv):
        # raise FileNotFoundError(f"pre_filter_csv not found: {args.pre_filter_csv}")

    interface_csv = args.interface_csv or os.path.join(interface_dir, "interface_analyzer.csv")
    output_csv = args.output_csv or os.path.join(interface_dir, "interface_analyzer_filter.csv")
    output_dir = os.path.dirname(output_csv)
    os.makedirs(output_dir, exist_ok=True)

    # 1) Build interface_analyzer.csv from all pkl files
    df_interface = build_interface_csv(interface_dir, interface_csv)
    print(f"[Step 0] Built interface csv: {interface_csv}")
    print(f"[Step 0] Total pkl samples: {len(df_interface)}")
    hist_path = plot_histograms(
        df_interface,
        left_col=args.energy_col,
        right_col=args.sasa_col,
        output_dir=output_dir,
        bins=args.plot_bins,
        file_prefix="interface_hist",
        title_prefix="InterfaceAnalyzer",
    )
    print(f"[Plot] Saved histogram to: {hist_path}")

    # 2) Keep only filenames remaining in foldx_filter.csv
    if args.pre_filter_csv:
        if not os.path.exists(args.pre_filter_csv):
            raise FileNotFoundError(f"pre_filter_csv not found: {args.pre_filter_csv}")
        df_pre = pd.read_csv(args.pre_filter_csv)
        if "filename" not in df_pre.columns:
            raise ValueError(f"'filename' column not found in {args.pre_filter_csv}")

        keep_names = set(df_pre["filename"].map(normalize_filename))
        n_before_pre = len(df_interface)
        df_keep = df_interface[df_interface["filename"].map(normalize_filename).isin(keep_names)].copy()
        n_after_pre = len(df_keep)
        print(
            f"[Step 0.5] Keep pre_filter_csv filenames: "
            f"before={n_before_pre}, removed={n_before_pre - n_after_pre}, after={n_after_pre}"
        )

    # 3) Remove worst 30% by Rosetta energy change (larger dG is worse)
    df_energy, n1, n2, r1 = remove_worst_fraction(
        df_keep, score_col=args.energy_col, frac=args.energy_remove_frac, larger_is_worse=True
    )
    print(
        f"[Step 1] Energy filter by '{args.energy_col}': "
        f"before={n1}, removed={r1}, after={n2}"
    )

    # 4) Remove worst 10% by interface SASA (smaller dSASA is worse)
    df_final, n3, n4, r2 = remove_worst_fraction(
        df_energy, score_col=args.sasa_col, frac=args.sasa_remove_frac, larger_is_worse=False
    )
    print(
        f"[Step 2] SASA filter by '{args.sasa_col}': "
        f"before={n3}, removed={r2}, after={n4}"
    )

    df_final = df_final.reset_index(drop=True)
    hist_path_filtered = plot_histograms(
        df_final,
        left_col=args.energy_col,
        right_col=args.sasa_col,
        output_dir=output_dir,
        bins=args.plot_bins,
        suffix="_filtered",
        file_prefix="interface_hist",
        title_prefix="InterfaceAnalyzer",
    )
    print(f"[Plot] Saved filtered histogram to: {hist_path_filtered}")

    df_final.to_csv(output_csv, index=False)
    print(f"[Done] Saved filtered interface csv to: {output_csv}")


if __name__ == "__main__":
    main()
