"""
计算每个 pocket(data_id) 的肽序列 diversity（序列层面指标补齐脚本）。

默认实现（每个 pocket 内部，按生成序列两两比较）：
1) 从 gen_info.csv 读取每条预测的 `aaseq`。
2) 可选去重（默认对 unique 序列两两比较，避免重复采样造成的假多样性）。
3) 定义序列一致率（与 evaluate_pepfull.py 的 seq_identity 口径接近）：
   - identity(gt_like, pred_like) = (#位点相同且 pred_like != 'X') / L
   - 对称一致率 = 0.5 * (iden(A,B) + iden(B,A))
4) sequence_diversity = 1 - mean(sym_identity) over all pairs of unique sequences

输出：
- {gen_path}/sequence_diversity.csv
"""

import os
import argparse
from itertools import combinations
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


def _get_gen_path(result_root: str, exp_name: str) -> str:
    if os.path.isdir(os.path.join(result_root, exp_name)):
        return os.path.join(result_root, exp_name)
    import re

    prefix = "^" + re.escape(exp_name) + r"_202[0-9_]+$"
    candidates = [x for x in os.listdir(result_root) if re.match(prefix, x)]
    if len(candidates) != 1:
        raise SystemExit(f"Need exactly one experiment dir. Found: {candidates}")
    return os.path.join(result_root, candidates[0])


def seq_identity_asym(seq1: str, seq2: str) -> float:
    """
    口径与 evaluate_pepfull.py 的 seq_identity 接近（seq2 是可能包含 X 的预测序列）。
    """
    if len(seq1) != len(seq2):
        return np.nan
    L = len(seq1)
    if L == 0:
        return np.nan
    return sum((a == b) and (b != "X") for a, b in zip(seq1, seq2)) / L


def seq_identity_sym(seq_a: str, seq_b: str) -> float:
    id_ab = seq_identity_asym(seq_a, seq_b)
    id_ba = seq_identity_asym(seq_b, seq_a)
    if np.isnan(id_ab) or np.isnan(id_ba):
        return np.nan
    return 0.5 * (id_ab + id_ba)


def _compute_diversity(seqs: list, dedupe: bool = True, max_pairs: Optional[int] = None):
    seqs = [s for s in seqs if isinstance(s, str) and len(s) > 0]
    if len(seqs) == 0:
        return np.nan, 0, 0

    total = len(seqs)
    if dedupe:
        seqs = sorted(set(seqs))
    unique_n = len(seqs)
    if unique_n < 2:
        # diversity defined over pairs; only one unique seq -> 0 diversity
        return 0.0, total, unique_n

    pairs = list(combinations(seqs, 2))
    if max_pairs is not None and len(pairs) > max_pairs:
        # deterministic sample: take first max_pairs
        pairs = pairs[:max_pairs]

    id_list = []
    for a, b in pairs:
        idv = seq_identity_sym(a, b)
        if idv == idv:  # not nan
            id_list.append(idv)

    if len(id_list) == 0:
        return np.nan, total, unique_n
    mean_id = float(np.mean(id_list))
    diversity = 1.0 - mean_id
    return diversity, total, unique_n


def evaluate_from_gen_info(
    gen_path: str,
    out_csv: Optional[str] = None,
    dedupe: bool = True,
    max_pairs: Optional[int] = None,
    only_succ: bool = False,
):
    df = pd.read_csv(os.path.join(gen_path, "gen_info.csv"))
    if only_succ and "tag" in df.columns:
        df = df[pd.isna(df["tag"])].copy()

    if "aaseq" not in df.columns:
        raise ValueError("gen_info.csv must have column `aaseq`.")

    rows = []
    for data_id, group in tqdm(df.groupby("data_id"), total=df["data_id"].nunique(), desc="sequence diversity"):
        seqs = group["aaseq"].tolist()
        diversity, n_total, n_unique = _compute_diversity(seqs, dedupe=dedupe, max_pairs=max_pairs)
        unique_ratio = (n_unique / n_total) if n_total > 0 else np.nan
        rows.append(
            {
                "data_id": data_id,
                "sequence_diversity": diversity,
                "unique_ratio": unique_ratio,
                "n_sequences_total": n_total,
                "n_sequences_unique": n_unique,
            }
        )

    out_df = pd.DataFrame(rows)
    if out_csv is None:
        out_csv = os.path.join(gen_path, "sequence_diversity.csv")
    out_df.to_csv(out_csv, index=False)
    return out_df


def main():
    parser = argparse.ArgumentParser(description="Compute peptide sequence diversity per pocket.")
    parser.add_argument("--gen_path", type=str, default=None)
    parser.add_argument("--result_root", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--out_csv", type=str, default=None)
    parser.add_argument("--dedupe", action="store_true", default=True)
    parser.add_argument("--no_dedupe", action="store_true")
    parser.add_argument("--max_pairs", type=int, default=None)
    parser.add_argument("--only_succ", action="store_true")
    args = parser.parse_args()

    if args.gen_path is None:
        if args.result_root is None or args.exp_name is None:
            raise SystemExit("Specify --gen_path or (--result_root and --exp_name).")
        args.gen_path = _get_gen_path(args.result_root, args.exp_name)

    dedupe_flag = args.dedupe and (not args.no_dedupe)
    evaluate_from_gen_info(
        gen_path=args.gen_path,
        out_csv=args.out_csv,
        dedupe=dedupe_flag,
        max_pairs=args.max_pairs,
        only_succ=args.only_succ,
    )


if __name__ == "__main__":
    main()

