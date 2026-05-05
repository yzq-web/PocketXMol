import argparse
import os
from typing import List
from tqdm import tqdm

import matplotlib.pyplot as plt
import pandas as pd
from Bio.Align import PairwiseAligner
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP

_SS_ALIGNER = PairwiseAligner()
_SS_ALIGNER.mode = "global"
_SS_ALIGNER.match_score = 1.0
_SS_ALIGNER.mismatch_score = -1.0
_SS_ALIGNER.open_gap_score = -2.0
_SS_ALIGNER.extend_gap_score = -0.5

def get_ss(pdb_path: str) -> str:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pep", pdb_path)[0]
    try:
        dssp = DSSP(structure, pdb_path, dssp="mkdssp")
    except FileNotFoundError:
        dssp_path = os.path.expanduser("~/anaconda3/envs/mol/bin/mkdssp")
        dssp = DSSP(structure, pdb_path, dssp=dssp_path)
    ss = [dssp[key][2] for key in dssp.keys()]
    return "".join(ss)


def count_residues(pdb_path: str) -> int:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pep", pdb_path)[0]
    return sum(1 for _ in structure.get_residues())


def ss_similarity(ss_a: str, ss_b: str) -> float:
    if not ss_a and not ss_b:
        return 1.0
    if not ss_a or not ss_b:
        return 0.0

    best_alignment = _SS_ALIGNER.align(ss_a, ss_b)[0]
    a_coords, b_coords = best_alignment.aligned

    matches = 0
    aligned_len = 0
    for (a_start, a_end), (b_start, b_end) in zip(a_coords, b_coords):
        block_len = min(a_end - a_start, b_end - b_start)
        aligned_len += block_len
        for offset in range(block_len):
            if ss_a[a_start + offset] == ss_b[b_start + offset]:
                matches += 1

    # The full alignment length includes matched/mismatched columns and gap columns.
    total_len = len(ss_a) + len(ss_b) - aligned_len
    if total_len <= 0:
        return 0.0
    return matches / total_len


def load_passed_data_ids(input_dir: str) -> set:
    check_csv_path = os.path.join(input_dir, "meta", "peptide_all_check.csv")
    if not os.path.isfile(check_csv_path):
        raise FileNotFoundError(f"Check file not found: {check_csv_path}")

    check_df = pd.read_csv(check_csv_path, usecols=["data_id", "pass_all_checks"])
    passed_mask = check_df["pass_all_checks"].astype(str).str.lower().eq("true")
    return set(check_df.loc[passed_mask, "data_id"].astype(str))


def collect_peptide_ss(input_dir: str, valid_data_ids: set) -> pd.DataFrame:
    complex_dir = os.path.join(input_dir, "complex")
    if not os.path.isdir(complex_dir):
        raise FileNotFoundError(f"Complex directory not found: {complex_dir}")

    records = []
    for subdir_name in tqdm(sorted(os.listdir(complex_dir)), desc="Collecting peptide ss", leave=False):
        if subdir_name not in valid_data_ids:
            continue
        peptide_pdb = os.path.join(complex_dir, subdir_name, "peptide.pdb")
        if not os.path.isfile(peptide_pdb):
            continue

        pep_len = count_residues(peptide_pdb)
        ss = get_ss(peptide_pdb)
        records.append(
            {
                "data_id": subdir_name,
                "pep_len": pep_len,
                "ss": ss,
            }
        )

    return pd.DataFrame(records, columns=["data_id", "pep_len", "ss"])


def build_sim_ss_column(df: pd.DataFrame) -> pd.DataFrame:
    sim_ss_list: List[str] = []
    sim_ratio_list: List[float] = []
    data = df.to_dict("records")

    for i, query in enumerate(tqdm(data, desc="Finding best sim_ss", leave=False)):
        query_len = int(query["pep_len"])
        query_ss = str(query["ss"])
        best_data_id = ""
        best_ratio = -1.0

        for j, target in enumerate(data):
            if i == j:
                continue
            target_len = int(target["pep_len"])
            if abs(target_len - query_len) > 1:
                continue

            sim = ss_similarity(query_ss, str(target["ss"]))
            if sim > best_ratio:
                best_ratio = sim
                best_data_id = str(target["data_id"])

        sim_ss_list.append(best_data_id)
        sim_ratio_list.append(best_ratio if best_ratio >= 0 else 0.0)

    out_df = df.copy()
    out_df["sim_ss"] = sim_ss_list
    out_df["sim_ratio"] = sim_ratio_list
    return out_df


def save_sim_ratio_histogram(df: pd.DataFrame, output_path: str) -> None:
    sim_ratio = pd.to_numeric(df["sim_ratio"], errors="coerce").dropna()

    plt.figure(figsize=(8, 5))
    if len(sim_ratio) > 0:
        plt.hist(sim_ratio, bins=20, range=(0, 1), edgecolor="black")
    else:
        plt.text(0.5, 0.5, "No valid sim_ratio values", ha="center", va="center")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
    plt.title("sim_ratio distribution")
    plt.xlabel("sim_ratio")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Collect peptide secondary structures from <input_dir>/complex/*/peptide.pdb "
            "and save to <input_dir>/meta/peptide_ss.csv."
        )
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing complex/ and meta/.",
    )
    args = parser.parse_args()

    valid_data_ids = load_passed_data_ids(args.input_dir)
    df = collect_peptide_ss(args.input_dir, valid_data_ids)
    df = build_sim_ss_column(df)

    meta_dir = os.path.join(args.input_dir, "meta")
    os.makedirs(meta_dir, exist_ok=True)
    output_path = os.path.join(meta_dir, "peptide_ss.csv")
    df.to_csv(output_path, index=False)
    hist_path = os.path.join(meta_dir, "hist_ss.png")
    save_sim_ratio_histogram(df, hist_path)
    print(f"Saved: {output_path}")
    print(f"Saved: {hist_path}")
    print(f"Total peptides: {len(df)}")


if __name__ == "__main__":
    main()
