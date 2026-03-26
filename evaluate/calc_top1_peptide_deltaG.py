"""
为每个 pocket（data_id）选择 Top1 肽段，并计算其 ΔG（默认使用 FoldX AnalyseComplex）。

默认实现：
1) 从 gen_info.csv 中按 `--rank_by` 列选择每个 data_id 的 Top1 预测（最大值）。
2) 用受体 PDB + 该 Top1 肽段 PDB 拼成 complex PDB（输出到 {gen_path}/complex_top1/）。
3) 调用仓库现有 FoldX 管线 `evaluate/foldx/foldx_pipeline.py` 在 complex 上跑 AnalyseComplex，
   从输出的能量 summary 中取 `energy` 作为 ΔG。

你需要：
- FoldX 及其 Python wrapper（evaluate/foldx/FoldX.py 依赖）可用
- rec/pep PDB 链 ID 满足脚本预期（默认 receptor chain 为 R，肽段会在拼接时重命名为 L）

输出：
- {gen_path}/top1_deltaG.csv
"""

import os
import argparse
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio.PDB import PDBParser, PDBIO


def _get_gen_path(result_root: str, exp_name: str) -> str:
    if os.path.isdir(os.path.join(result_root, exp_name)):
        return os.path.join(result_root, exp_name)
    import re

    prefix = "^" + re.escape(exp_name) + r"_202[0-9_]+$"
    candidates = [x for x in os.listdir(result_root) if re.match(prefix, x)]
    if len(candidates) != 1:
        raise SystemExit(f"Need exactly one experiment dir. Found: {candidates}")
    return os.path.join(result_root, candidates[0])


def _pick_first_chain(structure):
    for model in structure:
        for chain in model:
            return chain
    raise ValueError("No chain found")


def combine_receptor_ligand_simple(protein_path: str, ligand_path: str, output_path: str, rec_chain_id: str = "R"):
    """
    Minimal combine: receptor stays as rec_chain_id, ligand chain is renamed to 'L'.
    """
    parser = PDBParser(QUIET=True)
    protein = parser.get_structure("rec", protein_path)[0]
    receptor_chains = list(protein.get_chains())
    if len(receptor_chains) == 0:
        raise ValueError(f"No chains in receptor file: {protein_path}")
    # require specified chain exists
    receptor_chain = None
    for c in receptor_chains:
        if c.id == rec_chain_id:
            receptor_chain = c
            break
    if receptor_chain is None:
        # fallback: if only one chain, use it
        if len(receptor_chains) == 1:
            receptor_chain = receptor_chains[0]
        else:
            raise ValueError(f"Receptor chain id {rec_chain_id!r} not found in {protein_path}")

    ligand_struct = parser.get_structure("lig", ligand_path)
    ligand_model = ligand_struct[0]
    ligand_chain = _pick_first_chain(ligand_model)
    ligand_chain.id = "L"

    # Add ligand chain to protein model
    # Bio.PDB allows adding chain objects directly
    protein.add(ligand_chain)

    io = PDBIO()
    io.set_structure(protein)
    io.save(output_path)


def _load_foldx_energy_map(foldx_energy_dir: str) -> Dict[str, float]:
    """
    foldx_pipeline stores energy results as pkl files under {foldx_energy_dir}/energy/{xxx}.pkl
    each pkl dict includes 'filename' without '.pdb' and 'energy' field.
    """
    energy_map = {}
    if not os.path.isdir(foldx_energy_dir):
        return energy_map
    for fname in os.listdir(foldx_energy_dir):
        if not fname.endswith(".pkl"):
            continue
        pkl_path = os.path.join(foldx_energy_dir, fname)
        with open(pkl_path, "rb") as f:
            d = pickle.load(f)
        # d['filename'] may exist; fallback to stem
        base = d.get("filename", fname.replace(".pkl", ""))
        energy_map[str(base)] = float(d.get("energy", np.nan))
    return energy_map


def evaluate_top1_deltaG(
    gen_path: str,
    gt_dir: str,
    rec_dir: str,
    out_csv: Optional[str] = None,
    sdf_dirname: str = "SDF",
    rank_by: str = "cfd_traj",
    require_succ: bool = False,
    foldx_chain_tuple: str = "R,L",
    foldx_dirname: str = "foldx_top1",
    num_workers: int = 64,
    repair: bool = True,
    skip_if_foldx_done: bool = True,
    rec_chain_id: str = "R",
    use_sdf_pred_pdb_only: bool = True,
):
    df = pd.read_csv(os.path.join(gen_path, "gen_info.csv"))
    sdf_dir = os.path.join(gen_path, sdf_dirname)
    if not os.path.isdir(sdf_dir):
        raise ValueError(f"SDF dir not found: {sdf_dir}")

    if rank_by not in df.columns:
        raise ValueError(f"rank_by={rank_by!r} not in gen_info.csv columns: {list(df.columns)}")

    if require_succ and "tag" in df.columns:
        # In this repo: succ samples often have tag as NaN; nonstd uses tag='nonstd'
        df = df[pd.isna(df["tag"])].copy()

    # Select top1 per pocket
    df[rank_by] = pd.to_numeric(df[rank_by], errors="coerce")
    df["__rank__"] = df[rank_by].fillna(-np.inf)
    idx = df.groupby("data_id")["__rank__"].idxmax()
    df_top1 = df.loc[idx].copy()

    complex_dir = os.path.join(gen_path, "complex_top1")
    os.makedirs(complex_dir, exist_ok=True)

    # Prepare complex PDBs
    for _, row in tqdm(df_top1.iterrows(), total=len(df_top1), desc="prepare top1 complex pdb"):
        data_id = row["data_id"]
        filename = row["filename"]
        if not isinstance(filename, str) or not filename.endswith(".pdb"):
            continue

        pep_pred_path = os.path.join(sdf_dir, filename)
        if not os.path.isfile(pep_pred_path):
            raise FileNotFoundError(f"Missing pred peptide PDB: {pep_pred_path}")

        rec_path = os.path.join(rec_dir, f"{data_id}_pro.pdb")
        if not os.path.isfile(rec_path):
            raise FileNotFoundError(f"Missing receptor PDB: {rec_path}")

        out_complex_path = os.path.join(complex_dir, filename)
        if os.path.isfile(out_complex_path):
            continue
        combine_receptor_ligand_simple(rec_path, pep_pred_path, out_complex_path, rec_chain_id=rec_chain_id)

    # Run FoldX
    foldx_root_dir = os.path.join(gen_path, foldx_dirname)
    energy_dir = os.path.join(foldx_root_dir, "energy")

    if (not skip_if_foldx_done) or (not os.path.isdir(energy_dir)) or (len(os.listdir(energy_dir)) == 0):
        from evaluate.foldx.foldx_pipeline import calc_foldx_score

        os.makedirs(foldx_root_dir, exist_ok=True)
        calc_foldx_score(
            complex_dir=complex_dir,
            output_dir=foldx_root_dir,
            num_workers=num_workers,
            chain_tuple=foldx_chain_tuple,
            repair=repair,
        )

    # Parse foldx energies
    energy_map = _load_foldx_energy_map(energy_dir)

    rows = []
    for _, row in df_top1.iterrows():
        data_id = row["data_id"]
        filename = row["filename"]
        base = str(filename).replace(".pdb", "")
        rows.append(
            {
                "data_id": data_id,
                "filename": filename,
                "rank_by": rank_by,
                "rank_value": float(row["__rank__"]),
                "deltaG": energy_map.get(base, np.nan),
                "foldx_energy_filename_key": base,
            }
        )

    out_df = pd.DataFrame(rows)
    if out_csv is None:
        out_csv = os.path.join(gen_path, "top1_deltaG.csv")
    out_df.to_csv(out_csv, index=False)
    return out_df


def main():
    parser = argparse.ArgumentParser(description="Compute Top1 peptide deltaG per pocket (default FoldX).")
    parser.add_argument("--gen_path", type=str, default=None)
    parser.add_argument("--result_root", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--gt_dir", type=str, default="data/pepbdb/files/peptides")  # kept for API symmetry (unused in default FoldX)
    parser.add_argument("--rec_dir", type=str, default="data/pepbdb/files/proteins")
    parser.add_argument("--rank_by", type=str, default="cfd_traj")
    parser.add_argument("--require_succ", action="store_true")
    parser.add_argument("--foldx_chain_tuple", type=str, default="R,L", help="FoldX --analyseComplexChains option")
    parser.add_argument("--num_workers", type=int, default=64)
    parser.add_argument("--repair", action="store_true", default=True)
    parser.add_argument("--skip_if_foldx_done", action="store_true", default=True)
    parser.add_argument("--out_csv", type=str, default=None)
    parser.add_argument("--rec_chain_id", type=str, default="R")

    args = parser.parse_args()
    if args.gen_path is None:
        if args.result_root is None or args.exp_name is None:
            raise SystemExit("Specify --gen_path or (--result_root and --exp_name).")
        args.gen_path = _get_gen_path(args.result_root, args.exp_name)

    evaluate_top1_deltaG(
        gen_path=args.gen_path,
        gt_dir=args.gt_dir,
        rec_dir=args.rec_dir,
        out_csv=args.out_csv,
        rank_by=args.rank_by,
        require_succ=args.require_succ,
        foldx_chain_tuple=args.foldx_chain_tuple,
        num_workers=args.num_workers,
        repair=args.repair,
        skip_if_foldx_done=args.skip_if_foldx_done,
        rec_chain_id=args.rec_chain_id,
    )


if __name__ == "__main__":
    main()

