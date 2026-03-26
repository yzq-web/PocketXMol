"""
计算肽段“非标准残基”原子 RMSD（论文常见指标补齐脚本）。

默认实现口径（可通过参数调整）：
1) 用 GT 肽段标记哪些 residue 是 non-standard（resname 不属于 20 种标准氨基酸）。
2) 先用 GT 中“标准残基”的 CA 原子对 pred/gt 做叠合（Superimposer）。
3) 在同一个叠合变换下，仅计算属于 GT non-standard residue 的原子 RMSD
   （只比较重原子；按 atom name 一一配对，不存在的原子会被跳过）。

输入数据约定（与仓库输出一致）：
- gen_path: 实验目录，包含 `gen_info.csv` 和 `SDF/`
- pred peptide PDB: `SDF/{filename}`（filename 来自 gen_info.csv 的 filename 列）
- gt peptide PDB:
  - 默认: {gt_dir}/{data_id}_pep.pdb
  - 可选: --use_sdf_gt 使用 `SDF/{filename.replace('.pdb','_gt.pdb')}`

输出：
- 默认写入 {gen_path}/nonstd_rmsd.csv
"""

import os
import argparse
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio.PDB import PDBParser, Superimposer


STANDARD_AA_3 = {
    "ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU",
    "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR",
}


def _pick_first_chain(structure):
    for model in structure:
        for chain in model:
            return chain
    raise ValueError("No chain found in structure")


def _get_chain(structure, chain_id: Optional[str] = None):
    if chain_id is None:
        return _pick_first_chain(structure)
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                return chain
    raise ValueError(f"Chain id {chain_id!r} not found")


def _is_standard_res(res) -> bool:
    # resname is 3-letter code
    return res.resname in STANDARD_AA_3


def _heavy_atom_name(a) -> str:
    # Bio.PDB Atom.get_name() keeps padding, normalize by strip
    return a.get_name().strip()


def _is_hydrogen_atom(a) -> bool:
    name = _heavy_atom_name(a).upper()
    if name.startswith("H"):
        return True
    # fallback: Atom.element may be empty in some PDBs
    elem = getattr(a, "element", None)
    if elem is not None and str(elem).upper() == "H":
        return True
    return False


def _atoms_by_name(res) -> Dict[str, object]:
    d = {}
    for a in res.get_atoms():
        if _is_hydrogen_atom(a):
            continue
        d[_heavy_atom_name(a)] = a
    return d


def _collect_residues(chain):
    # Only protein residues: hetero flag ' ' and id[0] == ' '
    # Bio.PDB: residue.id is like (' ', resseq, icode)
    residues = []
    for res in chain.get_residues():
        if res.id[0] != " ":
            continue
        residues.append(res)
    return residues


def _calc_nonstd_rmsd(
    pred_pdb: str,
    gt_pdb: str,
    pep_chain_id: Optional[str] = None,
    align_ca_only_on_gt_standard: bool = True,
) -> Tuple[float, int, int]:
    """
    return: (nonstd_rmsd, n_atoms_compared, n_nonstd_residues_in_gt)
    """
    parser = PDBParser(QUIET=True)
    s_pred = parser.get_structure("pred", pred_pdb)
    s_gt = parser.get_structure("gt", gt_pdb)

    c_pred = _get_chain(s_pred, pep_chain_id)
    c_gt = _get_chain(s_gt, pep_chain_id)

    pred_res = _collect_residues(c_pred)
    gt_res = _collect_residues(c_gt)

    if len(pred_res) == 0 or len(gt_res) == 0:
        return (np.nan, 0, 0)
    if len(pred_res) != len(gt_res):
        # hard requirement for this simplified index-based mapping
        return (np.nan, 0, 0)

    # 1) alignment CA atoms from GT
    fixed_ca_atoms = []
    moving_ca_atoms = []
    for r_pred, r_gt in zip(pred_res, gt_res):
        if align_ca_only_on_gt_standard:
            if not _is_standard_res(r_gt):
                continue
        if "CA" in r_pred and "CA" in r_gt:
            fixed_ca_atoms.append(r_gt["CA"])
            moving_ca_atoms.append(r_pred["CA"])

    # if no standard CA found, fallback to all CA
    if len(fixed_ca_atoms) == 0 or len(moving_ca_atoms) == 0:
        for r_pred, r_gt in zip(pred_res, gt_res):
            if "CA" in r_pred and "CA" in r_gt:
                fixed_ca_atoms.append(r_gt["CA"])
                moving_ca_atoms.append(r_pred["CA"])

    if len(fixed_ca_atoms) == 0 or len(moving_ca_atoms) == 0:
        return (np.nan, 0, 0)

    sup = Superimposer()
    sup.set_atoms(fixed_ca_atoms, moving_ca_atoms)

    # 2) compute RMSD on GT non-standard residue atoms
    nonstd_pairs = []
    nonstd_res_count = 0
    # collect atoms to be transformed (pred atoms)
    pred_atoms_to_transform = []

    for r_pred, r_gt in zip(pred_res, gt_res):
        if not _is_standard_res(r_gt):
            nonstd_res_count += 1
            gt_atom_map = _atoms_by_name(r_gt)
            pred_atom_map = _atoms_by_name(r_pred)
            shared_names = sorted(set(gt_atom_map.keys()) & set(pred_atom_map.keys()))
            for name in shared_names:
                gt_atom = gt_atom_map[name]
                pred_atom = pred_atom_map[name]
                nonstd_pairs.append((gt_atom, pred_atom))
                pred_atoms_to_transform.append(pred_atom)

    if len(nonstd_pairs) == 0:
        return (np.nan, 0, nonstd_res_count)

    # Apply the CA-based transformation to pred atoms of interest
    sup.apply(pred_atoms_to_transform)

    sq = 0.0
    for gt_atom, pred_atom in nonstd_pairs:
        d = gt_atom.get_coord() - pred_atom.get_coord()
        sq += float(np.dot(d, d))

    n = len(nonstd_pairs)
    rmsd = float(np.sqrt(sq / n))
    return rmsd, n, nonstd_res_count


def _get_gen_path(result_root: str, exp_name: str) -> str:
    if os.path.isdir(os.path.join(result_root, exp_name)):
        return os.path.join(result_root, exp_name)
    import re

    prefix = "^" + re.escape(exp_name) + r"_202[0-9_]+$"
    candidates = [x for x in os.listdir(result_root) if re.match(prefix, x)]
    if len(candidates) != 1:
        raise SystemExit(f"Need exactly one experiment dir. Found: {candidates}")
    return os.path.join(result_root, candidates[0])


def evaluate_from_gen_info(
    gen_path: str,
    gt_dir: str,
    rec_dir: Optional[str] = None,  # kept for API symmetry; not used
    out_csv: Optional[str] = None,
    use_sdf_gt: bool = False,
    pep_chain_id: Optional[str] = None,
    align_ca_only_on_gt_standard: bool = True,
):
    df = pd.read_csv(os.path.join(gen_path, "gen_info.csv"))
    sdf_dir = os.path.join(gen_path, "SDF")

    rows = []
    for _, line in tqdm(df.iterrows(), total=len(df), desc="nonstd residue RMSD"):
        data_id = line["data_id"]
        filename = line["filename"]
        if not isinstance(filename, str) or not filename.endswith(".pdb"):
            continue

        pred_path = os.path.join(sdf_dir, filename)
        if not os.path.isfile(pred_path):
            rows.append({"data_id": data_id, "filename": filename, "nonstd_rmsd": np.nan, "n_nonstd_atoms": 0})
            continue

        if use_sdf_gt:
            gt_path = os.path.join(sdf_dir, filename.replace(".pdb", "_gt.pdb"))
        else:
            gt_path = os.path.join(gt_dir, f"{data_id}_pep.pdb")

        if not os.path.isfile(gt_path):
            rows.append({"data_id": data_id, "filename": filename, "nonstd_rmsd": np.nan, "n_nonstd_atoms": 0})
            continue

        rmsd, n_atoms, n_nonstd_res = _calc_nonstd_rmsd(
            pred_pdb=pred_path,
            gt_pdb=gt_path,
            pep_chain_id=pep_chain_id,
            align_ca_only_on_gt_standard=align_ca_only_on_gt_standard,
        )

        rows.append(
            {
                "data_id": data_id,
                "filename": filename,
                "nonstd_rmsd": rmsd,
                "n_nonstd_atoms": n_atoms,
                "n_nonstd_res": n_nonstd_res,
            }
        )

    out_df = pd.DataFrame(rows)
    if out_csv is None:
        out_csv = os.path.join(gen_path, "nonstd_rmsd.csv")
    out_df.to_csv(out_csv, index=False)
    return out_df


def main():
    parser = argparse.ArgumentParser(description="Compute non-standard residues RMSD for peptide docking.")
    parser.add_argument("--gen_path", type=str, default=None)
    parser.add_argument("--result_root", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--gt_dir", type=str, default="data/pepbdb/files/peptides")
    parser.add_argument("--use_sdf_gt", action="store_true")
    parser.add_argument("--out_csv", type=str, default=None)
    parser.add_argument("--pep_chain_id", type=str, default=None, help="Pep chain id in pred/gt PDB (default: first chain)")
    parser.add_argument("--align_ca_only_on_gt_standard", action="store_true", default=True)
    parser.add_argument("--no_align_ca_only_on_gt_standard", action="store_true")

    args = parser.parse_args()
    if args.gen_path is None:
        if args.result_root is None or args.exp_name is None:
            raise SystemExit("Specify --gen_path or (--result_root and --exp_name).")
        args.gen_path = _get_gen_path(args.result_root, args.exp_name)

    align_flag = args.align_ca_only_on_gt_standard and (not args.no_align_ca_only_on_gt_standard)

    evaluate_from_gen_info(
        gen_path=args.gen_path,
        gt_dir=args.gt_dir,
        out_csv=args.out_csv,
        use_sdf_gt=args.use_sdf_gt,
        pep_chain_id=args.pep_chain_id,
        align_ca_only_on_gt_standard=align_flag,
    )


if __name__ == "__main__":
    main()

