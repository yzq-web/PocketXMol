"""
在项目根目录执行:
  python extensions/pdb_combined_datatrain.py --db_name pepbdb
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# from evaluate.utils_eval import combine_chains, combine_receptor_ligand  # noqa: E402

def _default_paths(db_name: str, workspace: str) -> tuple[str, str, str, str]:
    base = os.path.join(workspace, "data_train", db_name, "files")
    meta_csv = os.path.join(workspace, "data_train", db_name, "dfs", "meta_filter.csv")
    pep_dir = os.path.join(base, "peptides")
    pro_dir = os.path.join(base, "proteins")
    out_dir = os.path.join(base, "proteins_combined")
    return meta_csv, pep_dir, pro_dir, out_dir

def _load_pdb_body_lines(pdb_path):
    body_lines = []
    with open(pdb_path, "r") as f:
        for line in f:
            rec = line[:6].strip()
            if rec in {"END", "ENDMDL"}:
                continue
            body_lines.append(line if line.endswith("\n") else line + "\n")
    return body_lines

def merge_protein_peptide_pdb(protein_path, peptide_path, output_path):
    """
    Merge protein and peptide PDB files into a single PDB file.
    """
    protein_lines = _load_pdb_body_lines(protein_path)
    peptide_lines = _load_pdb_body_lines(peptide_path)
    with open(output_path, "w") as f:
        f.writelines(protein_lines)
        f.writelines(peptide_lines)
        f.write("END\n")

def build_one_complex(
    data_id: str,
    pep_dir: str,
    pro_dir: str,
    out_dir: str,
    overwrite: bool,
) -> str | None:
    pep_path = os.path.join(pep_dir, f"{data_id}_pep.pdb")
    pro_path = os.path.join(pro_dir, f"{data_id}_pro.pdb")
    out_path = os.path.join(out_dir, f"{data_id}_combined.pdb")

    if not os.path.isfile(pep_path):
        print(f"[skip] missing peptide: {pep_path}", file=sys.stderr)
        return None
    if not os.path.isfile(pro_path):
        print(f"[skip] missing protein: {pro_path}", file=sys.stderr)
        return None
    if os.path.isfile(out_path) and not overwrite:
        return out_path

    merge_protein_peptide_pdb(pro_path, pep_path, out_path)

    # fd, tmp_pro = tempfile.mkstemp(suffix="_rec_combchain.pdb", prefix="pxm_")
    # os.close(fd)
    # try:
    #     combine_chains(pro_path, save_path=tmp_pro) # 将受体多链先合并为单链 R
    #     combine_receptor_ligand(tmp_pro, pep_path, out_path) # 将肽链（重命名为 L）写入同一 PDB
    # finally:
    #     if os.path.isfile(tmp_pro):
    #         os.remove(tmp_pro)
    
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge proteins + peptides into complex PDBs (proteins_combined).")
    parser.add_argument("--db_name", type=str, required=True, help="例如 pepbdb、cpep")
    parser.add_argument(
        "--workspace",
        type=str,
        default=_ROOT,
        help="项目根目录(含 data_train), 默认为本仓库根",
    )
    parser.add_argument(
        "--meta_csv",
        type=str,
        default=None,
        help="example: data_train/{db}/dfs/meta_filter.csv",
    )
    parser.add_argument("--overwrite", action="store_true", help="已存在输出时仍重新生成")
    args = parser.parse_args()

    meta_default, pep_dir, pro_dir, out_dir = _default_paths(args.db_name, args.workspace)
    meta_path = args.meta_csv if args.meta_csv else meta_default

    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"找不到 meta CSV: {meta_path}")

    df = pd.read_csv(meta_path)
    if "data_id" not in df.columns:
        raise ValueError(f"{meta_path} 中需要列 data_id, 当前列: {list(df.columns)}")

    os.makedirs(out_dir, exist_ok=True)

    data_ids = df["data_id"].astype(str).unique().tolist()
    ok = 0
    for data_id in tqdm(data_ids, desc=f"{args.db_name} complex"):
        if build_one_complex(data_id, pep_dir, pro_dir, out_dir, args.overwrite):
            ok += 1
    print(f"完成: {ok}/{len(data_ids)} 条写入 {out_dir}")


if __name__ == "__main__":
    main()
