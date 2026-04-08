import argparse
import os
import sys
import tempfile
from multiprocessing import Pool, cpu_count
import pandas as pd
from tqdm import tqdm

# _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# _ROOT = os.path.dirname(_SCRIPT_DIR)
# if _ROOT not in sys.path:
#     sys.path.insert(0, _ROOT)
sys.path.append("/home/yangziqing/PocketXMol")
from evaluate.evaluate_mols import get_dir_from_prefix
from evaluate.utils_eval import combine_chains, combine_receptor_ligand


def _load_pdb_body_lines(pdb_path):
    body_lines = []
    with open(pdb_path, "r") as f:
        for line in f:
            rec = line[:6].strip()
            if rec in {"END", "ENDMDL"}:
                continue
            body_lines.append(line if line.endswith("\n") else line + "\n")
    return body_lines


def _replace_chain_id(line, new_chain_id):
    if len(line) < 22:
        return line
    return line[:21] + new_chain_id + line[22:]


def _renumber_protein_chain_ids(lines):
    """
    Rename protein chain IDs starting from 'R' in encounter order:
    R, S, T, ... (single-character PDB chain IDs).
    """
    mapping = {}
    next_code = ord("R")
    updated = []
    for line in lines:
        if line.startswith(("ATOM", "HETATM", "TER")) and len(line) >= 22:
            old_chain = line[21]
            if old_chain not in mapping:
                if next_code > ord("Z"):
                    raise ValueError("Protein has too many chains to rename within 'R'..'Z'")
                mapping[old_chain] = chr(next_code)
                next_code += 1
            line = _replace_chain_id(line, mapping[old_chain])
        updated.append(line)
    return updated


def _rename_peptide_chain_to_l(lines):
    updated = []
    for line in lines:
        if line.startswith(("ATOM", "HETATM", "TER")) and len(line) >= 22:
            line = _replace_chain_id(line, "L")
        updated.append(line)
    return updated


def _insert_chain_separators(lines):
    """
    Ensure there is a TER separator between chains and at section end.
    Existing TER lines are dropped and regenerated to avoid duplicates.
    """
    out = []
    prev_chain = None
    for line in lines:
        rec = line[:6].strip()
        if rec == "TER":
            continue
        if rec in {"ATOM", "HETATM"} and len(line) >= 22:
            chain_id = line[21]
            if prev_chain is not None and chain_id != prev_chain:
                out.append("TER\n")
            out.append(line)
            prev_chain = chain_id
        else:
            out.append(line)
    if prev_chain is not None:
        out.append("TER\n")
    return out


def merge_protein_peptide_pdb(protein_path, peptide_path, output_path):
    """
    Merge protein and peptide PDB files into one complex PDB.
    - Protein chain IDs: R, S, T, ...
    - Peptide chain ID: L
    - Add explicit TER separators between chains.
    """
    protein_lines = _load_pdb_body_lines(protein_path)
    peptide_lines = _load_pdb_body_lines(peptide_path)

    protein_lines = _insert_chain_separators(_renumber_protein_chain_ids(protein_lines))
    peptide_lines = _insert_chain_separators(_rename_peptide_chain_to_l(peptide_lines))

    with open(output_path, "w") as f:
        f.writelines(protein_lines)
        f.writelines(peptide_lines)
        f.write("END\n")


def merge_protein_peptide_pdb_with_utils(protein_path, peptide_path, output_path):
    """
    Merge by evaluate.utils_eval:
    1) Merge all protein chains into one receptor chain 'R'.
    2) Rename peptide chain to 'L' and combine into one complex PDB.
    """

    fd, tmp_pro = tempfile.mkstemp(suffix="_rec_combined_R.pdb", prefix="pxm_")
    os.close(fd)
    try:
        combine_chains(protein_path, combined_chain_id="R", save_path=tmp_pro)
        combine_receptor_ligand(tmp_pro, peptide_path, output_path, rec_chain_id="R")
    finally:
        if os.path.isfile(tmp_pro):
            os.remove(tmp_pro)


def combine_pred_pair(protein_path, peptide_path, output_path, use_utils_eval=False):
    """
    Wrapper API for prediction-time PDB combination.
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if use_utils_eval:
        merge_protein_peptide_pdb_with_utils(protein_path, peptide_path, output_path)
    else:
        merge_protein_peptide_pdb(protein_path, peptide_path, output_path)


def _process_merge_task(task):
    protein_path = task["protein_path"]
    peptide_path = task["peptide_path"]
    output_path = task["output_path"]
    use_utils_eval = task["use_utils_eval"]

    if not os.path.isfile(protein_path):
        return {"status": "skipped", "message": f"Skipped: {protein_path} not found"}
    if not os.path.isfile(peptide_path):
        return {"status": "skipped", "message": f"Skipped: {peptide_path} not found"}

    try:
        combine_pred_pair(
            protein_path=protein_path,
            peptide_path=peptide_path,
            output_path=output_path,
            use_utils_eval=use_utils_eval,
        )
        return {"status": "ok", "message": ""}
    except Exception as e:
        return {"status": "failed", "message": str(e)}


def _run_merge_tasks(tasks, n_cores, desc):
    iterator = None
    pool = None
    if n_cores == 1:
        iterator = map(_process_merge_task, tasks)
    else:
        pool = Pool(processes=n_cores)
        iterator = pool.imap_unordered(_process_merge_task, tasks)

    ok = 0
    skipped = 0
    failed = 0
    try:
        for row_result in tqdm(iterator, total=len(tasks), desc=desc):
            status = row_result["status"]
            if status == "ok":
                ok += 1
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1
            if row_result["message"].startswith("Skipped:"):
                print(row_result["message"])
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return ok, skipped, failed


def batch_merge_from_gen_info(gen_path, protein_dir, mode="gen", use_utils_eval=False, n_cores=1, remove_data_ids=None):
    """
    Batch merge by reading {gen_path}/gen_info.csv with pandas.
    - mode='gen': only keep rows with tag is np.nan, merge filename -> gen_combined(_rosetta)
    - mode='gt': use df_gen.drop_duplicates(subset=["data_id"], keep="first").copy(),
      merge filename_gt -> gt_combined(_rosetta)
    - mode='best_cfd': keep rows with tag is np.nan first, then keep per-data_id row
      with max cfd_traj, merge filename -> best_cfd_combined(_rosetta)
    """

    gen_info_path = os.path.join(gen_path, "gen_info.csv")
    peptide_dir = os.path.join(gen_path, "SDF")

    mode = str(mode).lower()
    if mode not in {"gen", "gt", "best_cfd"}:
        raise ValueError(f"Unsupported mode: {mode}. Expected one of ['gen', 'gt', 'best_cfd'].")

    if use_utils_eval:
        output_dir = os.path.join(gen_path, f"{mode}_combined_rosetta")
    else:
        output_dir = os.path.join(gen_path, f"{mode}_combined")
    os.makedirs(output_dir, exist_ok=True)

    df_gen = pd.read_csv(gen_info_path)
    remove_data_ids = set(remove_data_ids or [])

    def _to_gt_filename(filename):
        base, ext = os.path.splitext(str(filename))
        if ext == "":
            ext = ".pdb"
        return f"{base}_gt{ext}"

    if mode == "gen":
        # Keep generation rows with tag == np.nan.
        df_mode = df_gen[df_gen["tag"].isna()].copy()
    elif mode == "gt":
        df_mode = df_gen.drop_duplicates(subset=["data_id"], keep="first").copy()
    else:  # mode == "best_cfd"
        df_mode = df_gen[df_gen["tag"].isna()].copy()
        idx = df_mode.groupby("data_id")["cfd_traj"].idxmax()
        df_mode = df_mode.loc[idx].reset_index(drop=True).copy()

    tasks = []
    for _, row in df_mode.iterrows():
        data_id = str(row["data_id"])
        if data_id in remove_data_ids:
            continue

        filename = str(row["filename"])
        if mode == "gt":
            filename = _to_gt_filename(filename)

        protein_path = os.path.join(protein_dir, f"{data_id}_pro.pdb")
        peptide_path = os.path.join(peptide_dir, filename)
        file_stem, _ = os.path.splitext(filename)
        output_path = os.path.join(output_dir, f"{file_stem}_combined.pdb")
        tasks.append(
            {
                "protein_path": protein_path,
                "peptide_path": peptide_path,
                "output_path": output_path,
                "use_utils_eval": use_utils_eval,
            }
        )

    if n_cores == -1:
        n_cores = max(1, cpu_count() - 1)
    n_cores = max(1, int(n_cores))

    ok, skipped, failed = _run_merge_tasks(tasks, n_cores, f"merge {mode} pdb")

    print(
        "Done. "
        f"mode={mode}, rows={len(df_mode)}, success={ok}, skipped={skipped}, failed={failed}"
    )
    print(f"{mode}_combined_dir:", output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Batch merge receptor/peptide PDBs by reading gen_info.csv."
    )
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment prefix or full exp dir name")
    parser.add_argument("--result_root", type=str, required=True, help="Root directory containing experiment outputs")
    parser.add_argument("--gt_dir", type=str, required=True, help="Protein(receptor) PDB directory")
    parser.add_argument(
        "--mode",
        type=str,
        default="gen",
        help="Comma-separated modes to process in order, e.g. 'gen', 'gt,gen', or 'best_cfd,gt'.",
    )
    parser.add_argument(
        "--use_utils_eval",
        action="store_true",
        help="Use evaluate.utils_eval to merge all protein chains into one receptor chain 'R' and peptide chain into 'L'.",
    )
    parser.add_argument(
        "--remove_data_ids",
        type=str,
        default=None,
        help="Comma-separated data_id list to skip, e.g. 'pepbdb_1abc_A,pepbdb_2def_B'",
    )
    parser.add_argument("--n_cores", type=int, default=1, help="Number of parallel processes (-1: all cores minus 1)")
    args = parser.parse_args()

    if args.remove_data_ids is not None and args.remove_data_ids != "":
        remove_data_ids = [data_id.strip() for data_id in args.remove_data_ids.split(",") if data_id.strip()]
        print("remove_data_ids:", remove_data_ids)
    else:
        remove_data_ids = []

    mode_list = [mode.strip().lower() for mode in str(args.mode).split(",") if mode.strip()]
    if not mode_list:
        raise ValueError("--mode is empty. Expected one or more modes from ['gen', 'gt', 'best_cfd'].")
    valid_modes = {"gen", "gt", "best_cfd"}
    invalid_modes = [mode for mode in mode_list if mode not in valid_modes]
    if invalid_modes:
        raise ValueError(
            f"Unsupported mode(s): {invalid_modes}. Expected each mode in {sorted(valid_modes)}."
        )
    print("mode_list:", mode_list)

    gen_path = get_dir_from_prefix(args.result_root, args.exp_name)
    print("gen_path:", gen_path)
    for mode in mode_list:
        batch_merge_from_gen_info(
            gen_path=gen_path,
            protein_dir=args.gt_dir,
            mode=mode,
            use_utils_eval=args.use_utils_eval,
            n_cores=args.n_cores,
            remove_data_ids=remove_data_ids,
        )


if __name__ == "__main__":
    main()

