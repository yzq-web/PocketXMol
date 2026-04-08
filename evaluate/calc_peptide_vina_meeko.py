import argparse
import os
import shutil
import uuid
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm import tqdm

import sys
sys.path.append(".")
from evaluate.evaluate_mols import get_dir_from_prefix
from utils.docking_vina import VinaDockingTask


def _process_one_vina(task):
    index = task["index"]
    data_id = task["data_id"]
    filename = task["filename"]
    tag = task["tag"]
    gen_dir = task["gen_dir"]
    protein_dir = task["protein_dir"]
    tmp_root = task["tmp_root"]
    remove_data_ids = task["remove_data_ids"]
    mode = task["mode"]
    exhaustiveness = task["exhaustiveness"]

    # only successful peptides are evaluated
    if pd.notna(tag):
        return index, np.nan, np.nan

    if data_id in remove_data_ids:
        return index, np.nan, np.nan

    pep_pred_path = os.path.join(gen_dir, filename)
    rec_path = os.path.join(protein_dir, f"{data_id}_pro.pdb")
    if not os.path.exists(pep_pred_path):
        return index, np.nan, f"Missing pred peptide: {pep_pred_path}"
    if not os.path.exists(rec_path):
        return index, np.nan, f"Missing receptor: {rec_path}"

    sample_tmp = os.path.join(tmp_root, f"{data_id}_{index}_{uuid.uuid4().hex[:8]}")
    os.makedirs(sample_tmp, exist_ok=True)
    try:
        ligand_mol = Chem.MolFromPDBFile(pep_pred_path, sanitize=False, removeHs=False)
        if ligand_mol is None or ligand_mol.GetNumAtoms() == 0:
            return index, np.nan, f"Failed to parse peptide pdb as rdkit mol: {pep_pred_path}"

        # copy receptor to tmp so generated .pqr/.pdbqt stay under gen_path/vina_tmp
        rec_tmp = os.path.join(sample_tmp, os.path.basename(rec_path))
        shutil.copy2(rec_path, rec_tmp)

        vina_task = VinaDockingTask(rec_tmp, ligand_mol, tmp_dir=sample_tmp)
        score = vina_task.run(mode=mode, exhaustiveness=exhaustiveness, save_pose=False)[0]["affinity"]
        return index, float(score), np.nan
    except Exception as e:
        print(f"Error: {e}")
        return index, np.nan, str(e)
    finally:
        # shutil.rmtree(sample_tmp, ignore_errors=True)
        pass


def evaluate_vina_df(
    df_gen,
    gen_dir,
    protein_dir,
    tmp_root,
    check_repeats=0,
    remove_data_ids=None,
    mode="score_only",
    exhaustiveness=8,
    n_cores=1,
):
    if remove_data_ids is None:
        remove_data_ids = []

    data_id_list = df_gen["data_id"].unique()
    print("Find %d generated mols with %d unique data_id" % (len(df_gen), len(data_id_list)))
    if check_repeats > 0:
        assert len(df_gen) / len(data_id_list) == check_repeats, (
            f"Repeat {check_repeats} not match: {len(df_gen)}:{len(data_id_list)}"
        )

    df_gen = df_gen.copy()
    df_gen["vina_score"] = np.nan
    df_gen["error_code"] = np.nan
    df_gen.reset_index(inplace=True, drop=True)

    if n_cores == -1:
        n_cores = max(1, cpu_count() - 1)
    n_cores = max(1, int(n_cores))
    print(f"Running meeko-vina scoring with {n_cores} process(es)")

    tasks = []
    remove_set = set(remove_data_ids)
    for index, line in df_gen.iterrows():
        tasks.append(
            {
                "index": index,
                "data_id": line["data_id"],
                "filename": line["filename"],
                "tag": line["tag"] if "tag" in df_gen.columns else np.nan,
                "gen_dir": gen_dir,
                "protein_dir": protein_dir,
                "tmp_root": tmp_root,
                "remove_data_ids": remove_set,
                "mode": mode,
                "exhaustiveness": exhaustiveness,
            }
        )

    if n_cores == 1:
        iterator = map(_process_one_vina, tasks)
        pool = None
    else:
        pool = Pool(processes=n_cores)
        iterator = pool.imap_unordered(_process_one_vina, tasks)

    try:
        for index, vina_score, error_code in tqdm(iterator, total=len(tasks), desc="calc vina score (meeko)"):
            df_gen.loc[index, "vina_score"] = vina_score
            df_gen.loc[index, "error_code"] = error_code
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    return df_gen


def main():
    parser = argparse.ArgumentParser(description="Compute Vina score for peptide PDB predictions (Meeko pipeline).")
    parser.add_argument("--exp_name", type=str, default="msel_base_fixendresbb")
    parser.add_argument("--result_root", type=str, default="./outputs_paper/dock_pepbdb")
    # Keep arg name aligned with calc_peptide_ca_rmsd.py:
    # here gt_dir is used as protein(receptor) pdb directory.
    parser.add_argument("--gt_dir", type=str, default="data/pepbdb/files/proteins")
    parser.add_argument("--check_repeats", type=int, default=0)
    parser.add_argument("--remove_data_ids", type=str, default=None)
    parser.add_argument("--mode", type=str, default="score_only", choices=["score_only", "minimize", "dock"])
    parser.add_argument("--exhaustiveness", type=int, default=8)
    parser.add_argument("--n_cores", type=int, default=1, help="Number of parallel processes (-1: all cores minus 1)")
    parser.add_argument("--output_csv", type=str, default="vina_score_meeko.csv")
    args = parser.parse_args()

    result_root = args.result_root
    exp_name = args.exp_name
    protein_dir = args.gt_dir

    if args.remove_data_ids is not None and args.remove_data_ids != "":
        remove_data_ids = [data_id.strip() for data_id in args.remove_data_ids.split(",")]
        print("remove_data_ids:", remove_data_ids)
    else:
        remove_data_ids = []

    gen_path = get_dir_from_prefix(result_root, exp_name)
    print("gen_path:", gen_path)

    pdb_path = os.path.join(gen_path, "SDF")  # predicted peptides.pdb
    vina_tmp_dir = os.path.join(gen_path, "vina_tmp")
    os.makedirs(vina_tmp_dir, exist_ok=True)

    df_gen = pd.read_csv(os.path.join(gen_path, "gen_info.csv"))
    df_vina = evaluate_vina_df(
        df_gen=df_gen,
        gen_dir=pdb_path,
        protein_dir=protein_dir,
        tmp_root=vina_tmp_dir,
        check_repeats=args.check_repeats,
        remove_data_ids=remove_data_ids,
        mode=args.mode,
        exhaustiveness=args.exhaustiveness,
        n_cores=args.n_cores,
    )

    output_csv = args.output_csv
    if not os.path.isabs(output_csv):
        output_csv = os.path.join(gen_path, output_csv)
    df_vina.to_csv(output_csv, index=False)
    print("Saved to", output_csv)


if __name__ == "__main__":
    main()
