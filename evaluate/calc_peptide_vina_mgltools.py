import argparse
import os
import re
import shutil
import subprocess
import uuid
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append(".")
from evaluate.evaluate_mols import get_dir_from_prefix


def _strip_rare_atoms_from_pdb(src_path, dst_path):
    """
    VinaDock cannot handle rare atoms. like: 'As'
    Remove rare atoms from a PDB file.
    Returns the number of removed atoms.
    """
    rare_atoms = ['AS']
    removed_count = 0
    kept_lines = []
    with open(src_path, "r") as f:
        for line in f:
            if line.startswith(("ATOM  ", "HETATM")):
                element = line[76:78].strip() if len(line) >= 78 else ""
                atom_name = line[12:16].strip() if len(line) >= 16 else ""
                is_rare_atom = (element in rare_atoms) or any(atom_name.startswith(rare_atom) for rare_atom in rare_atoms)
                if is_rare_atom:
                    removed_count += 1
                    continue
            kept_lines.append(line)

    with open(dst_path, "w") as f:
        f.writelines(kept_lines)

    return removed_count


def _keep_first_altloc_conformer_pdb(src_path, dst_path):
    """
    PDB column 17 is the alternate location indicator. When multiple non-blank
    altLocs appear (e.g. A and B), keep only the alphabetically first code
    (typically 'A'); ATOM/HETATM lines with blank altLoc are always kept.

    Returns:
        (n_removed, applied): applied is True iff multi-conformer filtering was done
        and dst_path was written.
    """
    with open(src_path, "r") as f:
        lines = f.readlines()
    alts = set()
    for line in lines:
        if line.startswith(("ATOM  ", "HETATM")) and len(line) > 16:
            c = line[16]
            if c not in " \t":
                alts.add(c)
    if len(alts) < 2:
        return 0, False
    keep_alt = min(alts) # first conformer
    removed = 0
    kept_lines = []
    for line in lines:
        if line.startswith(("ATOM  ", "HETATM")) and len(line) > 16:
            c = line[16]
            if c not in " \t" and c != keep_alt:
                removed += 1
                continue
        kept_lines.append(line)
    with open(dst_path, "w") as f:
        f.writelines(kept_lines)
    return removed, True


def _process_one_vina(task):
    index = task["index"]
    data_id = task["data_id"]
    filename = task["filename"]
    tag = task["tag"]
    gen_dir = task["gen_dir"]
    protein_dir = task["protein_dir"]
    tmp_root = task["tmp_root"]
    remove_data_ids = task["remove_data_ids"]
    pythonsh_path = task["pythonsh_path"]
    prepare_ligand_path = task["prepare_ligand_path"]
    prepare_receptor_path = task["prepare_receptor_path"]
    vina_bin_path = task["vina_bin_path"]
    vina_cpu = task["vina_cpu"]

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
        info_parts = []
        error_parts = []
        rec_path_clean = os.path.join(sample_tmp, f"{data_id}_{index}_receptor_clean.pdb")
        removed_rare_atom_count = _strip_rare_atoms_from_pdb(rec_path, rec_path_clean)
        if removed_rare_atom_count > 0:
            rec_path = rec_path_clean
            msg = f"Removed {removed_rare_atom_count} rare atoms from receptor"
            print(msg)
            info_parts.append(msg)

        rec_path_altloc = os.path.join(sample_tmp, f"{data_id}_{index}_receptor_altloc.pdb")
        removed_altloc, altloc_applied = _keep_first_altloc_conformer_pdb(rec_path, rec_path_altloc)
        if altloc_applied:
            rec_path = rec_path_altloc
            msg = f"Kept first altLoc conformer only (removed {removed_altloc} atoms)"
            print(msg)
            info_parts.append(msg)

        def _format_cmd_error(name, cmd, returncode=None, stdout="", stderr="", exc=None):
            cmd_str = " ".join(cmd)
            if exc is not None:
                return f"[{name}] exception: {exc}; cmd: {cmd_str}"
            return (
                f"[{name}] returncode={returncode}; cmd: {cmd_str}; "
                f"stdout: {(stdout or '').strip()}; stderr: {(stderr or '').strip()}"
            )

        lig_pdbqt = os.path.join(sample_tmp, f"{data_id}_{index}_ligand.pdbqt")
        rec_pdbqt = os.path.join(sample_tmp, f"{data_id}_{index}_receptor.pdbqt")

        ligand_cmd = [
            pythonsh_path,
            prepare_ligand_path,
            "-l",
            pep_pred_path,
            "-o",
            lig_pdbqt,
            "-F" # 如果Peptide断裂, 则取最大片段
        ]
        try:
            ligand_result = subprocess.run(
                ligand_cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            if ligand_result.returncode != 0:
                error_parts.append(
                    _format_cmd_error(
                        "ligand_cmd",
                        ligand_cmd,
                        returncode=ligand_result.returncode,
                        stdout=ligand_result.stdout,
                        stderr=ligand_result.stderr,
                    )
                )
        except Exception as e:
            error_parts.append(_format_cmd_error("ligand_cmd", ligand_cmd, exc=e))

        receptor_cmd = [
            pythonsh_path,
            prepare_receptor_path,
            "-r",
            rec_path,
            "-o",
            rec_pdbqt,
        ]
        try:
            receptor_result = subprocess.run(
                receptor_cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            if receptor_result.returncode != 0:
                error_parts.append(
                    _format_cmd_error(
                        "receptor_cmd",
                        receptor_cmd,
                        returncode=receptor_result.returncode,
                        stdout=receptor_result.stdout,
                        stderr=receptor_result.stderr,
                    )
                )
        except Exception as e:
            error_parts.append(_format_cmd_error("receptor_cmd", receptor_cmd, exc=e))

        if not os.path.exists(lig_pdbqt):
            error_parts.append(f"[ligand_cmd] Missing output file: {lig_pdbqt}")
        if not os.path.exists(rec_pdbqt):
            error_parts.append(f"[receptor_cmd] Missing output file: {rec_pdbqt}")
        if not (os.path.exists(lig_pdbqt) and os.path.exists(rec_pdbqt)):
            all_msgs = info_parts + error_parts
            return index, np.nan, "; ".join(all_msgs) if all_msgs else np.nan

        vina_cmd = [
            vina_bin_path,
            "--receptor",
            rec_pdbqt,
            "--ligand",
            lig_pdbqt,
            "--cpu",
            str(vina_cpu),
            "--score_only",
        ]
        result = subprocess.run(vina_cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            error_parts.append(
                _format_cmd_error(
                    "vina_cmd",
                    vina_cmd,
                    returncode=result.returncode,
                    stdout=result.stdout,
                    stderr=result.stderr,
                )
            )
        print("vina output:", result.stdout)

        affinity_match = re.search(r"Affinity:\s+([-\d.]+)", result.stdout)
        if affinity_match is None:
            combined_out = (result.stdout or "") + "\n" + (result.stderr or "")
            error_parts.append(f"Cannot parse affinity from vina output: {combined_out}")
            all_msgs = info_parts + error_parts
            return index, np.nan, "; ".join(all_msgs) if all_msgs else np.nan
        all_msgs = info_parts + error_parts
        return index, float(affinity_match.group(1)), "; ".join(all_msgs) if all_msgs else np.nan
    except Exception as e:
        return index, np.nan, str(e)
    finally:
        shutil.rmtree(sample_tmp, ignore_errors=True)


def evaluate_vina_df(
    df_gen,
    gen_dir,
    protein_dir,
    tmp_root,
    check_repeats=0,
    remove_data_ids=None,
    pythonsh_path="",
    prepare_ligand_path="",
    prepare_receptor_path="",
    vina_bin_path="",
    vina_cpu=1,
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
    print(f"Running vina scoring with {n_cores} process(es)")

    tasks = []
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
                "remove_data_ids": set(remove_data_ids),
                "pythonsh_path": pythonsh_path,
                "prepare_ligand_path": prepare_ligand_path,
                "prepare_receptor_path": prepare_receptor_path,
                "vina_bin_path": vina_bin_path,
                "vina_cpu": vina_cpu,
            }
        )

    iterator = None
    pool = None
    if n_cores == 1:
        iterator = map(_process_one_vina, tasks)
    else:
        pool = Pool(processes=n_cores)
        iterator = pool.imap_unordered(_process_one_vina, tasks)

    try:
        for index, vina_score, error_code in tqdm(iterator, total=len(tasks), desc="calc vina score"):
            df_gen.loc[index, "vina_score"] = vina_score
            df_gen.loc[index, "error_code"] = error_code
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    return df_gen


def evaluate_vina_df_gt(
    df_gen,
    gen_dir,
    protein_dir,
    tmp_root,
    remove_data_ids=None,
    pythonsh_path="",
    prepare_ligand_path="",
    prepare_receptor_path="",
    vina_bin_path="",
    vina_cpu=1,
    n_cores=1,
):
    # keep one sample per data_id for GT scoring
    df_gt = df_gen.drop_duplicates(subset=["data_id"], keep="first").copy()

    def _to_gt_filename(filename):
        basename, ext = os.path.splitext(filename)
        if ext == "":
            ext = ".pdb"
        return f"{basename}_gt{ext}"

    df_gt["filename"] = df_gt["filename"].map(_to_gt_filename)
    df_gt["tag"] = np.nan

    return evaluate_vina_df(
        df_gen=df_gt,
        gen_dir=gen_dir,
        protein_dir=protein_dir,
        tmp_root=tmp_root,
        check_repeats=0,
        remove_data_ids=remove_data_ids,
        pythonsh_path=pythonsh_path,
        prepare_ligand_path=prepare_ligand_path,
        prepare_receptor_path=prepare_receptor_path,
        vina_bin_path=vina_bin_path,
        vina_cpu=vina_cpu,
        n_cores=n_cores,
    )


def main():
    parser = argparse.ArgumentParser(description="Compute Vina score for peptide PDB predictions.")
    parser.add_argument("--exp_name", type=str, default="msel_base_fixendresbb")
    parser.add_argument("--result_root", type=str, default="./outputs_paper/dock_pepbdb")
    # Keep arg name aligned with calc_peptide_ca_rmsd.py:
    # here gt_dir is used as protein(receptor) pdb directory.
    parser.add_argument("--gt_dir", type=str, default="data/pepbdb/files/proteins")
    parser.add_argument("--check_repeats", type=int, default=0)
    parser.add_argument("--remove_data_ids", type=str, default=None)
    parser.add_argument("--pythonsh_path", type=str, default="/home/yangziqing/software/mgltools_x86_64Linux2_1.5.6/bin/pythonsh")
    parser.add_argument("--prepare_ligand_path", type=str, default="/home/yangziqing/software/mgltools_x86_64Linux2_1.5.6/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py")
    parser.add_argument("--prepare_receptor_path", type=str, default="/home/yangziqing/software/mgltools_x86_64Linux2_1.5.6/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py")
    parser.add_argument("--vina_bin_path", type=str, default="/home/yangziqing/software/autodock_vina_1_1_2_linux_x86/bin/vina")
    parser.add_argument("--vina_cpu", type=int, default=1)
    parser.add_argument("--n_cores", type=int, default=1, help="Number of parallel processes (-1: all cores minus 1)")
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
    vina_tmp_dir = os.path.join(gen_path, "vina_tmp")  # temporary vina files
    os.makedirs(vina_tmp_dir, exist_ok=True)

    df_gen = pd.read_csv(os.path.join(gen_path, "gen_info.csv"))

    vina_csv_path = os.path.join(gen_path, "vina_score.csv")
    if not os.path.exists(vina_csv_path):
        df_vina = evaluate_vina_df(
            df_gen=df_gen,
            gen_dir=pdb_path,
            protein_dir=protein_dir,
            tmp_root=vina_tmp_dir,
            check_repeats=args.check_repeats,
            remove_data_ids=remove_data_ids,
            pythonsh_path=args.pythonsh_path,
            prepare_ligand_path=args.prepare_ligand_path,
            prepare_receptor_path=args.prepare_receptor_path,
            vina_bin_path=args.vina_bin_path,
            vina_cpu=args.vina_cpu,
            n_cores=args.n_cores,
        )
        df_vina.to_csv(vina_csv_path, index=False)
        print("Saved to", vina_csv_path)
    else:
        print("vina_score.csv already exists")

    vina_gt_csv_path = os.path.join(gen_path, "vina_score_gt.csv")
    if not os.path.exists(vina_gt_csv_path):
        df_vina_gt = evaluate_vina_df_gt(
            df_gen=df_gen,
            gen_dir=pdb_path,
            protein_dir=protein_dir,
            tmp_root=vina_tmp_dir,
            remove_data_ids=remove_data_ids,
            pythonsh_path=args.pythonsh_path,
            prepare_ligand_path=args.prepare_ligand_path,
            prepare_receptor_path=args.prepare_receptor_path,
            vina_bin_path=args.vina_bin_path,
            vina_cpu=args.vina_cpu,
            n_cores=args.n_cores,
        )
        use_col = ['data_id', 'db', 'filename', 'vina_score', 'error_code']
        df_vina_gt = df_vina_gt[use_col]
        df_vina_gt.to_csv(vina_gt_csv_path, index=False)
        print("Saved to", vina_gt_csv_path)
    else:
        print("vina_score_gt.csv already exists")

    # remove vina_tmp_dir
    shutil.rmtree(vina_tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()