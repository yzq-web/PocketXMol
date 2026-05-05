import argparse
import os
import subprocess
import tempfile
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional

import pandas as pd
import yaml
from Bio.PDB import PDBIO, PDBParser
from tqdm import tqdm

import sys
# Make import robust to current working directory changes.
# Project root is two levels above this file: PocketXMol/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from evaluate.rep_peptide_eval.rename_chain_ids import rename_pdb_chain_ids
from evaluate.rep_peptide_eval.renumber_residue_ids import renumber_pdb_by_text


DOCKQ_COLUMNS = ["dockq", "irmsd", "lrmsd"]
PATH_DOCKQ = "/data1/home/yangziqing/software/DockQ"
assert os.path.exists(PATH_DOCKQ), f"PATH_DOCKQ not found: {PATH_DOCKQ}"


def resolve_pred_pdb(row: pd.Series, pred_dir: str) -> Optional[str]:
    """
    Resolve predicted pdb path. Priority:
    1) data_id -> <data_id>.pdb
    2) filename column
    """
    candidates = []

    data_id = row.get("data_id", None)
    if pd.notna(data_id):
        data_id = str(data_id).strip()
        if data_id:
            candidates.append(data_id if data_id.endswith(".pdb") else f"{data_id}.pdb")

    filename = row.get("filename", None)
    if pd.notna(filename):
        filename = os.path.basename(str(filename).strip())
        if filename:
            candidates.append(filename)

    for name in candidates:
        path = os.path.join(pred_dir, name)
        if os.path.exists(path):
            return path
    return None


def get_single_chain_id(pdb_path: str) -> str:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("lig", pdb_path)[0]
    chain_ids = [chain.id for chain in structure.get_chains()]
    if len(chain_ids) != 1:
        raise ValueError(f"Ligand must contain exactly one chain: {pdb_path}, got {chain_ids}")
    return chain_ids[0]


def get_chain_ids(pdb_path: str) -> List[str]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("s", pdb_path)[0]
    return [chain.id for chain in structure.get_chains()]


def choose_free_ligand_chain_id(receptor_path: str) -> str:
    used = set(get_chain_ids(receptor_path))
    # Prefer common ligand chain names first.
    for cid in ["L", "Z", "Y", "X", "W", "V", "U", "T", "S", "Q", "P"]:
        if cid not in used:
            return cid
    for cid in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789":
        if cid not in used:
            return cid
    raise ValueError(f"No free chain ID available for receptor: {receptor_path}")


def ligand_resseq_starts_from_one(pdb_path: str) -> bool:
    """
    Return True when the first ATOM/HETATM residue index is 1.
    """
    with open(pdb_path, "r", encoding="utf-8") as fin:
        for line in fin:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            try:
                return int(line[22:26]) == 1
            except ValueError:
                return False
    return False


def get_dockq(pred_ligand_path: str, gt_ligand_path: str, receptor_path: str) -> Dict[str, float]:
    """
    Local DockQ wrapper migrated from evaluate/utils_eval.py.
    Before running DockQ, detect chain IDs from pred and gt ligands and pass
    them as model/native chain inputs.
    """
    if not os.path.isdir(PATH_DOCKQ):
        raise FileNotFoundError(f"PATH_DOCKQ not found: {PATH_DOCKQ}")

    # Detect original chain ids for sanity check/debug.
    pred_chain_id = get_single_chain_id(pred_ligand_path)
    gt_chain_id = get_single_chain_id(gt_ligand_path)
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp_base = tmp.name
    tmp.close()
    tmp_pred_lig_renum = tmp_base + "_pred_lig_renum.pdb"
    tmp_pred_lig = tmp_base + "_pred_lig.pdb"
    tmp_pred = tmp_base + "_pred.pdb"
    tmp_gt = tmp_base + "_gt.pdb"

    parser = PDBParser(QUIET=True)
    try:
        # If pred ligand residue numbering does not start from 1, renumber first.
        if not ligand_resseq_starts_from_one(pred_ligand_path):
            renumber_pdb_by_text(pred_ligand_path, tmp_pred_lig_renum)
            pred_ligand_for_dockq = tmp_pred_lig_renum
        else:
            pred_ligand_for_dockq = pred_ligand_path

        # If chain IDs differ, convert pred_ligand chain id to gt_chain_id first.
        if pred_chain_id != gt_chain_id:
            rename_pdb_chain_ids(pred_ligand_for_dockq, tmp_pred_lig, [gt_chain_id])
            pred_ligand_for_dockq = tmp_pred_lig

        for lig_path, tmp_path in [(pred_ligand_for_dockq, tmp_pred), (gt_ligand_path, tmp_gt)]:
            rec = parser.get_structure("rec", receptor_path)[0]
            lig = parser.get_structure("lig", lig_path)[0]
            lig_chain = list(lig.get_chains())[0]
            rec.add(lig_chain)
            io = PDBIO()
            io.set_structure(rec)
            io.save(tmp_path)

        cmd = ["python", f"{PATH_DOCKQ}/DockQ.py", tmp_pred, tmp_gt]
        cmd += ["-model_chain1", gt_chain_id, "-native_chain1", gt_chain_id, "-no_needle"]
        output = subprocess.run(cmd, capture_output=True, text=True)
        print(output.stdout)
        print(output.stderr)
        if output.returncode != 0:
            raise ValueError(
                "DockQ errored: "
                + output.stderr
                + f" | pred_chain={pred_chain_id}, gt_chain={gt_chain_id}"
            )

        results = output.stdout.split("\n")[-4:-1]
        if len(results) < 3 or "DockQ" not in results[-1]:
            raise ValueError("DockQ failed: " + output.stdout)

        irmsd = results[0].split()[1]
        lrmsd = results[1].split()[1]
        dockq = results[2].split()[1]
        print('irmsd:', irmsd, 'lrmsd:', lrmsd, 'dockq:', dockq)
        return {"irmsd": float(irmsd), "lrmsd": float(lrmsd), "dockq": float(dockq)}
    finally:
        for p in [tmp_pred_lig_renum, tmp_pred_lig, tmp_pred, tmp_gt, tmp_base]:
            if os.path.exists(p):
                os.remove(p)


def data_id_to_sample_name(data_id: str) -> str:
    """
    Convert config data_id to output sample_name.
    Examples:
    - dockpep_1g7q_a_p -> dock_pep_1g7q_a_p
    - dockshuffle_xxx -> dock_shuffle_xxx
    - dockrandom_xxx -> dock_random_xxx
    - dockss_xxx -> dock_ss_xxx
    """
    normalize_prefix_map = {
        "dock_pep": ["dock_pep", "dockpep"],
        "dock_shuffle": ["dock_shuffle", "dockshuffle"],
        "dock_random": ["dock_random", "dockrandom"],
        "dock_ss": ["dock_ss", "dockss"],
    }

    for canonical_prefix, aliases in normalize_prefix_map.items():
        for alias in aliases:
            if data_id.startswith(alias + "_"):
                suffix = data_id[len(alias) + 1 :]
                return f"{canonical_prefix}_{suffix}"
            if data_id.startswith(alias):
                suffix = data_id[len(alias) :]
                if suffix.startswith("_"):
                    suffix = suffix[1:]
                return f"{canonical_prefix}_{suffix}" if suffix else canonical_prefix

    return data_id


def resolve_project_path(path_value: str) -> str:
    if os.path.isabs(path_value):
        return path_value
    return os.path.abspath(os.path.join(PROJECT_ROOT, path_value))


def prefer_renamed_chain_path(gt_receptor: str, gt_ligand: str):
    """
    Prefer renamed-chain files if present:
    - <dir_of_receptor>/rename_chain/receptor.pdb
    - <dir_of_ligand>/rename_chain/peptide.pdb
    """
    receptor_dir = os.path.dirname(gt_receptor)
    ligand_dir = os.path.dirname(gt_ligand)

    receptor_renamed = os.path.join(receptor_dir, "rename_chain", "receptor.pdb")
    ligand_renamed = os.path.join(ligand_dir, "rename_chain", "peptide.pdb")

    if os.path.isfile(receptor_renamed):
        gt_receptor = receptor_renamed
    if os.path.isfile(ligand_renamed):
        gt_ligand = ligand_renamed
    return gt_receptor, gt_ligand


def load_config_tasks(config_dir: str, output_dock_dir: str, task_name: str, sample_name: Optional[str]) -> List[Dict[str, str]]:
    config_files = sorted(
        [
            os.path.join(config_dir, x)
            for x in os.listdir(config_dir)
            if x.endswith(".yml") or x.endswith(".yaml")
        ]
    )
    if not config_files:
        raise FileNotFoundError(f"No yaml config found in: {config_dir}")

    tasks = []
    for cfg_path in config_files:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        data_cfg = cfg.get("data", {})
        pocmol_cfg = data_cfg.get("pocmol_args", {})
        data_id = str(pocmol_cfg.get("data_id", "")).strip()
        if not data_id:
            raise ValueError(f"Missing data.pocmol_args.data_id in config: {cfg_path}")

        this_sample = data_id_to_sample_name(data_id)
        if sample_name and this_sample != sample_name:
            continue

        gt_receptor = str(data_cfg.get("protein_path", "")).strip()
        if not gt_receptor:
            raise ValueError(f"Missing data.protein_path in config: {cfg_path}")

        pocket_args = data_cfg.get("pocket_args", {}) or {}
        if task_name in ["gt_pocket", "pesto_pocket", "random_pocket"]:
            gt_ligand = str(data_cfg.get("input_ligand", "")).strip()
            gt_ligand_src = "data.input_ligand"
        elif task_name in ["shuffle_pocket", "ss_pocket"]:
            gt_ligand = str(pocket_args.get("ref_ligand_path", "")).strip()
            gt_ligand_src = "data.pocket_args.ref_ligand_path"
        else:
            raise ValueError(f"Unsupported task_name for gt_ligand resolution: {task_name}")

        if not gt_ligand:
            raise ValueError(f"Missing gt_ligand source ({gt_ligand_src}) in config: {cfg_path}")

        gt_receptor = resolve_project_path(gt_receptor)
        gt_ligand = resolve_project_path(gt_ligand)
        gt_receptor, gt_ligand = prefer_renamed_chain_path(gt_receptor, gt_ligand)

        sample_dir = os.path.join(output_dock_dir, task_name, "output", this_sample)
        tasks.append(
            {
                "sample_name": this_sample,
                "data_id": data_id,
                "config_path": cfg_path,
                "gt_receptor": gt_receptor,
                "gt_ligand": gt_ligand,
                "sample_dir": sample_dir,
                "pred_dir": os.path.join(sample_dir, f"{this_sample}_SDF"),
                "gen_info_path": os.path.join(sample_dir, "gen_info.csv"),
                "result_path": os.path.join(sample_dir, "dockq_result.csv"),
            }
        )
    return tasks


def evaluate_one_sample(task: Dict[str, str]) -> Dict[str, int]:
    pred_dir = task["pred_dir"]
    gt_ligand = task["gt_ligand"]
    gt_receptor = task["gt_receptor"]
    gen_info_path = task["gen_info_path"]
    result_path = task["result_path"]

    if not os.path.isdir(pred_dir):
        raise FileNotFoundError(f"Pred dir not found: {pred_dir}")
    if not os.path.exists(gt_ligand):
        raise FileNotFoundError(f"GT ligand not found: {gt_ligand}")
    if not os.path.exists(gt_receptor):
        raise FileNotFoundError(f"GT receptor not found: {gt_receptor}")

    if os.path.exists(gen_info_path):
        df = pd.read_csv(gen_info_path)
    else:
        pdb_files = sorted([x for x in os.listdir(pred_dir) if x.endswith(".pdb")])
        df = pd.DataFrame({"filename": pdb_files})
        df["data_id"] = task["data_id"]

    if df.empty:
        df.to_csv(gen_info_path, index=False)
        df.to_csv(result_path, index=False)
        return {"processed": 0, "failed": 0, "skipped": 0}

    for col in DOCKQ_COLUMNS:
        if col not in df.columns:
            df[col] = float("nan")
    if "dockq_error" not in df.columns:
        df["dockq_error"] = ""

    processed = 0
    failed = 0
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=task["sample_name"]):
        pred_pdb = resolve_pred_pdb(row, pred_dir)
        if pred_pdb is None:
            df.loc[idx, "dockq_error"] = "pred_pdb_not_found"
            failed += 1
            continue
        try:
            metrics = get_dockq(pred_pdb, gt_ligand, gt_receptor)
            for k in DOCKQ_COLUMNS:
                df.loc[idx, k] = metrics.get(k, float("nan"))
            df.loc[idx, "dockq_error"] = ""
            processed += 1
        except Exception as exc:
            df.loc[idx, "dockq_error"] = str(exc).replace("\n", " ")[:500]
            failed += 1

    # Keep compatibility: write back gen_info if it exists.
    df.to_csv(gen_info_path, index=False)
    result_cols = [c for c in ["data_id", "filename", "dockq", "irmsd", "lrmsd", "dockq_error"] if c in df.columns]
    df[result_cols].to_csv(result_path, index=False)
    return {"processed": processed, "failed": failed, "skipped": 0}


def run_one_sample_task(task: Dict[str, str]):
    sample_name = task["sample_name"]
    if not os.path.isdir(task["sample_dir"]):
        return {
            "sample_name": sample_name,
            "processed": 0,
            "failed": 0,
            "skipped": 1,
            "message": f"Sample output dir not found: {task['sample_dir']}",
        }

    try:
        stats = evaluate_one_sample(task)
        stats["sample_name"] = sample_name
        stats["message"] = ""
        return stats
    except Exception as exc:
        return {
            "sample_name": sample_name,
            "processed": 0,
            "failed": 1,
            "skipped": 0,
            "message": str(exc).replace("\n", " ")[:500],
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate DockQ for rep_peptide docking outputs using config files.")
    parser.add_argument(
        "--config_dir",
        type=str,
        required=True,
        help="Directory containing YAML configs (e.g. data/rep_peptide_2/configs/gt_pocket).",
    )
    parser.add_argument(
        "--output_dock_dir",
        type=str,
        required=True,
        help="Dock output root (e.g. data/rep_peptide_2_dock_output).",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task name under output_dock_dir (e.g. gt_pocket).",
    )
    parser.add_argument(
        "--sample_name",
        type=str,
        default=None,
        help="Optional single sample_name to evaluate (e.g. dock_pep_1g7q_a_p).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of worker processes for sample-level parallelism. 0 means auto.",
    )
    args = parser.parse_args()

    tasks = load_config_tasks(
        config_dir=args.config_dir,
        output_dock_dir=args.output_dock_dir,
        task_name=args.task,
        sample_name=args.sample_name,
    )

    total_processed = 0
    total_failed = 0
    total_skipped = 0
    all_results = []

    if len(tasks) == 0:
        worker_count = 0
    elif args.num_workers and args.num_workers > 0:
        worker_count = min(args.num_workers, len(tasks))
    else:
        worker_count = min(cpu_count(), len(tasks))

    print(f"[Info] Using {worker_count} worker process(es)")

    if worker_count <= 1:
        task_iter = (run_one_sample_task(task) for task in tasks)
    else:
        pool = Pool(processes=worker_count)
        task_iter = pool.imap_unordered(run_one_sample_task, tasks, chunksize=1)

    for stats in tqdm(task_iter, total=len(tasks), desc="samples"):
        sample_name = stats["sample_name"]
        if stats.get("message"):
            if stats["skipped"] > 0:
                print(f"[Skip] {sample_name}: {stats['message']}")
            elif stats["failed"] > 0:
                print(f"[Fail] {sample_name}: {stats['message']}")

        total_processed += stats["processed"]
        total_failed += stats["failed"]
        total_skipped += stats["skipped"]
        print(
            f"[Done] {sample_name}: processed={stats['processed']} "
            f"failed={stats['failed']} skipped={stats['skipped']}"
        )

        result_path = next((t["result_path"] for t in tasks if t["sample_name"] == sample_name), None)
        if result_path and os.path.exists(result_path):
            sample_df = pd.read_csv(result_path)
            if not sample_df.empty:
                sample_df.insert(0, "sample_name", sample_name)
                all_results.append(sample_df)

    if worker_count > 1:
        pool.close()
        pool.join()

    all_result_path = os.path.join(args.output_dock_dir, args.task, "dockq_result_all.csv")
    if all_results:
        pd.concat(all_results, axis=0, ignore_index=True).to_csv(all_result_path, index=False)
        print(f"[Save] all results -> {all_result_path}")
    else:
        pd.DataFrame().to_csv(all_result_path, index=False)
        print(f"[Save] empty all results -> {all_result_path}")

    print(
        f"[Done] processed={total_processed}, failed={total_failed}, skipped={total_skipped}, "
        f"samples={len(tasks)}"
    )


if __name__ == "__main__":
    main()
