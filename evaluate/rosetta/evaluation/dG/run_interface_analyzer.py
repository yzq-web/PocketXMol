#!/usr/bin/python
# -*- coding:utf-8 -*-
import argparse
import os
import pickle
from tqdm import tqdm
from multiprocessing import Pool
import traceback


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run fastrelax and interface analyzer for all PDB files in a directory."
    )
    parser.add_argument("--input_dir", required=True, type=str, help="Directory containing input .pdb files")
    parser.add_argument("--output_dir", required=True, type=str, help="Directory to save *_score.pkl results")
    parser.add_argument("--pep_chain", default="B", type=str, help="Peptide chain ID (default: B)")
    parser.add_argument("--pro_chain", default="A", type=str, help="Protein/receptor chain ID (default: A)")
    parser.add_argument(
        "--rfdiff_config",
        action="store_true",
        help="Use RFdiffusion relax configuration in pyrosetta_fastrelax",
    )
    parser.add_argument(
        "--num_workers",
        default=max(1, (os.cpu_count() or 1) // 2),
        type=int,
        help="Number of worker processes (default: half of CPU cores). Use 1 for serial mode.",
    )
    return parser.parse_args()


def _load_energy_funcs():
    try:
        from .energy import pyrosetta_fastrelax, pyrosetta_interface_energy
    except ImportError:
        from energy import pyrosetta_fastrelax, pyrosetta_interface_energy
    return pyrosetta_fastrelax, pyrosetta_interface_energy


def run_single(pdb_path, output_dir, pep_chain, pro_chain, rfdiff_config):
    pyrosetta_fastrelax, pyrosetta_interface_energy = _load_energy_funcs()
    base_name = os.path.splitext(os.path.basename(pdb_path))[0]
    dirname = os.path.dirname(pdb_path)
    relaxed_pdb = os.path.join(dirname, f"{base_name}_fastrelax.pdb")
    score_path = os.path.join(output_dir, f"{base_name}_score.pkl")

    pyrosetta_fastrelax(
        pdb_path,
        relaxed_pdb,
        pep_chain,
        rfdiff_config=rfdiff_config,
    )
    score = pyrosetta_interface_energy(relaxed_pdb, pro_chain, pep_chain, return_dict=True)

    with open(score_path, "wb") as f:
        pickle.dump(score, f)

    return score_path, score


def _run_single_worker(task):
    try:
        score_path, score = run_single(*task)
        return task[0], score_path, score, None
    except Exception:
        return task[0], None, None, traceback.format_exc()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    pdb_files = sorted(
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.endswith(".pdb") and 'fastrelax' not in f
    )
    if not pdb_files:
        raise ValueError(f"No .pdb files found in input_dir: {args.input_dir}")

    tasks = [
        (pdb_path, args.output_dir, args.pep_chain, args.pro_chain, args.rfdiff_config)
        for pdb_path in pdb_files
    ]

    if args.num_workers <= 1:
        for task in tasks:
            pdb_path, score_path, score, err = _run_single_worker(task)
            if err is None:
                print(f"[OK] {os.path.basename(pdb_path)} -> {score_path} (score={score})")
            else:
                print(f"[FAILED] {os.path.basename(pdb_path)}")
                print(err)
        return

    with Pool(args.num_workers) as pool:
        for pdb_path, score_path, score, err in tqdm(
            pool.imap_unordered(_run_single_worker, tasks),
            total=len(tasks),
            desc="Running interface analyzer",
        ):
            if err is None:
                print(f"[OK] {os.path.basename(pdb_path)} -> {score_path} (score={score})")
            else:
                print(f"[FAILED] {os.path.basename(pdb_path)}")
                print(err)


if __name__ == "__main__":
    main()
