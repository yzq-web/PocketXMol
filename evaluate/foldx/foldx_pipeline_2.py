import sys
sys.path.append(".")

import os
import shutil
import subprocess
import tempfile
from functools import partial
from multiprocessing import Pool

import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm

from evaluate.evaluate_mols import get_dir_from_prefix


def resolve_foldx_bin(foldx_bin=None):
    if foldx_bin:
        return foldx_bin
    env_bin = os.environ.get("FOLDX_BINARY") or os.environ.get("FOLDX_BIN")
    if env_bin:
        return env_bin
    for name in ("foldx", "FoldX", "foldx_linux64"):
        path = shutil.which(name)
        if path:
            return path
    raise FileNotFoundError(
        "FoldX executable not found. Set --foldx_bin or env FOLDX_BINARY/FOLDX_BIN."
    )


def prepare_rotabase(workdir, foldx_bin, rotabase_path=None):
    dst = os.path.join(workdir, "rotabase.txt")
    if os.path.exists(dst):
        return
    candidates = [
        rotabase_path,
        os.environ.get("FOLDX_ROTABASE"),
        os.path.join(os.path.dirname(foldx_bin), "rotabase.txt"),
        os.path.join(os.path.dirname(foldx_bin), "Rotabase.txt"),
    ]
    for src in candidates:
        if src and os.path.exists(src):
            shutil.copy2(src, dst)
            return


def run_foldx(workdir, foldx_bin, pdb_name, command_name, options=None):
    cmd = [
        foldx_bin,
        f"--command={command_name}",
        f"--pdb={pdb_name}",
    ]
    if options:
        cmd.extend(list(options))
    result = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"FoldX failed for {pdb_name} ({command_name})\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


def fetch_stability_score(path):
    u = pd.read_csv(path, sep="\t", header=None)
    return u.values[0][1]


def fetch_binding_affinity(path):
    with open(path, "r") as f:
        lines = f.readlines()
    line = lines[-1].split("\t")
    result = {
        "clash_rec": float(line[-5]),
        "clash_lig": float(line[-4]),
        "energy": float(line[-3]),
        "stable_rec": float(line[-2]),
        "stable_lig": float(line[-1]),
    }
    return result


def repair_one_file(filename, input_dir, output_dir, foldx_bin, rotabase_path=None):
    try:
        if not filename.endswith(".pdb"):
            return
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        if os.path.exists(output_path):
            return
        if not os.path.exists(input_path):
            print("File not found for repair:", input_path)
            return

        with tempfile.TemporaryDirectory(prefix="foldx_repair_") as workdir:
            prepare_rotabase(workdir, foldx_bin, rotabase_path=rotabase_path)
            shutil.copy2(input_path, os.path.join(workdir, filename))
            run_foldx(workdir, foldx_bin, filename, "RepairPDB")

            repaired_name = filename.replace(".pdb", "_Repair.pdb")
            repaired_path = os.path.join(workdir, repaired_name)
            if not os.path.exists(repaired_path):
                raise FileNotFoundError(f"FoldX repaired file not found: {repaired_path}")
            shutil.copy2(repaired_path, output_path)
    except Exception as e:
        print("ERROR repair:", filename, e)


def stability_one_file(filename, input_dir, output_dir, foldx_bin, rotabase_path=None):
    try:
        if not filename.endswith(".pdb"):
            return
        save_path = os.path.join(output_dir, filename.replace(".pdb", ".pkl"))
        if os.path.exists(save_path):
            return

        input_path = os.path.join(input_dir, filename)
        if not os.path.exists(input_path):
            print("File not found for stability:", input_path)
            return

        with tempfile.TemporaryDirectory(prefix="foldx_stability_") as workdir:
            prepare_rotabase(workdir, foldx_bin, rotabase_path=rotabase_path)
            shutil.copy2(input_path, os.path.join(workdir, filename))
            run_foldx(workdir, foldx_bin, filename, "Stability")

            stem = filename.replace(".pdb", "")
            fxout_path = os.path.join(workdir, f"{stem}_0_ST.fxout")
            if not os.path.exists(fxout_path):
                raise FileNotFoundError(f"FoldX stability file not found: {fxout_path}")
            score = fetch_stability_score(fxout_path)

        result = {"filename": stem, "stability": score}
        with open(save_path, "wb") as f:
            pickle.dump(result, f)
    except Exception as e:
        print("ERROR stability:", filename, e)


def energy_one_file(filename, input_dir, output_dir, chain_tuple, foldx_bin, rotabase_path=None):
    try:
        if not filename.endswith(".pdb"):
            return
        save_path = os.path.join(output_dir, filename.replace(".pdb", ".pkl"))
        if os.path.exists(save_path):
            return

        input_path = os.path.join(input_dir, filename)
        if not os.path.exists(input_path):
            print("File not found for energy:", input_path)
            return

        with tempfile.TemporaryDirectory(prefix="foldx_energy_") as workdir:
            prepare_rotabase(workdir, foldx_bin, rotabase_path=rotabase_path)
            shutil.copy2(input_path, os.path.join(workdir, filename))
            run_foldx(
                workdir,
                foldx_bin,
                filename,
                "AnalyseComplex",
                options=[f"--analyseComplexChains={chain_tuple}"],
            )

            stem = filename.replace(".pdb", "")
            fxout_path = os.path.join(workdir, f"Summary_{stem}_AC.fxout")
            if not os.path.exists(fxout_path):
                raise FileNotFoundError(f"FoldX AC summary not found: {fxout_path}")
            result = fetch_binding_affinity(fxout_path)
            result = {"filename": stem, **result}

        with open(save_path, "wb") as f:
            pickle.dump(result, f)
    except Exception as e:
        print("ERROR energy:", filename, e)


def calc_foldx_score(
    complex_dir,
    output_dir,
    num_workers,
    chain_tuple,
    repair=True,
    foldx_bin=None,
    rotabase_path=None,
):
    foldx_bin = resolve_foldx_bin(foldx_bin)
    all_files = [f for f in os.listdir(complex_dir) if f.endswith(".pdb")]
    all_files = np.random.permutation(all_files)

    if repair:
        repair_dir = os.path.join(output_dir, "repaired")
        os.makedirs(repair_dir, exist_ok=True)
        with Pool(num_workers) as p:
            list(
                tqdm(
                    p.imap_unordered(
                        partial(
                            repair_one_file,
                            input_dir=complex_dir,
                            output_dir=repair_dir,
                            foldx_bin=foldx_bin,
                            rotabase_path=rotabase_path,
                        ),
                        all_files,
                    ),
                    total=len(all_files),
                    desc="Foldx repairing...",
                )
            )
    else:
        repair_dir = complex_dir

    stability_dir = os.path.join(output_dir, "stability")
    os.makedirs(stability_dir, exist_ok=True)
    with Pool(num_workers) as p:
        list(
            tqdm(
                p.imap_unordered(
                    partial(
                        stability_one_file,
                        input_dir=repair_dir,
                        output_dir=stability_dir,
                        foldx_bin=foldx_bin,
                        rotabase_path=rotabase_path,
                    ),
                    all_files,
                ),
                total=len(all_files),
                desc="Calculating foldx stability...",
            )
        )

    energy_dir = os.path.join(output_dir, "energy")
    os.makedirs(energy_dir, exist_ok=True)
    with Pool(num_workers) as p:
        list(
            tqdm(
                p.imap_unordered(
                    partial(
                        energy_one_file,
                        input_dir=repair_dir,
                        output_dir=energy_dir,
                        chain_tuple=chain_tuple,
                        foldx_bin=foldx_bin,
                        rotabase_path=rotabase_path,
                    ),
                    all_files,
                ),
                total=len(all_files),
                desc="Calculating energy...",
            )
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="base_pxm")
    parser.add_argument("--result_root", type=str, default="outputs_test/dock_pepbdb")

    parser.add_argument("--complex_dir", type=str, default="")
    parser.add_argument("--foldx_dir", type=str, default="")
    parser.add_argument("--num_workers", type=int, default=126)
    parser.add_argument("--chain_tuple", type=str, default="R,L")
    parser.add_argument("--foldx_bin", type=str, default="")
    parser.add_argument("--rotabase_path", type=str, default="")
    parser.add_argument("--disable_repair", action="store_true")
    args = parser.parse_args()

    # if args.exp_name and args.result_root:
    #     gen_path = get_dir_from_prefix(args.result_root, args.exp_name)
    #     print("gen_path:", gen_path)
    #     complex_dir = os.path.join(gen_path, "complex")
    #     foldx_dir = os.path.join(gen_path, "foldx")
    # else:
    #     complex_dir = args.complex_dir
    #     foldx_dir = args.foldx_dir

    complex_dir = args.complex_dir
    foldx_dir = args.foldx_dir
    
    calc_foldx_score(
        complex_dir,
        foldx_dir,
        num_workers=args.num_workers,
        chain_tuple=args.chain_tuple,
        repair=not args.disable_repair,
        foldx_bin=args.foldx_bin if args.foldx_bin else None,
        rotabase_path=args.rotabase_path if args.rotabase_path else None,
    )

    df_foldx = []
    energy_dir = os.path.join(foldx_dir, "energy")
    for file in os.listdir(energy_dir):
        if file.endswith(".pkl"):
            path = os.path.join(energy_dir, file)
            with open(path, "rb") as f:
                score = pickle.load(f)
            df_foldx.append(score)
    df_foldx = pd.DataFrame(df_foldx)
    df_foldx.to_csv(os.path.join(foldx_dir, "foldx.csv"), index=False)

    print("Done.")
