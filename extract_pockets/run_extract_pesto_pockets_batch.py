#!/usr/bin/env python3
import argparse
import csv
import re
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from Bio.PDB import PDBIO, PDBParser
import shutil

from extract_pockets import extract_pockets

PATH_DOCKQ = '/data1/home/yangziqing/software/DockQ'
assert Path(PATH_DOCKQ).is_dir(), f"DockQ not found at {PATH_DOCKQ}"

def collect_subdirs(input_dir):
    return sorted([p for p in input_dir.iterdir() if p.is_dir()])


def count_residues_in_pdb(pdb_path):
    residues = set()
    with pdb_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            chain_id = line[21].strip() or "_"
            resseq = line[22:26].strip()
            icode = line[26].strip()
            resname = line[17:20].strip()
            residues.add((chain_id, resseq, icode, resname))
    return len(residues)


def compute_pocket_center_from_pdb(pdb_path):
    x_sum = 0.0
    y_sum = 0.0
    z_sum = 0.0
    atom_count = 0
    with pdb_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            try:
                x_sum += float(line[30:38].strip())
                y_sum += float(line[38:46].strip())
                z_sum += float(line[46:54].strip())
                # print(line, line[30:38].strip(), line[38:46].strip(), line[46:54].strip())
            except ValueError:
                continue
            atom_count += 1
    if atom_count == 0:
        return ""
    x_center = x_sum / atom_count
    y_center = y_sum / atom_count
    z_center = z_sum / atom_count
    return f"{x_center:.3f},{y_center:.3f},{z_center:.3f}"


def _pocket_sort_key(pocket_path):
    m = re.search(r"_poc(\d+)\.pdb$", pocket_path.name)
    if m is None:
        return 10**9
    return int(m.group(1))


def _collect_chain_ids(pdb_path):

    parser = PDBParser(QUIET=True)
    model = parser.get_structure("s", str(pdb_path))[0]
    return {chain.id for chain in model.get_chains()}


def _choose_ligand_chain_id(receptor_path, pocket_path):
    used = _collect_chain_ids(receptor_path) | _collect_chain_ids(pocket_path)
    for cid in ["L", "Z", "Y", "X", "W", "V", "U", "T", "S", "R", "Q", "P"]:
        if cid not in used:
            return cid
    for cid in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789":
        if cid not in used:
            return cid
    raise ValueError("No free chain id available for ligand.")


def _build_complex(receptor_path, peptide_path, output_path, ligand_chain_id):
    from Bio.PDB import PDBIO, PDBParser

    parser = PDBParser(QUIET=True)
    receptor_model = parser.get_structure("rec", str(receptor_path))[0]
    peptide_model = parser.get_structure("pep", str(peptide_path))[0]
    peptide_chains = list(peptide_model.get_chains())
    if not peptide_chains:
        raise ValueError(f"No chain found in peptide PDB: {peptide_path}")

    ligand_chain = deepcopy(peptide_chains[0])
    ligand_chain.id = ligand_chain_id
    receptor_model.add(ligand_chain)

    io = PDBIO()
    io.set_structure(receptor_model)
    io.save(str(output_path))


# def _renumber_model_residues_from_one(model):
#     for chain in model:
#         residues = list(chain)
#         if not residues:
#             continue

#         # Keep the first residue number when possible, then advance by one for
#         # every following residue in chain order. This makes cases like 36,36A
#         # become 36,37 while removing insertion codes.
#         start_idx = residues[0].id[1]
#         new_ids = [(" ", start_idx + i, " ") for i in range(len(residues))]

#         # Avoid sibling-id collision in Biopython by using a temporary id range
#         # before assigning the final continuous ids.
#         temp_offset = 1000000
#         for i, residue in enumerate(residues):
#             residue.id = (" ", temp_offset + i, " ")
#         for residue, new_id in zip(residues, new_ids):
#             residue.id = new_id


def _renumber_receptor_pdb_by_text(input_path, output_path):
    chain_state = {}
    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.startswith(("ATOM", "HETATM")):
                fout.write(line)
                continue

            chain_id = line[21]
            hetflag = "W" if line[17:20] == "HOH" else " "
            resseq_text = line[22:26]
            try:
                resseq = int(resseq_text)
            except ValueError:
                fout.write(line)
                continue
            icode = line[26]
            residue_key = (chain_id, hetflag, resseq, icode)

            if chain_id not in chain_state:
                chain_state[chain_id] = {"last_key": None, "next_idx": None, "mapping": {}}
            state = chain_state[chain_id]

            if residue_key not in state["mapping"]:
                if state["last_key"] is None:
                    state["next_idx"] = 1 # start from 1
                else:
                    state["next_idx"] += 1
                state["mapping"][residue_key] = state["next_idx"]
                state["last_key"] = residue_key

            new_resseq = state["mapping"][residue_key]
            new_line = f"{line[:22]}{new_resseq:4d} {line[27:]}"
            fout.write(new_line)


def _build_complex_with_renumbered_receptor(receptor_path, peptide_path, output_path, ligand_chain_id):

    parser = PDBParser(QUIET=True)
    with tempfile.TemporaryDirectory(prefix="renum_rec_") as tmp_dir:
        renumbered_receptor = Path(tmp_dir) / "receptor_renumbered.pdb"
        _renumber_receptor_pdb_by_text(receptor_path, renumbered_receptor)
        receptor_model = parser.get_structure("rec", str(renumbered_receptor))[0]

        peptide_model = parser.get_structure("pep", str(peptide_path))[0]
        peptide_chains = list(peptide_model.get_chains())
        if not peptide_chains:
            raise ValueError(f"No chain found in peptide PDB: {peptide_path}")

        ligand_chain = deepcopy(peptide_chains[0])
        ligand_chain.id = ligand_chain_id
        receptor_model.add(ligand_chain)

        io = PDBIO()
        io.set_structure(receptor_model)
        io.save(str(output_path))


def _parse_dockq_output(stdout):

    results = stdout.split('\n')[-6:-1]
    if 'DockQ' not in results[-1]:
        raise ValueError('DockQ failed: ' + stdout)
    
    fnat = results[0].split()[1]
    irmsd = results[2].split()[1]
    lrmsd = results[3].split()[1]
    dockq = results[4].split()[1]
    print('DockQ output:', 'fnat:', fnat, 'irmsd:', irmsd, 'lrmsd:', lrmsd, 'dockq:', dockq, '\n')
    # print(stdout)
    
    return {
        'fnat': float(fnat),
        'irmsd': float(irmsd),
        'lrmsd': float(lrmsd),
        'dockq': float(dockq),
    }


def _calc_dockq_for_pocket(pocket_path, receptor_path, peptide_path):
    dockq_script = Path(PATH_DOCKQ) / "DockQ.py"
    ligand_chain_id = _choose_ligand_chain_id(receptor_path, pocket_path)

    with tempfile.TemporaryDirectory(prefix="dockq_pocket_") as tmp_dir:
        tmp_dir = Path(tmp_dir)
        gt_complex = tmp_dir / "complex_gt.pdb"
        pred_complex = tmp_dir / "complex_pred.pdb"
        _build_complex_with_renumbered_receptor(receptor_path, peptide_path, gt_complex, ligand_chain_id)
        _build_complex(pocket_path, peptide_path, pred_complex, ligand_chain_id)

        cmd = [
            sys.executable,
            str(dockq_script),
            str(pred_complex),
            str(gt_complex),
            "-model_chain1",
            ligand_chain_id,
            "-native_chain1",
            ligand_chain_id,
            "-no_needle",
        ]
        output = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        print(f"[DockQ CMD] {' '.join(cmd)}")
        if output.stdout.strip():
            print("[DockQ STDOUT]")
            print(output.stdout)
        if output.stderr.strip():
            print("[DockQ STDERR]")
            print(output.stderr)
        if output.returncode != 0:
            msg = output.stderr.strip() or output.stdout.strip() or f"DockQ errored with code {output.returncode}"
            raise ValueError(msg)
        return _parse_dockq_output(output.stdout)


def _format_error_code(exc):
    msg = str(exc).strip().replace("\n", " | ")
    m = re.search(r"(AssertionError:\s*[^|]+)", msg)
    if m:
        return m.group(1).strip()
    return msg[:300]


def write_pocket_meta_csv(
    pocket_output_dir,
    data_id,
    input_stem,
    receptor_path,
    peptide_path,
):
    pocket_output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = pocket_output_dir / "pocket_meta.csv"
    pocket_paths = sorted(
        pocket_output_dir.glob(f"{input_stem}_poc*.pdb"),
        key=_pocket_sort_key,
    )

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "data_id",
                "pocket_idx",
                "pocket_path",
                "num_res",
                "pocket_center",
                "fnat",
                "irmsd",
                "lrmsd",
                "dockq",
                "error_code",
            ]
        )
        for pocket_path in pocket_paths:
            pocket_abs = pocket_path.resolve()
            pocket_idx = pocket_path.stem
            num_res = count_residues_in_pdb(pocket_path)
            pocket_center = compute_pocket_center_from_pdb(pocket_path)
            fnat = ""
            irmsd = ""
            lrmsd = ""
            dockq = ""
            error_code = ""
            if receptor_path.is_file() and peptide_path.is_file():
                try:
                    dockq_dict = _calc_dockq_for_pocket(
                        pocket_abs,
                        receptor_path,
                        peptide_path,
                    )
                    fnat = dockq_dict.get("fnat", "")
                    irmsd = dockq_dict.get("irmsd", "")
                    lrmsd = dockq_dict.get("lrmsd", "")
                    dockq = dockq_dict.get("dockq", "")
                except Exception as exc:  # pylint: disable=broad-except
                    error_code = _format_error_code(exc)
                    print(f"[WARN] DockQ failed for {pocket_abs}: {exc}")
            writer.writerow(
                [
                    data_id,
                    pocket_idx,
                    str(pocket_abs),
                    num_res,
                    pocket_center,
                    fnat,
                    irmsd,
                    lrmsd,
                    dockq,
                    error_code,
                ]
            )


def merge_all_pocket_meta(input_dir, subdirs, pocket_output_dirname):
    merged_csv = input_dir.parent / "meta" / "pocket_meta_all.csv"
    header = [
        "data_id",
        "pocket_idx",
        "pocket_path",
        "num_res",
        "pocket_center",
        "fnat",
        "irmsd",
        "lrmsd",
        "dockq",
        "error_code",
    ]
    all_rows = []

    for subdir in subdirs:
        csv_path = subdir / pocket_output_dirname / "pocket_meta.csv"
        if not csv_path.is_file():
            continue
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_rows.append([row.get(col, "") for col in header])

    with merged_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(all_rows)

    print(f"[OK] merged pocket metadata -> {merged_csv}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Batch process PeSTo outputs: for each subdirectory under input_dir, "
            "run pocket extraction on pesto_output/receptor_i0.pdb and write to "
            "a sibling directory pesto_pocket."
        )
    )
    parser.add_argument("input_dir", help="Root folder containing multiple job subdirectories")
    parser.add_argument(
        "--pesto-output-dirname",
        default="pesto_output",
        help="Directory name for PeSTo outputs (default: pesto_output)",
    )
    parser.add_argument(
        "--input-pdb-name",
        default="receptor_i0.pdb",
        help="Input PDB name inside pesto_output (default: receptor_i0.pdb)",
    )
    parser.add_argument(
        "--pocket-output-dirname",
        default="pesto_pocket",
        help="Pocket output directory name (default: pesto_pocket)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="B-factor threshold for residue selection (default: 0.5)",
    )
    parser.add_argument(
        "--distance-cutoff",
        type=float,
        default=8.0,
        help="Distance cutoff in Angstrom for pocket clustering (default: 8.0)",
    )
    parser.add_argument(
        "--min-pocket-size",
        type=int,
        default=1,
        help="Minimum residue count required to keep a pocket (default: 1)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")

    subdirs = collect_subdirs(input_dir)
    if not subdirs:
        print(f"No subdirectories found under: {input_dir}")
        return

    total = 0
    success = 0
    skipped = 0
    failed = 0
    pockets_total = 0

    for subdir in subdirs:
        total += 1
        pesto_output_dir = subdir / args.pesto_output_dirname
        input_pdb = pesto_output_dir / args.input_pdb_name

        if not input_pdb.is_file():
            skipped += 1
            print(f"[SKIP] Missing file: {input_pdb}")
            continue

        pocket_output_dir = subdir / args.pocket_output_dirname
        # Clean up the pocket output directory
        if pocket_output_dir.exists() and pocket_output_dir.is_dir():
            for item in pocket_output_dir.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
   
        receptor_path = subdir / "receptor.pdb"
        peptide_path = subdir / "peptide.pdb"
        try:
            n_pockets = extract_pockets(
                input_pdb=str(input_pdb),
                threshold=args.threshold,
                distance_cutoff=args.distance_cutoff,
                output_dir=str(pocket_output_dir),
                min_pocket_size=args.min_pocket_size,
            )
            write_pocket_meta_csv(
                pocket_output_dir=pocket_output_dir,
                data_id=subdir.name,
                input_stem=input_pdb.stem,
                receptor_path=receptor_path,
                peptide_path=peptide_path,
            )
            success += 1
            pockets_total += n_pockets
            print(f"[OK] {input_pdb} -> {pocket_output_dir} (pockets={n_pockets})")
        except Exception as exc:  # pylint: disable=broad-except
            failed += 1
            print(f"[FAIL] {input_pdb}: {exc}")

    print(
        "Done. "
        f"total={total}, success={success}, skipped={skipped}, failed={failed}, "
        f"pockets_total={pockets_total}"
    )
    merge_all_pocket_meta(input_dir, subdirs, args.pocket_output_dirname)


if __name__ == "__main__":
    main()
