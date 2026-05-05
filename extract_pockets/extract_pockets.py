#!/usr/bin/env python3
import argparse
import math
from collections import defaultdict
from pathlib import Path


AA3_TO_AA1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "SEC": "U", "PYL": "O", "ASX": "B", "GLX": "Z", "UNK": "X",
}


def _parse_atom_line(line):
    record = line[0:6].strip()
    if record not in {"ATOM", "HETATM"}:
        return None

    atom_name = line[12:16].strip()
    resname = line[17:20].strip()
    chain_id = line[21].strip() or "_"
    resseq = int(line[22:26].strip())
    icode = line[26].strip()
    x = float(line[30:38].strip())
    y = float(line[38:46].strip())
    z = float(line[46:54].strip())
    bfactor = float(line[60:66].strip()) if line[60:66].strip() else 0.0

    return {
        "line": line.rstrip("\n"),
        "record": record,
        "atom_name": atom_name,
        "resname": resname,
        "chain_id": chain_id,
        "resseq": resseq,
        "icode": icode,
        "xyz": (x, y, z),
        "bfactor": bfactor,
    }


def _residue_sort_key(res_key):
    chain_id, resseq, icode, _ = res_key
    return (chain_id, resseq, icode)


def _residue_to_aa1(resname):
    return AA3_TO_AA1.get(resname.upper(), "X")


def _distance(a, b):
    return math.sqrt(
        (a[0] - b[0]) ** 2 +
        (a[1] - b[1]) ** 2 +
        (a[2] - b[2]) ** 2
    )


def _build_residue_rep_coords(residue_atoms):
    rep_coords = {}
    for res_key, atoms in residue_atoms.items():
        ca_atoms = [a for a in atoms if a["atom_name"] == "CA" and a["record"] == "ATOM"]
        if ca_atoms:
            rep_coords[res_key] = ca_atoms[0]["xyz"]
            continue
        xs = [a["xyz"][0] for a in atoms]
        ys = [a["xyz"][1] for a in atoms]
        zs = [a["xyz"][2] for a in atoms]
        rep_coords[res_key] = (sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs))
    return rep_coords


def _connected_components(selected_residues, rep_coords, distance_cutoff):
    nodes = list(selected_residues)
    n = len(nodes)
    adjacency = [[] for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            if _distance(rep_coords[nodes[i]], rep_coords[nodes[j]]) <= distance_cutoff:
                adjacency[i].append(j)
                adjacency[j].append(i)

    visited = [False] * n
    components = []
    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        comp = []
        while stack:
            cur = stack.pop()
            comp.append(nodes[cur])
            for nxt in adjacency[cur]:
                if not visited[nxt]:
                    visited[nxt] = True
                    stack.append(nxt)
        components.append(sorted(comp, key=_residue_sort_key))

    # Larger pocket first for readability.
    components.sort(key=lambda c: len(c), reverse=True)
    return components


def _write_component_pdb(out_path, atoms, component_residues):
    selected = set(component_residues)
    lines = [a["line"] for a in atoms if (a["chain_id"], a["resseq"], a["icode"], a["resname"]) in selected]
    with out_path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")
        f.write("END\n")


def _write_component_fasta(out_path, component_residues, prefix):
    per_chain = defaultdict(list)
    for chain_id, resseq, icode, resname in sorted(component_residues, key=_residue_sort_key):
        per_chain[chain_id].append((resseq, icode, _residue_to_aa1(resname)))

    with out_path.open("w", encoding="utf-8") as f:
        for chain_id in sorted(per_chain):
            seq = "".join(x[2] for x in per_chain[chain_id])
            f.write(f">{prefix}|chain={chain_id}\n")
            for i in range(0, len(seq), 80):
                f.write(seq[i:i + 80] + "\n")


def extract_pockets(
    input_pdb,
    threshold=0.5,
    distance_cutoff=8.0,
    output_dir=None,
    min_pocket_size=1,
):
    input_path = Path(input_pdb)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input PDB not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    atoms = []
    residue_atoms = defaultdict(list)
    for line in raw_lines:
        parsed = _parse_atom_line(line)
        if parsed is None:
            continue
        atoms.append(parsed)
        res_key = (parsed["chain_id"], parsed["resseq"], parsed["icode"], parsed["resname"])
        residue_atoms[res_key].append(parsed)

    selected_residues = []
    for res_key, atom_list in residue_atoms.items():
        max_b = max(a["bfactor"] for a in atom_list)
        if max_b > threshold:
            selected_residues.append(res_key)

    if not selected_residues:
        return 0

    rep_coords = _build_residue_rep_coords(residue_atoms)
    components = _connected_components(selected_residues, rep_coords, distance_cutoff)
    components = [c for c in components if len(c) >= min_pocket_size]
    if not components:
        return 0

    stem = input_path.stem
    out_dir = Path(output_dir) if output_dir is not None else input_path.parent
    if out_dir.exists():
        for file in out_dir.iterdir():
            if file.is_file():
                file.unlink()
            elif file.is_dir():
                # Only remove empty dirs, for safety; skip non-empty subdirs
                try:
                    file.rmdir()
                except OSError:
                    pass
        print(f"Info: Output directory '{out_dir}' exists. All files inside were deleted.")
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, component in enumerate(components, start=1):
        out_prefix = f"{stem}_poc{idx}"
        pdb_path = out_dir / f"{out_prefix}.pdb"
        fasta_path = out_dir / f"{out_prefix}.fasta"
        _write_component_pdb(pdb_path, atoms, component)
        _write_component_fasta(fasta_path, component, out_prefix)

    return len(components)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract residues with B-factor above threshold, cluster them into "
            "pockets/interfaces, and export each pocket to PDB + FASTA."
        )
    )
    parser.add_argument("input_pdb", help="Input PDB filepath, e.g. <input_name>.pdb")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Residue selection threshold on max atom B-factor (default: 0.5)",
    )
    parser.add_argument(
        "--distance-cutoff",
        type=float,
        default=8.0,
        help="Distance cutoff (Angstrom) to connect residues into one pocket (default: 8.0)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory (default: same directory as input PDB)",
    )
    parser.add_argument(
        "--min-pocket-size",
        type=int,
        default=1,
        help="Minimum residue count required to keep a pocket (default: 1)",
    )
    args = parser.parse_args()

    n_pockets = extract_pockets(
        args.input_pdb,
        threshold=args.threshold,
        distance_cutoff=args.distance_cutoff,
        output_dir=args.output_dir,
        min_pocket_size=args.min_pocket_size,
    )
    print(f"Detected pockets/interfaces: {n_pockets}")


if __name__ == "__main__":
    main()
