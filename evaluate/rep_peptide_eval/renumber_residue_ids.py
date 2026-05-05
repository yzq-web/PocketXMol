import argparse
from pathlib import Path


def _renumber_pdb_by_text(input_path: Path, output_path: Path):
    """
    Renumber residue IDs in each chain from 1, preserving chain order and
    residue encounter order in the original text.
    """
    chain_state = {}

    def _map_residue_index(line: str):
        chain_id = line[21]
        hetflag = "W" if line[17:20] == "HOH" else " "
        resseq_text = line[22:26]
        try:
            resseq = int(resseq_text)
        except ValueError:
            return None
        icode = line[26]
        residue_key = (chain_id, hetflag, resseq, icode)

        if chain_id not in chain_state:
            chain_state[chain_id] = {"last_key": None, "next_idx": None, "mapping": {}}
        state = chain_state[chain_id]

        if residue_key not in state["mapping"]:
            if state["last_key"] is None:
                state["next_idx"] = 1
            else:
                state["next_idx"] += 1
            state["mapping"][residue_key] = state["next_idx"]
            state["last_key"] = residue_key

        return state["mapping"][residue_key]

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if line.startswith(("ATOM", "HETATM")):
                new_resseq = _map_residue_index(line)
                if new_resseq is None:
                    fout.write(line)
                    continue
                new_line = f"{line[:22]}{new_resseq:4d} {line[27:]}"
                fout.write(new_line)
                continue

            if line.startswith("TER"):
                # Keep TER residue index in sync with preceding renumbering.
                new_resseq = _map_residue_index(line)
                if new_resseq is None:
                    fout.write(line)
                    continue
                new_line = f"{line[:22]}{new_resseq:4d} {line[27:]}"
                fout.write(new_line)
                continue

            else:
                fout.write(line)
                continue


def renumber_pdb_by_text(input_path, output_path):
    """
    Public helper for reuse in other scripts.
    Accepts either str or Path.
    """
    _renumber_pdb_by_text(Path(input_path), Path(output_path))


def _renumber_file_inplace(pdb_path: Path):
    if not pdb_path.is_file():
        raise FileNotFoundError(f"Missing PDB: {pdb_path}")
    tmp_path = pdb_path.with_suffix(pdb_path.suffix + ".tmp")
    _renumber_pdb_by_text(pdb_path, tmp_path)
    tmp_path.replace(pdb_path)


def process_one_subdir(input_dir: Path, subdir_name: str):
    rename_dir = input_dir / "complex" / subdir_name / "rename_chain"
    pep_path = rename_dir / "peptide.pdb"
    rec_path = rename_dir / "receptor.pdb"

    _renumber_file_inplace(pep_path)
    _renumber_file_inplace(rec_path)
    print(f"[Done] {subdir_name}: renumbered {pep_path.name} and {rec_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Renumber residue IDs for peptide.pdb and receptor.pdb under "
            "<input_dir>/complex/<subdir>/rename_chain, in-place."
        )
    )
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument(
        "--subdir",
        type=str,
        default=None,
        help="Optional: process only one subdir name.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    complex_dir = input_dir / "complex"
    if not complex_dir.is_dir():
        raise NotADirectoryError(f"complex dir not found: {complex_dir}")

    if args.subdir:
        subdirs = [args.subdir]
    else:
        subdirs = sorted([p.name for p in complex_dir.iterdir() if p.is_dir()])

    for subdir_name in subdirs:
        process_one_subdir(input_dir, subdir_name)

    print(f"[All Done] Processed {len(subdirs)} subdir(s).")


if __name__ == "__main__":
    main()
