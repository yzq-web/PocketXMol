import argparse
from pathlib import Path

from rdkit import Chem


def convert_one_pdb_to_sdf(input_pdb: Path, output_sdf: Path) -> bool:
    """Convert one PDB file to SDF format."""
    mol = Chem.MolFromPDBFile(str(input_pdb), removeHs=False, sanitize=True)
    if mol is None or mol.GetNumAtoms() == 0:
        print(f"[FAILED] Could not parse valid molecule: {input_pdb}")
        return False

    writer = Chem.SDWriter(str(output_sdf))
    try:
        writer.write(mol)
    finally:
        writer.close()
    return True


def batch_convert(
    peptides_dir: Path = Path("./data/pepbdb/files/peptides"),
    mols_dir: Path = Path("./data/pepbdb/files/mols"),
) -> None:
    """Batch convert *_pep.pdb files in peptides_dir to *_mol.sdf in mols_dir."""
    if not peptides_dir.exists():
        raise FileNotFoundError(f"Peptides directory does not exist: {peptides_dir}")

    mols_dir.mkdir(parents=True, exist_ok=True)

    pdb_files = sorted(peptides_dir.glob("*_pep.pdb"))
    if not pdb_files:
        print(f"No *_pep.pdb files found in: {peptides_dir}")
        return

    success = 0
    failed = 0
    for pdb_file in pdb_files:
        base_name = pdb_file.name.replace("_pep.pdb", "")
        output_sdf = mols_dir / f"{base_name}_mol.sdf"
        ok = convert_one_pdb_to_sdf(pdb_file, output_sdf)
        if ok:
            success += 1
            print(f"[OK] {pdb_file.name} -> {output_sdf.name}")
        else:
            failed += 1

    print(f"Done. Success: {success}, Failed: {failed}, Output: {mols_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert PDB files in data/pepbdb/files/peptides to SDF files in data/pepbdb/files/mols."
    )
    parser.add_argument(
        "--peptides_dir",
        type=Path,
        default=Path("./data/pepbdb/files/peptides"),
        help="Input directory containing *_pep.pdb files.",
    )
    parser.add_argument(
        "--mols_dir",
        type=Path,
        default=Path("./data/pepbdb/files/mols"),
        help="Output directory for *_mol.sdf files.",
    )
    args = parser.parse_args()

    batch_convert(args.peptides_dir, args.mols_dir)


if __name__ == "__main__":
    main()
