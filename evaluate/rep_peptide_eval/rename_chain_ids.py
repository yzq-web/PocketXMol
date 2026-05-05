import argparse
import os
from string import ascii_uppercase
from typing import List

from Bio.PDB import PDBIO, PDBParser


def get_chain_id_sequence(start_id: str, n: int) -> List[str]:
    """
    Build chain IDs starting from `start_id` in alphabetical order.
    Example: start_id='R', n=5 -> ['R', 'S', 'T', 'U', 'V']
    """
    if len(start_id) != 1:
        raise ValueError(f"start_id must be one character, got: {start_id}")
    if start_id not in ascii_uppercase:
        raise ValueError(f"start_id must be A-Z, got: {start_id}")
    if n <= 0:
        return []

    start_idx = ascii_uppercase.index(start_id)
    ordered = list(ascii_uppercase[start_idx:]) + list(ascii_uppercase[:start_idx])
    if n > len(ordered):
        raise ValueError(f"Too many chains ({n}); only supports up to {len(ordered)} unique IDs.")
    return ordered[:n]


def rename_pdb_chain_ids(input_pdb: str, output_pdb: str, chain_ids: List[str]) -> int:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", input_pdb)
    model = structure[0]
    chains = list(model.get_chains())
    if len(chains) == 0:
        raise ValueError(f"No chains found in: {input_pdb}")
    if len(chains) > len(chain_ids):
        raise ValueError(
            f"Provided {len(chain_ids)} chain IDs, but file has {len(chains)} chains: {input_pdb}"
        )

    # Detach all chains first to avoid duplicate-id conflict during renaming.
    for chain in chains:
        model.detach_child(chain.id)

    for chain, new_id in zip(chains, chain_ids):
        chain.id = new_id
        model.add(chain)

    os.makedirs(os.path.dirname(output_pdb), exist_ok=True)
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb)
    return len(chains)


def process_one_subdir(input_dir: str, subdir_name: str):
    base_dir = os.path.join(input_dir, "complex", subdir_name)
    peptide_path = os.path.join(base_dir, "peptide.pdb")
    receptor_path = os.path.join(base_dir, "receptor.pdb")
    output_dir = os.path.join(base_dir, "rename_chain")
    out_peptide = os.path.join(output_dir, "peptide.pdb")
    out_receptor = os.path.join(output_dir, "receptor.pdb")

    if not os.path.isfile(peptide_path):
        raise FileNotFoundError(f"Missing peptide.pdb: {peptide_path}")
    if not os.path.isfile(receptor_path):
        raise FileNotFoundError(f"Missing receptor.pdb: {receptor_path}")

    peptide_chain_ids = get_chain_id_sequence("P", 26)
    peptide_n = rename_pdb_chain_ids(peptide_path, out_peptide, peptide_chain_ids)

    receptor_chain_ids = get_chain_id_sequence("R", 26)
    receptor_n = rename_pdb_chain_ids(receptor_path, out_receptor, receptor_chain_ids)

    print(
        f"[Done] {subdir_name}: peptide chains={peptide_n} -> "
        f"{peptide_chain_ids[:peptide_n]}, receptor chains={receptor_n} -> {receptor_chain_ids[:receptor_n]}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Rename chain IDs in <input_dir>/complex/<subdir>/peptide.pdb and receptor.pdb."
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Root directory containing complex/")
    parser.add_argument(
        "--subdir",
        type=str,
        default=None,
        help="Optional single subdir under complex/. If not set, process all subdirs.",
    )
    args = parser.parse_args()

    complex_dir = os.path.join(args.input_dir, "complex")
    if not os.path.isdir(complex_dir):
        raise FileNotFoundError(f"complex dir not found: {complex_dir}")

    if args.subdir:
        subdirs = [args.subdir]
    else:
        subdirs = sorted([d for d in os.listdir(complex_dir) if os.path.isdir(os.path.join(complex_dir, d))])

    for subdir_name in subdirs:
        process_one_subdir(args.input_dir, subdir_name)

    print(f"[All Done] Processed {len(subdirs)} subdir(s).")


if __name__ == "__main__":
    main()
