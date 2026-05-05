#!/usr/bin/env python3
import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from process.utils_process import extract_pocket, make_dummy_mol_with_coordinate
from utils.parser import PDBProtein


def _parse_float(v: str) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_low_fnat_meta(meta_csv: Path, fnat_threshold: float) -> Dict[str, List[str]]:
    data_to_paths: Dict[str, List[str]] = {}
    if not meta_csv.is_file():
        return data_to_paths

    with meta_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data_id = (row.get("data_id") or "").strip()
            pocket_path = (row.get("pocket_path") or "").strip()
            fnat_value = _parse_float((row.get("fnat") or "").strip())
            if not data_id or not pocket_path or fnat_value is None:
                continue
            if fnat_value < fnat_threshold:
                data_to_paths.setdefault(data_id, []).append(pocket_path)
    return data_to_paths


def collect_complex_subdirs(input_dir: Path) -> List[Path]:
    complex_dir = input_dir / "complex"
    if not complex_dir.is_dir():
        return []
    return sorted([p for p in complex_dir.iterdir() if p.is_dir()])


def residue_id_set_from_pdb(pdb_path: Path) -> Set[str]:
    if not pdb_path.is_file():
        return set()
    pdb = PDBProtein(str(pdb_path))
    return {residue["chain_res_id"] for residue in pdb.residues}


def resolve_pocket_path(path_str: str, input_dir: Path, data_id: str) -> Optional[Path]:
    raw = Path(path_str)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append((input_dir / raw).resolve())
    candidates.append((input_dir / data_id / "pesto_pocket" / raw.name).resolve())
    candidates.append((input_dir / "complex" / data_id / "pesto_pocket" / raw.name).resolve())
    for c in candidates:
        if c.is_file():
            return c
    return None


def sample_surface_near_center(
    receptor_atom_pos: np.ndarray,
    receptor_center: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    atom_idx = int(rng.integers(0, len(receptor_atom_pos)))
    atom_pos = receptor_atom_pos[atom_idx]
    outward = atom_pos - receptor_center
    if np.linalg.norm(outward) < 1e-6:
        outward = rng.normal(size=3)
    noise = rng.normal(size=3)
    direction = 0.7 * outward + 0.3 * noise
    direction_norm = np.linalg.norm(direction)
    if direction_norm < 1e-8:
        direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        direction = direction / direction_norm
    offset = rng.uniform(1.0, 4.0)
    return atom_pos + direction * offset


def extract_random_non_overlapping_pocket(
    receptor: PDBProtein,
    forbidden_residue_ids: Set[str],
    radius: float,
    min_num_res: int,
    max_trials: int,
    rng: np.random.Generator,
) -> Tuple[Optional[List[dict]], Optional[np.ndarray], int]:
    receptor_atom_pos = np.asarray(receptor.pos, dtype=np.float64)
    receptor_center = receptor_atom_pos.mean(axis=0)

    for trial in range(1, max_trials + 1):
        center = sample_surface_near_center(receptor_atom_pos, receptor_center, rng)
        dummy_mol = make_dummy_mol_with_coordinate(center.tolist())
        selected = receptor.query_residues_ligand(
            dummy_mol,
            radius=radius,
            criterion="center_of_mass",
        )
        if len(selected) <= min_num_res:
            continue
        selected_ids = {residue["chain_res_id"] for residue in selected}
        if selected_ids & forbidden_residue_ids:
            continue
        return selected, center, trial
    return None, None, max_trials


def write_single_pocket_meta(
    csv_path: Path,
    data_id: str,
    pocket_idx: str,
    pocket_path: Path,
    num_res: int,
    pocket_center: np.ndarray,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    center_str = f"{pocket_center[0]:.3f},{pocket_center[1]:.3f},{pocket_center[2]:.3f}"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["data_id", "pocket_idx", "pocket_path", "num_res", "pocket_center"])
        writer.writerow([data_id, pocket_idx, str(pocket_path.resolve()), num_res, center_str])


def process_one_subdir(
    input_dir: Path,
    complex_subdir: Path,
    low_fnat_meta: Dict[str, List[str]],
    radius: float,
    max_trials: int,
    min_num_res: int,
    rng: np.random.Generator,
) -> Tuple[bool, str]:
    data_id = complex_subdir.name
    receptor_path = complex_subdir / "receptor.pdb"
    peptide_path = complex_subdir / "peptide.pdb"
    pocket10_path = complex_subdir / "pocket10.pdb"

    if not receptor_path.is_file():
        return False, f"[SKIP] {data_id}: missing receptor.pdb"
    if not peptide_path.is_file():
        return False, f"[SKIP] {data_id}: missing peptide.pdb"

    try:
        extract_pocket(
            protein_path=str(receptor_path),
            mol_path=str(peptide_path),
            radius=radius,
            save_path=str(pocket10_path),
            criterion="center_of_mass",
        )
    except Exception as exc:  # pylint: disable=broad-except
        return False, f"[FAIL] {data_id}: pocket10 extraction failed: {exc}"

    forbidden_ids = residue_id_set_from_pdb(pocket10_path)
    for pocket_path_str in low_fnat_meta.get(data_id, []):
        resolved = resolve_pocket_path(pocket_path_str, input_dir, data_id)
        if resolved is None:
            continue
        forbidden_ids |= residue_id_set_from_pdb(resolved)

    receptor = PDBProtein(str(receptor_path))
    selected, center, trial_used = extract_random_non_overlapping_pocket(
        receptor=receptor,
        forbidden_residue_ids=forbidden_ids,
        radius=radius,
        min_num_res=min_num_res,
        max_trials=max_trials,
        rng=rng,
    )
    if selected is None or center is None:
        return False, f"[FAIL] {data_id}: no valid random pocket found within {max_trials} trials"

    random_dir = input_dir / "complex" / data_id / "random_pocket"
    random_dir.mkdir(parents=True, exist_ok=True)
    random_pocket_path = random_dir / "random_pocket10.pdb"
    pocket_block = receptor.residues_to_pdb_block(selected, name="RANDOM_POCKET")
    with random_pocket_path.open("w", encoding="utf-8") as f:
        f.write(pocket_block)

    write_single_pocket_meta(
        csv_path=random_dir / "pocket_meta.csv",
        data_id=data_id,
        pocket_idx="random_pocket10",
        pocket_path=random_pocket_path,
        num_res=len(selected),
        pocket_center=center,
    )
    return True, f"[OK] {data_id}: random pocket found in {trial_used} trials -> {random_pocket_path}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "For each <input_dir>/complex/<subdir>, extract pocket10 from peptide/receptor, "
            "then sample a non-overlapping random pocket (radius=10) near receptor surface."
        )
    )
    parser.add_argument("input_dir", help="Dataset root directory")
    parser.add_argument("--radius", type=float, default=10.0, help="Pocket radius (default: 10)")
    parser.add_argument("--fnat-threshold", type=float, default=0.1, help="Filter threshold for available pockets (default: 0.1)")
    parser.add_argument("--max-trials", type=int, default=3000, help="Max random trials per subdir (default: 3000)")
    parser.add_argument("--min-num-res", type=int, default=5, help="Require random pocket residue number > min_num_res (default: 5)")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed (default: 2026)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")

    complex_subdirs = collect_complex_subdirs(input_dir)
    if not complex_subdirs:
        raise FileNotFoundError(f"No subdirectories found under: {input_dir / 'complex'}")

    meta_csv = input_dir / "meta" / "pocket_meta_all.csv"
    low_fnat_meta = load_low_fnat_meta(meta_csv, args.fnat_threshold)
    rng = np.random.default_rng(args.seed)

    total = len(complex_subdirs)
    success = 0
    failed = 0

    for complex_subdir in complex_subdirs:
        ok, message = process_one_subdir(
            input_dir=input_dir,
            complex_subdir=complex_subdir,
            low_fnat_meta=low_fnat_meta,
            radius=args.radius,
            max_trials=args.max_trials,
            min_num_res=args.min_num_res,
            rng=rng,
        )
        print(message)
        if ok:
            success += 1
        else:
            failed += 1

    print(f"Done. total={total}, success={success}, failed={failed}")


if __name__ == "__main__":
    main()
