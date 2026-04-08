#!/usr/bin/env python3
"""
Convert PepMerge directory layout to a bpep-like layout.

Input (PepMerge):
  <pepmerge_root>/<sample_id>/peptide.pdb
  <pepmerge_root>/<sample_id>/receptor.pdb
  <pepmerge_root>/<sample_id>/receptor_merge.pdb

Output:
  <output_root>/peptides/pepmerge_<sample_id>_pep.pdb
  <output_root>/proteins/pepmerge_<sample_id>_pro.pdb
  <output_root>/merged_proteins/pepmerge_<sample_id>_merge.pdb
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert PepMerge samples into bpep-like folder structure "
            "(peptides/proteins/merged_proteins)."
        )
    )
    parser.add_argument(
        "--pepmerge-root",
        type=Path,
        default=Path("~/PocketXMol/data_zip/PepMerge").expanduser(),
        help="PepMerge root directory containing sample subfolders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("~/PocketXMol/data_train/pepmerge/files").expanduser(),
        help="Output root directory to place peptides/proteins/merged_proteins.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="pepmerge",
        help="Filename prefix for output files (default: pepmerge).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing target files.",
    )
    return parser.parse_args()


def ensure_output_dirs(output_root: Path) -> dict[str, Path]:
    out_dirs = {
        "pep": output_root / "peptides",
        "pro": output_root / "proteins",
        "merge": output_root / "merged_proteins",
    }
    for directory in out_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return out_dirs


def collect_sample_dirs(pepmerge_root: Path) -> list[Path]:
    return sorted([p for p in pepmerge_root.iterdir() if p.is_dir()])


def build_target_paths(
    out_dirs: dict[str, Path], prefix: str, sample_id: str
) -> dict[str, Path]:
    return {
        "pep": out_dirs["pep"] / f"{prefix}_{sample_id}_pep.pdb",
        "pro": out_dirs["pro"] / f"{prefix}_{sample_id}_pro.pdb",
        "merge": out_dirs["merge"] / f"{prefix}_{sample_id}_merge.pdb",
    }


def copy_if_allowed(src: Path, dst: Path, overwrite: bool) -> bool:
    if dst.exists() and not overwrite:
        return False
    shutil.copy2(src, dst)
    return True


def main() -> None:
    args = parse_args()
    pepmerge_root = args.pepmerge_root
    output_root = args.output_root

    if not pepmerge_root.exists() or not pepmerge_root.is_dir():
        raise SystemExit(f"[ERROR] PepMerge root does not exist: {pepmerge_root}")

    out_dirs = ensure_output_dirs(output_root)
    sample_dirs = collect_sample_dirs(pepmerge_root)

    copied = 0
    skipped_missing = 0
    skipped_exists = 0

    for sample_dir in sample_dirs:
        sample_id = sample_dir.name
        source_paths = {
            "pep": sample_dir / "peptide.pdb",
            "pro": sample_dir / "receptor.pdb",
            "merge": sample_dir / "receptor_merge.pdb",
        }

        if not all(path.exists() for path in source_paths.values()):
            missing = [k for k, p in source_paths.items() if not p.exists()]
            print(f"[SKIP-MISSING] {sample_id} missing: {', '.join(missing)}")
            skipped_missing += 1
            continue

        target_paths = build_target_paths(out_dirs, args.prefix, sample_id)

        copied_all = True
        for key in ("pep", "pro", "merge"):
            ok = copy_if_allowed(source_paths[key], target_paths[key], args.overwrite)
            if not ok:
                copied_all = False

        if copied_all:
            copied += 1
            print(f"[OK] {sample_id}")
        else:
            skipped_exists += 1
            print(f"[SKIP-EXISTS] {sample_id} (use --overwrite to replace)")

    print("\n=== Done ===")
    print(f"PepMerge root   : {pepmerge_root}")
    print(f"Output root     : {output_root}")
    print(f"Total samples   : {len(sample_dirs)}")
    print(f"Copied samples  : {copied}")
    print(f"Missing skipped : {skipped_missing}")
    print(f"Exists skipped  : {skipped_exists}")


if __name__ == "__main__":
    main()
