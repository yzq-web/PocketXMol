#!/usr/bin/env python3
"""Batch docking sampler for secondary-structure-similar pocket setting.

This script runs in two steps:
1) Read input_dir/meta/peptide_ss.csv and keep rows with sim_ratio > 0.6, then
   generate one yml per valid row into input_dir/configs/ss_pocket.
2) Run PocketXMol docking (scripts/sample_use.py) for generated yml files.
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

try:
    from scripts.batch_sample.run_sampling_utils import (  # type: ignore[reportMissingImports]
        count_residues_in_pdb,
        filter_rows_by_pass_peptide_check,
        get_pep_len_from_cfg,
        list_task_yaml_in_cfg_dir,
        run_sampling,
        run_sampling_loop,
        to_repo_relative,
    )
except ModuleNotFoundError:
    from run_sampling_utils import (  # type: ignore[reportMissingImports]
        count_residues_in_pdb,
        filter_rows_by_pass_peptide_check,
        get_pep_len_from_cfg,
        list_task_yaml_in_cfg_dir,
        run_sampling,
        run_sampling_loop,
        to_repo_relative,
    )


def _parse_float(value: str) -> Optional[float]:
    try:
        return float(str(value).strip())
    except Exception:
        return None


def load_selected_rows(meta_csv_path: Path, sim_ratio_threshold: float = 0.6) -> List[Dict[str, object]]:
    """Load rows where sim_ratio > threshold and required ids are valid."""
    if not meta_csv_path.is_file():
        raise FileNotFoundError(f"Missing csv file: {meta_csv_path}")

    selected: List[Dict[str, object]] = []
    with meta_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data_id = str(row.get("data_id", "")).strip()
            sim_ss = str(row.get("sim_ss", "")).strip()
            sim_ratio = _parse_float(str(row.get("sim_ratio", "")))
            if not data_id or not sim_ss:
                continue
            if sim_ratio is None or sim_ratio <= sim_ratio_threshold:
                continue
            selected.append(
                {
                    "data_id": data_id,
                    "sim_ss": sim_ss,
                    "sim_ratio": sim_ratio,
                }
            )
    return selected


def build_config_from_row(
    template_cfg: Dict,
    input_dir: Path,
    repo_root: Path,
    row: Dict[str, object],
    batch_size: int,
    num_mols: int,
    pocket_radius: float,
) -> Dict:
    cfg = dict(template_cfg)
    cfg["sample"] = dict(cfg.get("sample", {}))
    cfg["data"] = dict(cfg.get("data", {}))

    pocket_args = dict(cfg["data"].get("pocket_args", {}))
    pocmol_args = dict(cfg["data"].get("pocmol_args", {}))

    data_id = str(row["data_id"])
    sim_ss = str(row["sim_ss"])

    ligand_path = (input_dir / "complex" / data_id / "peptide.pdb").resolve()
    receptor_path = (input_dir / "complex" / sim_ss / "receptor.pdb").resolve()
    ref_ligand_path = (input_dir / "complex" / sim_ss / "peptide.pdb").resolve()

    cfg["sample"]["batch_size"] = batch_size
    cfg["sample"]["num_mols"] = num_mols

    cfg["data"]["protein_path"] = to_repo_relative(receptor_path, repo_root)
    cfg["data"]["input_ligand"] = to_repo_relative(ligand_path, repo_root)
    cfg["data"]["is_pep"] = True

    pocket_args["radius"] = pocket_radius
    pocket_args["ref_ligand_path"] = to_repo_relative(ref_ligand_path, repo_root)
    cfg["data"]["pocket_args"] = pocket_args

    pocmol_args["data_id"] = f"dockss_{sim_ss}__{data_id}"
    pocmol_args["pdbid"] = sim_ss
    cfg["data"]["pocmol_args"] = pocmol_args
    return cfg


def generate_configs(
    selected_rows: List[Dict[str, object]],
    template_cfg: Dict,
    input_dir: Path,
    repo_root: Path,
    cfg_dir: Path,
    batch_size: int,
    num_mols: int,
    pocket_radius: float,
) -> Tuple[List[Path], List[str]]:
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_paths: List[Path] = []
    failures: List[str] = []
    used_names = set()

    for idx, row in enumerate(selected_rows):
        data_id = str(row["data_id"])
        sim_ss = str(row["sim_ss"])
        sim_ratio = float(row["sim_ratio"])

        ligand_path = input_dir / "complex" / data_id / "peptide.pdb"
        receptor_path = input_dir / "complex" / sim_ss / "receptor.pdb"
        ref_ligand_path = input_dir / "complex" / sim_ss / "peptide.pdb"
        if not ligand_path.is_file():
            failures.append(f"[Step1] {sim_ss}__{data_id}: missing ligand peptide.pdb ({ligand_path})")
            continue
        if not receptor_path.is_file():
            failures.append(f"[Step1] {sim_ss}__{data_id}: missing receptor.pdb ({receptor_path})")
            continue
        if not ref_ligand_path.is_file():
            failures.append(f"[Step1] {sim_ss}__{data_id}: missing ref_ligand peptide.pdb ({ref_ligand_path})")
            continue
        pep_len = count_residues_in_pdb(ligand_path)
        if pep_len > 25:
            print(f"[Step1 SKIP] {data_id}: pep_len={pep_len} > 25")
            continue
        effective_batch_size = 25 if pep_len > 30 else (50 if pep_len > 10 else batch_size)

        cfg = build_config_from_row(
            template_cfg=template_cfg,
            input_dir=input_dir,
            repo_root=repo_root,
            row=row,
            batch_size=effective_batch_size,
            num_mols=num_mols,
            pocket_radius=pocket_radius,
        )
        cfg_name = f"dock_ss_{sim_ss}__{data_id}.yml"
        if cfg_name in used_names:
            cfg_name = f"dock_ss_{sim_ss}__{data_id}__row{idx}.yml"
        used_names.add(cfg_name)
        cfg_path = cfg_dir / cfg_name
        with cfg_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        cfg_paths.append(cfg_path)
        print(
            f"[Step1 OK] {cfg_name} "
            f"(protein={sim_ss}, ligand={data_id}, sim_ratio={sim_ratio:.3f}, radius={pocket_radius})"
        )

    return cfg_paths, failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate yml from peptide_ss.csv and run PocketXMol docking."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Input directory containing complex/ and meta/.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output root; docking outputs go to output_dir/ss_pocket/output.",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("configs/sample/examples/dock_pep_test.yml"),
        help="Template task yml path.",
    )
    parser.add_argument(
        "--config_model",
        type=str,
        default="configs/sample/pxm.yml",
        help="Model config path passed to sample_use.py.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for sampling, e.g. cuda:0.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="Value written to sample.batch_size.",
    )
    parser.add_argument(
        "--num_mols",
        type=int,
        required=True,
        help="Value written to sample.num_mols.",
    )
    parser.add_argument(
        "--pocket_radius",
        type=float,
        default=10.0,
        help="Value written to data.pocket_args.radius.",
    )
    parser.add_argument(
        "--sim_ratio_threshold",
        type=float,
        default=0.6,
        help="Filter rows with sim_ratio > threshold.",
    )
    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        help="Stop immediately when any docking task fails.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    template_path = args.template
    if not template_path.is_absolute():
        template_path = (repo_root / template_path).resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not template_path.is_file():
        raise FileNotFoundError(f"Template yml not found: {template_path}")

    meta_csv_path = input_dir / "meta" / "peptide_ss.csv"
    cfg_dir = input_dir / "configs" / "ss_pocket"
    dock_output_dir = output_dir / "ss_pocket" / "output"
    dock_output_dir.mkdir(parents=True, exist_ok=True)

    with template_path.open("r", encoding="utf-8") as f:
        template_cfg = yaml.safe_load(f)

    cfg_dir.mkdir(parents=True, exist_ok=True)
    existing_cfg = list_task_yaml_in_cfg_dir(cfg_dir)
    if existing_cfg:
        cfg_paths = existing_cfg
        failures: List[str] = []
        print("\n=== Step 1/2: Generate yml files from peptide_ss.csv ===")
        print(
            f"[Step1] SKIP: reuse {len(cfg_paths)} existing yml under {cfg_dir} "
            "(skip generation; Step 2 order by pep_len from each config)"
        )
    else:
        selected_rows = load_selected_rows(
            meta_csv_path=meta_csv_path,
            sim_ratio_threshold=args.sim_ratio_threshold,
        )
        if not selected_rows:
            print(
                f"No valid rows with sim_ratio > {args.sim_ratio_threshold} found in: {meta_csv_path}"
            )
            return

        n_ss = len(selected_rows)
        selected_rows, n_rm = filter_rows_by_pass_peptide_check(
            selected_rows, input_dir, id_keys=("data_id",)
        )
        if n_rm:
            print(
                f"[Step1] peptide_all_check filter: removed {n_rm} / {n_ss} rows "
                f"(pass_all_checks & pep_len<25 only; by ligand data_id)"
            )
        if not selected_rows:
            print(
                "No rows remain after meta/peptide_all_check.csv filter "
                "(pass_all_checks, pep_len<25)."
            )
            return

        print("\n=== Step 1/2: Generate yml files from peptide_ss.csv ===")
        cfg_paths, failures = generate_configs(
            selected_rows=selected_rows,
            template_cfg=template_cfg,
            input_dir=input_dir,
            repo_root=repo_root,
            cfg_dir=cfg_dir,
            batch_size=args.batch_size,
            num_mols=args.num_mols,
            pocket_radius=args.pocket_radius,
        )
        print(f"[Step1] selected rows: {len(selected_rows)}")
        print(f"[Step1] generated yml: {len(cfg_paths)} at {cfg_dir}")
        if failures:
            for item in failures:
                print(f"[FAILED] {item}")

    if not cfg_paths:
        print("No yml generated. Stop.")
        return

    print("\n=== Step 2/2: Run PocketXMol docking ===")
    cfg_paths_sorted = sorted(cfg_paths, key=get_pep_len_from_cfg)
    step2_stats = run_sampling_loop(
        cfg_paths_sorted=cfg_paths_sorted,
        output_dir=dock_output_dir,
        num_mols=args.num_mols,
        get_pep_len=get_pep_len_from_cfg,
        run_sampling_for_cfg=lambda cfg_path: run_sampling(
            cfg_path=cfg_path,
            repo_root=repo_root,
            output_dir=dock_output_dir,
            config_model=args.config_model,
            device=args.device,
        ),
        failures=failures,
        stop_on_error=args.stop_on_error,
        run_time_file=output_dir / "ss_pocket" / "run_time_ss_pocket.txt",
        failure_log_file=output_dir / "ss_pocket" / "failure_log_ss_pocket.txt",
        log_prefix="[Step2]",
    )
    skipped_done = int(step2_stats["skipped_done"])
    skipped_long_pep = int(step2_stats["skipped_long_pep"])

    print("\n=== Batch finished ===")
    print(
        f"Total yml: {len(cfg_paths)}, Skipped(done): {skipped_done}, "
        f"Skipped(pep_len>25): {skipped_long_pep}, Failed: {len(failures)}"
    )
    if failures:
        print("Failure details:")
        for item in failures:
            print(f"- {item}")


if __name__ == "__main__":
    main()
