#!/usr/bin/env python3
"""Batch docking sampler using random pocket centers from each sample subdir.

This script runs in two steps:
1) Traverse input_dir/complex/<subdir_name>/random_pocket/pocket_meta.csv and generate
   one yml per pocket record into input_dir/configs/random_pocket.
2) Run PocketXMol docking (scripts/sample_use.py) for generated yml files.
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

try:
    from scripts.batch_sample.run_sampling_utils import (  # type: ignore[reportMissingImports]
        FlowStyleList,
        FlowStyleSafeDumper,
        count_residues_in_pdb,
        get_pep_len_from_cfg,
        list_task_yaml_in_cfg_dir,
        load_pass_samples,
        parse_pocket_center,
        run_sampling,
        run_sampling_loop,
        to_repo_relative,
    )
except ModuleNotFoundError:
    from run_sampling_utils import (  # type: ignore[reportMissingImports]
        FlowStyleList,
        FlowStyleSafeDumper,
        count_residues_in_pdb,
        get_pep_len_from_cfg,
        list_task_yaml_in_cfg_dir,
        load_pass_samples,
        parse_pocket_center,
        run_sampling,
        run_sampling_loop,
        to_repo_relative,
    )


def load_selected_rows(input_dir: Path, sample_ids: List[str]) -> Tuple[List[Dict[str, object]], List[str]]:
    """Load random pocket rows from each complex/<subdir>/random_pocket/pocket_meta.csv."""
    complex_dir = input_dir / "complex"
    if not complex_dir.is_dir():
        raise FileNotFoundError(f"Missing complex directory: {complex_dir}")

    selected_rows: List[Dict[str, object]] = []
    failures: List[str] = []
    for data_id in sample_ids:
        sample_dir = complex_dir / data_id
        if not sample_dir.is_dir():
            failures.append(f"[Step1] {data_id}: sample directory not found in {complex_dir}")
            continue
        pocket_meta_path = sample_dir / "random_pocket" / "pocket_meta.csv"
        if not pocket_meta_path.is_file():
            failures.append(f"[Step1] {data_id}: missing {pocket_meta_path}")
            continue

        try:
            with pocket_meta_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                row_count = 0
                for idx, row in enumerate(reader):
                    row_count += 1
                    pocket_idx = str(row.get("pocket_idx", "")).strip() or f"random_pocket_{idx}"
                    pocket_center = parse_pocket_center(str(row.get("pocket_center", "")))
                    if pocket_center is None:
                        failures.append(
                            f"[Step1] {data_id}/{pocket_idx}: invalid pocket_center in {pocket_meta_path}"
                        )
                        continue
                    selected_rows.append(
                        {
                            "data_id": data_id,
                            "pocket_idx": pocket_idx,
                            "pocket_center": pocket_center,
                        }
                    )
                if row_count == 0:
                    failures.append(f"[Step1] {data_id}: empty csv {pocket_meta_path}")
        except Exception as exc:
            failures.append(f"[Step1] {data_id}: failed to read {pocket_meta_path}: {exc}")
    return selected_rows, failures


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

    ref_lp = pocket_args.get("ref_ligand_path")
    if ref_lp:
        try:
            pocket_args["ref_ligand_path"] = to_repo_relative(Path(str(ref_lp)).resolve(), repo_root)
        except Exception:
            pass

    data_id = str(row["data_id"])
    pocket_idx = str(row["pocket_idx"])
    pocket_center = list(row["pocket_center"])

    receptor_path = (input_dir / "complex" / data_id / "receptor.pdb").resolve()
    peptide_path = (input_dir / "complex" / data_id / "peptide.pdb").resolve()

    cfg["sample"]["batch_size"] = batch_size
    cfg["sample"]["num_mols"] = num_mols

    cfg["data"]["protein_path"] = to_repo_relative(receptor_path, repo_root)
    cfg["data"]["input_ligand"] = to_repo_relative(peptide_path, repo_root)
    cfg["data"]["is_pep"] = True

    pocket_args["radius"] = pocket_radius
    pocket_args["pocket_coord"] = FlowStyleList([float(x) for x in pocket_center])
    cfg["data"]["pocket_args"] = pocket_args

    pocmol_args["data_id"] = f"dockrandom_{data_id}_{pocket_idx}"
    pocmol_args["pdbid"] = data_id
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

    for row in selected_rows:
        data_id = str(row["data_id"])
        pocket_idx = str(row["pocket_idx"])
        receptor_path = input_dir / "complex" / data_id / "receptor.pdb"
        peptide_path = input_dir / "complex" / data_id / "peptide.pdb"
        if not receptor_path.is_file() or not peptide_path.is_file():
            failures.append(
                f"[Step1] {data_id}/{pocket_idx}: missing receptor.pdb or peptide.pdb"
            )
            continue

        pep_len = count_residues_in_pdb(peptide_path)
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
        cfg_name = f"dock_random_{data_id}_{pocket_idx}.yml"
        cfg_path = cfg_dir / cfg_name
        with cfg_path.open("w", encoding="utf-8") as f:
            yaml.dump(
                cfg,
                f,
                sort_keys=False,
                allow_unicode=True,
                Dumper=FlowStyleSafeDumper,
            )
        cfg_paths.append(cfg_path)
        print(
            f"[Step1 OK] {cfg_name} (pep_len={pep_len}, batch_size={effective_batch_size}, "
            f"radius={pocket_radius})"
        )

    return cfg_paths, failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate yml from random pocket csv and run PocketXMol docking."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Input directory containing complex/.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output root; docking outputs go to output_dir/random_pocket/output.",
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
        default=15.0,
        help="Value written to data.pocket_args.radius.",
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

    cfg_dir = input_dir / "configs" / "random_pocket"
    dock_output_dir = output_dir / "random_pocket" / "output"
    dock_output_dir.mkdir(parents=True, exist_ok=True)

    with template_path.open("r", encoding="utf-8") as f:
        template_cfg = yaml.safe_load(f)

    failures: List[str] = []
    cfg_dir.mkdir(parents=True, exist_ok=True)
    existing_cfg = list_task_yaml_in_cfg_dir(cfg_dir)
    if existing_cfg:
        cfg_paths = existing_cfg
        print("\n=== Step 1/2: Generate yml files from random pocket meta ===")
        print(
            f"[Step1] SKIP: reuse {len(cfg_paths)} existing yml under {cfg_dir} "
            "(skip generation; Step 2 order by pep_len from each config)"
        )
    else:
        pass_samples = load_pass_samples(input_dir)
        if not pass_samples:
            print(
                f"No pass_all_checks=True samples found in: "
                f"{input_dir / 'meta' / 'peptide_all_check.csv'}"
            )
            return
        sample_ids = [str(item["data_id"]) for item in pass_samples]
        print(
            f"Selected {len(sample_ids)} pass samples from: "
            f"{input_dir / 'meta' / 'peptide_all_check.csv'}"
        )

        selected_rows, load_failures = load_selected_rows(input_dir=input_dir, sample_ids=sample_ids)
        failures.extend(load_failures)
        if not selected_rows:
            print("No valid random pocket rows found in complex/*/random_pocket/pocket_meta.csv")
            if failures:
                for item in failures:
                    print(f"[FAILED] {item}")
            return

        print("\n=== Step 1/2: Generate yml files from random pocket meta ===")
        cfg_paths, step1_failures = generate_configs(
            selected_rows=selected_rows,
            template_cfg=template_cfg,
            input_dir=input_dir,
            repo_root=repo_root,
            cfg_dir=cfg_dir,
            batch_size=args.batch_size,
            num_mols=args.num_mols,
            pocket_radius=args.pocket_radius,
        )
        failures.extend(step1_failures)
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
        run_time_file=output_dir / "random_pocket" / "run_time_random_pocket.txt",
        failure_log_file=output_dir / "random_pocket" / "failure_log_random_pocket.txt",
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
