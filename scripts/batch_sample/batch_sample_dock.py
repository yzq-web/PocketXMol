#!/usr/bin/env python3
"""Batch docking sampler for peptide inputs.

Given an input directory where each subdirectory is a sample, this script:
1) Converts ``peptide.pdb`` to ``peptide.sdf``.
2) Generates ``dock_pep.yml`` for each sample from a template config.
3) Runs docking sampling via ``scripts/sample_use.py`` for all samples.
"""

import argparse
from pathlib import Path
from typing import Dict, List

import yaml
try:
    from scripts.batch_sample.run_sampling_utils import (  # type: ignore[reportMissingImports]
        count_residues_in_pdb,
        get_pep_len_from_cfg,
        list_task_yaml_in_cfg_dir,
        load_pass_samples,
        run_sampling,
        run_sampling_loop,
        to_repo_relative,
    )
except ModuleNotFoundError:
    from run_sampling_utils import (  # type: ignore[reportMissingImports]
        count_residues_in_pdb,
        get_pep_len_from_cfg,
        list_task_yaml_in_cfg_dir,
        load_pass_samples,
        run_sampling,
        run_sampling_loop,
        to_repo_relative,
    )


def build_config(
    template_cfg: Dict,
    sample_dir: Path,
    repo_root: Path,
    batch_size: int,
    num_mols: int,
    pocket_radius: float,
    use_sdf_ligand: bool = False,
) -> Dict:
    """Create one sample-specific config dict."""
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

    sample_name = sample_dir.name

    receptor_path = sample_dir / "receptor.pdb"
    peptide_pdb_path = sample_dir / "peptide.pdb"
    peptide_sdf_path = sample_dir / "peptide.sdf"

    cfg["sample"]["batch_size"] = batch_size
    cfg["sample"]["num_mols"] = num_mols

    cfg["data"]["protein_path"] = to_repo_relative(receptor_path, repo_root)
    ligand_path = peptide_sdf_path if use_sdf_ligand else peptide_pdb_path
    cfg["data"]["input_ligand"] = to_repo_relative(ligand_path, repo_root)
    cfg["data"]["is_pep"] = True

    # Prefer SDF as pocket reference to avoid RDKit PDB valence parsing issues.
    # pocket_args["ref_ligand_path"] = to_repo_relative(peptide_sdf_path, repo_root)
    pocket_args["radius"] = pocket_radius
    cfg["data"]["pocket_args"] = pocket_args

    pocmol_args["data_id"] = f"dockpep_{sample_name}"
    pocmol_args["pdbid"] = sample_name
    cfg["data"]["pocmol_args"] = pocmol_args

    return cfg


def generate_config(
    sample_dir: Path,
    sample_cfg_path: Path,
    template_cfg: Dict,
    repo_root: Path,
    batch_size: int,
    num_mols: int,
    pocket_radius: float,
    use_sdf_ligand: bool = False,
) -> None:
    """Generate sample-specific dock_pep.yml."""
    receptor_path = sample_dir / "receptor.pdb"
    peptide_pdb_path = sample_dir / "peptide.pdb"
    peptide_sdf_path = sample_dir / "peptide.sdf"
    if not receptor_path.exists():
        raise FileNotFoundError(f"Missing receptor file: {receptor_path}")
    if not peptide_pdb_path.exists():
        raise FileNotFoundError(f"Missing peptide file: {peptide_pdb_path}")
    if use_sdf_ligand and (not peptide_sdf_path.exists()):
        raise FileNotFoundError(f"Missing peptide file: {peptide_sdf_path}")

    cfg = build_config(
        template_cfg=template_cfg,
        sample_dir=sample_dir,
        repo_root=repo_root,
        batch_size=batch_size,
        num_mols=num_mols,
        pocket_radius=pocket_radius,
        use_sdf_ligand=use_sdf_ligand,
    )
    with sample_cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch generate dock_pep.yml and run docking sampling."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Input directory where each child directory is one sample.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output root directory. Results are stored in output_dir/<sample_name>.",
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
        "--stop_on_error",
        action="store_true",
        help="Stop immediately when any sample fails.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    gt_root_dir = output_dir / "gt_pocket"
    gt_output_dir = gt_root_dir / "output"
    cfg_dir = input_dir / "configs" / "gt_pocket"

    template_path = args.template
    if not template_path.is_absolute():
        template_path = (repo_root / template_path).resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not template_path.exists():
        raise FileNotFoundError(f"Template yml not found: {template_path}")

    gt_output_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)
    with template_path.open("r", encoding="utf-8") as f:
        template_cfg = yaml.safe_load(f)

    pass_samples = load_pass_samples(input_dir)
    if not pass_samples:
        print(f"No pass_all_checks=True samples found in: {input_dir / 'meta' / 'peptide_all_check.csv'}")
        return
    sample_names = [str(item["data_id"]) for item in pass_samples]
    sample_dirs = [input_dir / "complex" / name for name in sample_names]
    print(
        f"Selected {len(sample_dirs)} pass samples from: "
        f"{input_dir / 'meta' / 'peptide_all_check.csv'}"
    )
    print("Processing order: shorter peptides first.")

    failures: List[str] = []
    sample_meta = {str(item["data_id"]): item for item in pass_samples}
    remaining_dirs = [p for p in sample_dirs if p.is_dir()]
    missing_dirs = [p for p in sample_dirs if not p.is_dir()]
    for missing_dir in missing_dirs:
        msg = f"[Init] {missing_dir.name}: sample directory not found"
        failures.append(msg)
        print(f"[FAILED] {msg}")

    print("\n=== Step 1/2: Generate dock_pep.yml for pass samples ===")
    existing_cfg = list_task_yaml_in_cfg_dir(cfg_dir)
    if existing_cfg:
        cfg_paths = existing_cfg
        print(
            f"[Step1] SKIP: reuse {len(cfg_paths)} existing yml under {cfg_dir} "
            "(skip generation; Step 2 order by pep_len from each config)"
        )
    else:
        cfg_paths = []
        for sample_dir in remaining_dirs:
            try:
                peptide_path = sample_dir / "peptide.pdb"
                pep_len = count_residues_in_pdb(peptide_path)
                if pep_len > 25:
                    print(f"[Step1 SKIP] {sample_dir.name}: pep_len={pep_len} > 25")
                    continue
                effective_batch_size = 25 if pep_len > 30 else (50 if pep_len > 10 else args.batch_size)
                cfg_path = cfg_dir / f"dock_pep_{sample_dir.name}.yml"
                generate_config(
                    sample_dir=sample_dir,
                    sample_cfg_path=cfg_path,
                    template_cfg=template_cfg,
                    repo_root=repo_root,
                    batch_size=effective_batch_size,
                    num_mols=args.num_mols,
                    pocket_radius=args.pocket_radius,
                    use_sdf_ligand=False,
                )
                print(
                    f"[Step1 OK] {sample_dir.name} "
                    f"(pep_len={pep_len}, batch_size={effective_batch_size}, "
                    f"input_ligand=peptide.pdb, "
                    f"pocket_radius={args.pocket_radius})"
                )
                cfg_paths.append(cfg_path)
            except Exception as exc:
                msg = f"[Step1] {sample_dir.name}: {exc}"
                failures.append(msg)
                print(f"[FAILED] {msg}")
                if args.stop_on_error:
                    remaining_dirs = []
                    break
        print(f"[Step1] generated yml: {len(cfg_paths)} at {cfg_dir}")

    print("\n=== Step 2/2: Run sampling for pass samples ===")
    cfg_paths_sorted = sorted(cfg_paths, key=get_pep_len_from_cfg)
    step2_stats = run_sampling_loop(
        cfg_paths_sorted=cfg_paths_sorted,
        output_dir=gt_output_dir,
        num_mols=args.num_mols,
        get_pep_len=get_pep_len_from_cfg,
        run_sampling_for_cfg=lambda cfg_path: run_sampling(
            cfg_path=cfg_path,
            repo_root=repo_root,
            output_dir=gt_output_dir,
            config_model=args.config_model,
            device=args.device,
        ),
        failures=failures,
        stop_on_error=args.stop_on_error,
        run_time_file=gt_root_dir / "run_time_gt_pocket.txt",
        failure_log_file=gt_root_dir / "failure_log_gt_pocket.txt",
        log_prefix="[Step2]",
    )
    skipped_done = int(step2_stats["skipped_done"])
    skipped_long_pep = int(step2_stats["skipped_long_pep"])

    print("\n=== Batch finished ===")
    print(
        f"Total(pass): {len(sample_dirs)}, Skipped(done): {skipped_done}, "
        f"Skipped(pep_len>25): {skipped_long_pep}, "
        f"Failed: {len(failures)}"
    )
    if failures:
        print("Failure details:")
        for item in failures:
            print(f"- {item}")


if __name__ == "__main__":
    main()
