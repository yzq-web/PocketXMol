#!/usr/bin/env python3
"""Batch docking sampler for shuffled pocket setting.

Given input_dir/complex/<subdir_name> samples, this script:
1) Builds cross-sample receptor/peptide pairs (excluding self-pairs), writes one yml per pair.
2) Runs docking sampling via scripts/sample_use.py for all generated yml files.
"""

import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

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

SHUFFLE_PAIR_SEED = 20260430


def build_config(
    template_cfg: Dict,
    receptor_dir: Path,
    peptide_dir: Path,
    repo_root: Path,
    batch_size: int,
    num_mols: int,
    pocket_radius: float,
) -> Dict:
    """Create one shuffled sample-specific config dict."""
    cfg = dict(template_cfg)
    cfg["sample"] = dict(cfg.get("sample", {}))
    cfg["data"] = dict(cfg.get("data", {}))

    pocket_args = dict(cfg["data"].get("pocket_args", {}))
    pocmol_args = dict(cfg["data"].get("pocmol_args", {}))

    receptor_path = receptor_dir / "receptor.pdb"
    ligand_path = peptide_dir / "peptide.pdb"
    ref_ligand_path = receptor_dir / "peptide.pdb"

    pair_name = f"{receptor_dir.name}__{peptide_dir.name}"

    cfg["sample"]["batch_size"] = batch_size
    cfg["sample"]["num_mols"] = num_mols

    cfg["data"]["protein_path"] = to_repo_relative(receptor_path, repo_root)
    cfg["data"]["input_ligand"] = to_repo_relative(ligand_path, repo_root)
    cfg["data"]["is_pep"] = True

    pocket_args["radius"] = pocket_radius
    pocket_args["ref_ligand_path"] = to_repo_relative(ref_ligand_path, repo_root)
    cfg["data"]["pocket_args"] = pocket_args

    pocmol_args["data_id"] = f"dockshuffle_{pair_name}"
    pocmol_args["pdbid"] = receptor_dir.name
    cfg["data"]["pocmol_args"] = pocmol_args
    return cfg


def generate_pair_configs(
    complex_dir: Path,
    sample_ids: List[str],
    cfg_dir: Path,
    template_cfg: Dict,
    repo_root: Path,
    batch_size: int,
    num_mols: int,
    pocket_radius: float,
) -> Tuple[List[Path], List[str]]:
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_paths: List[Path] = []
    failures: List[str] = []
    rng = random.Random(SHUFFLE_PAIR_SEED)
    sample_dirs: List[Path] = []
    for sample_id in sample_ids:
        sample_dir = complex_dir / sample_id
        if not sample_dir.is_dir():
            failures.append(f"[Step1] {sample_id}: sample directory not found in {complex_dir}")
            continue
        sample_dirs.append(sample_dir)

    valid_receptor_dirs: List[Path] = []
    for receptor_dir in sample_dirs:
        receptor_path = receptor_dir / "receptor.pdb"
        ref_ligand_path = receptor_dir / "peptide.pdb"
        if not receptor_path.is_file() or not ref_ligand_path.is_file():
            failures.append(
                f"[Step1] {receptor_dir.name}: missing receptor.pdb or peptide.pdb"
            )
            continue
        valid_receptor_dirs.append(receptor_dir)

    for peptide_dir in sample_dirs:
        ligand_path = peptide_dir / "peptide.pdb"
        if not ligand_path.is_file():
            failures.append(
                f"[Step1] {peptide_dir.name}: missing peptide.pdb"
            )
            continue
        pep_len = count_residues_in_pdb(ligand_path)
        if pep_len > 25:
            print(f"[Step1 SKIP] {peptide_dir.name}: pep_len={pep_len} > 25")
            continue

        candidate_receptors = [
            receptor_dir
            for receptor_dir in valid_receptor_dirs
            if receptor_dir.name != peptide_dir.name
        ]
        if not candidate_receptors:
            failures.append(
                f"[Step1] {peptide_dir.name}: no available non-self receptor.pdb to pair"
            )
            continue

        receptor_dir = rng.choice(candidate_receptors)
        effective_batch_size = 25 if pep_len > 30 else (50 if pep_len > 10 else batch_size)

        cfg = build_config(
            template_cfg=template_cfg,
            receptor_dir=receptor_dir,
            peptide_dir=peptide_dir,
            repo_root=repo_root,
            batch_size=effective_batch_size,
            num_mols=num_mols,
            pocket_radius=pocket_radius,
        )
        cfg_name = f"dock_shuffle_{receptor_dir.name}__{peptide_dir.name}.yml"
        cfg_path = cfg_dir / cfg_name
        with cfg_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        cfg_paths.append(cfg_path)
        print(
            f"[Step1 OK] {cfg_name} "
            f"(receptor={receptor_dir.name}, ligand={peptide_dir.name}, "
            f"pep_len={pep_len}, batch_size={effective_batch_size}, "
            f"pocket_radius={pocket_radius})"
        )

    # Keep shorter ligand peptides first in Step2. 优先运行pep_len较小的样本
    cfg_paths.sort(key=lambda p: count_residues_in_pdb(_get_ligand_path_from_cfg(p)))
    return cfg_paths, failures


def _get_ligand_path_from_cfg(cfg_path: Path) -> Path:
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return Path(str(cfg.get("data", {}).get("input_ligand", "")))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch generate shuffle-pocket yml and run docking sampling."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Input directory containing complex/ subdirectory.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output root directory.",
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
        help="Stop immediately when any pair task fails.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    complex_dir = input_dir / "complex"
    cfg_dir = input_dir / "configs" / "shuffle_pocket"
    shuffle_root_dir = output_dir / "shuffle_pocket"
    shuffle_output_dir = shuffle_root_dir / "output"

    template_path = args.template
    if not template_path.is_absolute():
        template_path = (repo_root / template_path).resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not complex_dir.exists() or not complex_dir.is_dir():
        raise FileNotFoundError(f"Missing complex directory: {complex_dir}")
    if not template_path.exists():
        raise FileNotFoundError(f"Template yml not found: {template_path}")

    shuffle_output_dir.mkdir(parents=True, exist_ok=True)
    with template_path.open("r", encoding="utf-8") as f:
        template_cfg = yaml.safe_load(f)

    cfg_dir.mkdir(parents=True, exist_ok=True)
    existing_cfg = list_task_yaml_in_cfg_dir(cfg_dir)
    if existing_cfg:
        cfg_paths = existing_cfg
        failures: List[str] = []
        print("\n=== Step 1/2: Generate shuffle pocket yml ===")
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

        print("\n=== Step 1/2: Generate shuffle pocket yml ===")
        print(f"[Step1] fixed random seed: {SHUFFLE_PAIR_SEED}")
        cfg_paths, failures = generate_pair_configs(
            complex_dir=complex_dir,
            sample_ids=sample_ids,
            cfg_dir=cfg_dir,
            template_cfg=template_cfg,
            repo_root=repo_root,
            batch_size=args.batch_size,
            num_mols=args.num_mols,
            pocket_radius=args.pocket_radius,
        )
        print(f"[Step1] generated yml: {len(cfg_paths)} at {cfg_dir}")
        if failures:
            for item in failures:
                print(f"[FAILED] {item}")

    if not cfg_paths:
        print("No yml generated. Stop.")
        return

    print("\n=== Step 2/2: Run sampling for shuffle pocket ===")
    cfg_paths_sorted = sorted(cfg_paths, key=get_pep_len_from_cfg)
    step2_stats = run_sampling_loop(
        cfg_paths_sorted=cfg_paths_sorted,
        output_dir=shuffle_output_dir,
        num_mols=args.num_mols,
        get_pep_len=get_pep_len_from_cfg,
        run_sampling_for_cfg=lambda cfg_path: run_sampling(
            cfg_path=cfg_path,
            repo_root=repo_root,
            output_dir=shuffle_output_dir,
            config_model=args.config_model,
            device=args.device,
        ),
        failures=failures,
        stop_on_error=args.stop_on_error,
        run_time_file=shuffle_root_dir / "run_time_shuffle_pocket.txt",
        failure_log_file=shuffle_root_dir / "failure_log_shuffle_pocket.txt",
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
