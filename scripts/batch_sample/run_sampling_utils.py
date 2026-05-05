#!/usr/bin/env python3
"""Shared utilities for Step2 batch sampling loops."""

import csv
import os
import subprocess
import sys
import shutil
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import yaml


class FlowStyleList(list):
    """List represented in YAML flow style, e.g. [x, y, z]."""


class FlowStyleSafeDumper(yaml.SafeDumper):
    """Safe dumper that supports FlowStyleList."""


def _represent_flow_style_list(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


FlowStyleSafeDumper.add_representer(FlowStyleList, _represent_flow_style_list)


def to_repo_relative(path: Path, repo_root: Path) -> str:
    """Return path relative to repo root when possible, otherwise absolute path."""
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def count_residues_in_pdb(pdb_path: Path) -> int:
    """Count distinct residues from PDB ATOM/HETATM records."""
    residues = set()
    with pdb_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            chain_id = line[21].strip() or "_"
            resseq = line[22:26].strip()
            icode = line[26].strip()
            resname = line[17:20].strip()
            residues.add((chain_id, resseq, icode, resname))
    return len(residues)


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def parse_int(value: str, default: int = 9999) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return default


def parse_pocket_center(center_text: str) -> Optional[List[float]]:
    """Parse 'x,y,z' to [x, y, z]."""
    items = [x.strip() for x in str(center_text).split(",")]
    if len(items) != 3:
        return None
    coords: List[float] = []
    for item in items:
        try:
            coords.append(float(item))
        except Exception:
            return None
    return coords


def load_pass_samples(input_dir: Path) -> List[Dict[str, object]]:
    """
    Load pass-filtered samples from meta/peptide_all_check.csv.
    Supports both `data_id` and `subdir_name` sample id columns.
    """
    check_csv_path = input_dir / "meta" / "peptide_all_check.csv"
    if not check_csv_path.exists():
        raise FileNotFoundError(f"Missing check CSV: {check_csv_path}")

    selected: List[Dict[str, object]] = []
    with check_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = row.get("data_id") or row.get("subdir_name")
            if not sample_id:
                continue
            if not parse_bool(row.get("pass_all_checks", "")):
                continue
            pep_len = parse_int(row.get("pep_len", ""), default=9999)
            if pep_len >= 25:
                continue
            selected.append(
                {
                    "data_id": sample_id,
                    "pep_len": pep_len,
                }
            )
    selected.sort(key=lambda x: (int(x["pep_len"]), str(x["data_id"])))
    return selected


def filter_rows_by_pass_peptide_check(
    rows: List[Dict[str, object]],
    input_dir: Path,
    id_keys: Tuple[str, ...] = ("data_id",),
) -> Tuple[List[Dict[str, object]], int]:
    """
    Keep only rows whose ID fields all appear in ``load_pass_samples(input_dir)``
    (same rules as ``meta/peptide_all_check.csv``: pass_all_checks, pep_len < 25).

    Returns ``(filtered_rows, num_removed)``.
    """
    allowed = {str(s["data_id"]) for s in load_pass_samples(input_dir)}
    kept: List[Dict[str, object]] = []
    for r in rows:
        if all(str(r.get(k, "")).strip() in allowed for k in id_keys):
            kept.append(r)
    return kept, len(rows) - len(kept)


def run_sampling(
    cfg_path: Path,
    repo_root: Path,
    output_dir: Path,
    config_model: str,
    device: str,
) -> None:
    """Run docking sampling for one generated config."""
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config file: {cfg_path}")

    sample_outdir = output_dir / cfg_path.stem
    sample_outdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "sample_use.py"),
        "--config_task",
        str(cfg_path),
        "--config_model",
        config_model,
        "--outdir",
        str(sample_outdir),
        "--use_outdir_as_logdir",
        "--device",
        device,
    ]
    run_line = f"[RUN] {' '.join(cmd)}"
    log_path = sample_outdir / "log_complete.txt"
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    with log_path.open("w", encoding="utf-8", newline="") as logf:
        logf.write(run_line + "\n")
        logf.flush()
        print(run_line)
        # 使用 PIPE 在父进程里转发，避免把自定义对象当作 stdout（会触发 fileno/isatty 等）
        proc = subprocess.Popen(
            cmd,
            cwd=str(repo_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert proc.stdout is not None
        while True:
            chunk = proc.stdout.read(8192)
            if not chunk:
                break
            logf.write(chunk)
            logf.flush()
            sys.stdout.write(chunk)
            sys.stdout.flush()
        ret = proc.wait()
        if ret != 0:
            raise subprocess.CalledProcessError(ret, cmd)


def get_pep_len_from_cfg(cfg_path: Path) -> int:
    """Read peptide path from yml and count peptide residue length."""
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        peptide_path = Path(str(cfg.get("data", {}).get("input_ligand", "")))
        if peptide_path.is_file():
            return count_residues_in_pdb(peptide_path)
    except Exception:
        pass
    return 9999


def list_task_yaml_in_cfg_dir(cfg_dir: Path) -> List[Path]:
    """
    List ``*.yml`` / ``*.yaml`` directly under cfg_dir (non-recursive), sorted by file name.
    Returns an empty list if cfg_dir is missing or is not a directory.
    """
    if not cfg_dir.is_dir():
        return []
    paths = [
        p
        for p in cfg_dir.iterdir()
        if p.is_file() and p.suffix.lower() in (".yml", ".yaml")
    ]
    paths.sort(key=lambda p: p.name.lower())
    return paths


def clear_directory_contents(dir_path: Path) -> None:
    """Remove all files/subdirectories under dir_path, keep dir_path itself."""
    if not dir_path.exists():
        return
    for child in dir_path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def count_non_empty_data_rows(file_path: Path) -> int:
    """Count non-empty data rows (excluding header) in a CSV-like text file."""
    count = 0
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return max(count - 1, 0)


def _write_session_log_block(path: Path, header_line: str, body_lines: List[str], saved_msg: str) -> None:
    """Write one session block; if path exists, prepend the same separator block as run_time_file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    append = path.exists()
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        if append:
            f.write("\n")
            f.write("# " + "=" * 58 + "\n")
            f.write(f"# appended {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("# " + "=" * 58 + "\n\n")
        f.write(header_line + "\n")
        for line in body_lines:
            f.write(line + "\n")
    print(saved_msg)


def run_sampling_loop(
    cfg_paths_sorted: List[Path],
    output_dir: Path,
    num_mols: int,
    get_pep_len: Callable[[Path], int],
    run_sampling_for_cfg: Callable[[Path], None],
    failures: Optional[List[str]] = None,
    stop_on_error: bool = False,
    run_time_file: Optional[Path] = None,
    failure_log_file: Optional[Path] = None,
    log_prefix: str = "[Step2]",
) -> Dict[str, object]:
    """
    Shared Step2 loop for docking sampling.

    Behavior:
    1) Iterate over cfg_paths_sorted.
    2) Skip tasks with pep_len >= 25.
    3) Skip complete tasks judged by gen_info.csv row count >= num_mols.
    4) Clean incomplete outputs, run sampling, and record per-task runtime.
    5) Export runtime records when run_time_file is provided.
    6) Export Step2-only failure lines when failure_log_file is provided (append rules same as run_time_file).
    """
    if failures is None:
        failures = []

    failures_start = len(failures)
    skipped_done = 0
    skipped_long_pep = 0
    run_time_records: List[Tuple[str, float]] = []

    for cfg_path in cfg_paths_sorted:
        sample_name = cfg_path.stem
        sample_outdir = output_dir / sample_name
        done_flag = sample_outdir / "gen_info.csv"

        try:
            pep_len = int(get_pep_len(cfg_path))
            if pep_len > 25:
                skipped_long_pep += 1
                print(f"{log_prefix} SKIP {sample_name}: pep_len={pep_len} > 25")
                continue

            if done_flag.exists():
                line_count = count_non_empty_data_rows(done_flag)
                if line_count >= num_mols:
                    skipped_done += 1
                    print(
                        f"{log_prefix} SKIP {sample_name}: already finished "
                        f"({done_flag}, rows={line_count}, num_mols={num_mols})"
                    )
                    continue
                print(
                    f"{log_prefix} CLEAN {sample_name}: incomplete gen_info.csv "
                    f"({done_flag}, rows={line_count}, num_mols={num_mols})"
                )

            if sample_outdir.exists():
                print(f"{log_prefix} CLEAN {sample_name}: incomplete outputs in {sample_outdir}")
                clear_directory_contents(sample_outdir)

            t0 = time.perf_counter()
            run_sampling_for_cfg(cfg_path)
            elapsed = time.perf_counter() - t0
            run_time_records.append((sample_name, elapsed))
            print(f"{log_prefix} OK {sample_name}")
        except Exception as exc:
            msg = f"{log_prefix} {sample_name}: {exc}"
            failures.append(msg)
            print(f"[FAILED] {msg}")
            if stop_on_error:
                break

    if run_time_file is not None:
        run_rows = [f"{name}\t{secs:.3f}" for name, secs in run_time_records]
        _write_session_log_block(
            run_time_file,
            "sample_name\trun_time_seconds",
            run_rows,
            f"[OK] run time saved to: {run_time_file}",
        )

    if failure_log_file is not None:
        session_failures = failures[failures_start:]
        fail_rows = [m.replace("\r", " ").replace("\n", " ") for m in session_failures]
        _write_session_log_block(
            failure_log_file,
            "failure_message",
            fail_rows,
            f"[OK] failure log saved to: {failure_log_file}",
        )

    return {
        "skipped_done": skipped_done,
        "skipped_long_pep": skipped_long_pep,
        "run_time_records": run_time_records,
        "failures": failures,
    }
