import argparse
import os
import sys
import time
import shutil
import subprocess
import warnings
import xml.etree.ElementTree as ET
from multiprocessing import Pool, cpu_count

import pandas as pd
from tqdm import tqdm

sys.path.append(".")
from evaluate.evaluate_mols import get_dir_from_prefix
from extensions.pdb_combined_pred import combine_pred_pair

warnings.filterwarnings("ignore")

FIXED_HEADERS = [
    'Input',  
    'hydrophobic_interactions', 
    'hydrogen_bonds', 
    'water_bridges', 
    'salt_bridges', 
    'pi_stacks', 
    'pi_cation_interactions', 
    'halogen_bonds', 
    'metal_complexes'
] # plip interaction types


def find_plip_report(output_dir, base_name):
    """
    Find PLIP XML report path across different PLIP output layouts.
    适配不同版本的PLIP输出格式
    """
    candidates = [
        os.path.join(output_dir, "report.xml"),
        os.path.join(output_dir, f"{base_name}_report.xml"),
        os.path.join(output_dir, "plipreport.xml"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    if os.path.isdir(output_dir):
        for file_name in os.listdir(output_dir):
            if file_name.endswith(".xml"):
                return os.path.join(output_dir, file_name)
    return None

def count_interactions(xml_file):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        interaction_counts = {}

        interactions = root.find('bindingsite/interactions')
        if interactions is not None:
            for child in interactions:
                interaction_type = child.tag
                interaction_count = len(child.findall('./*'))
                interaction_counts[interaction_type] = interaction_count

        return interaction_counts
    except Exception as e:
        print(f"Error processing {xml_file}: {e}", file=sys.stderr)
        return {}

def standardize_interaction_types(interaction_counts):
    aliases = {
        'hydrophobic': 'hydrophobic_interactions',
        'hydrophobic_interaction': 'hydrophobic_interactions',
        'h_bond': 'hydrogen_bonds',
        'h_bonds': 'hydrogen_bonds',
        'waterbridge': 'water_bridges',
        'water_bridge': 'water_bridges',
        'saltbridge': 'salt_bridges',
        'salt_bridge': 'salt_bridges',
        'pistack': 'pi_stacks',
        'pi_stack': 'pi_stacks',
        'pication': 'pi_cation_interactions',
        'pi_cation': 'pi_cation_interactions',
        'halogenbond': 'halogen_bonds',
        'halogen_bond': 'halogen_bonds',
        'metalcomplex': 'metal_complexes',
        'metal_complex': 'metal_complexes'
    }
    
    standardized = {key: 0 for key in FIXED_HEADERS[1:]}
    for key, value in interaction_counts.items():
        normalized_key = key.lower().replace('_', '').rstrip('s')
        standardized_key = aliases.get(normalized_key, key)
        if standardized_key in standardized:
            standardized[standardized_key] = value
    
    return standardized


def get_peptide_chain_ids(peptide_path):
    chain_ids = set()
    with open(peptide_path, "r") as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")) and len(line) >= 22:
                chain_ids.add(line[21].strip() or " ")

    if not chain_ids:
        raise ValueError(f"No ATOM/HETATM records found in peptide: {peptide_path}")

    return sorted(chain_ids)


def validate_peptide_chain_ids(chain_ids, peptide_path):
    if not chain_ids:
        raise ValueError(f"No peptide chain IDs found in {peptide_path}")
    if " " in chain_ids:
        raise ValueError(f"Peptide chain ID contains blank chain in {peptide_path}: {chain_ids}")


def merge_protein_peptide_pdb(protein_path, peptide_path, output_path):
    """
    Merge protein and peptide PDB files into a single PDB file.
    """
    combine_pred_pair(protein_path, peptide_path, output_path)


def run_plip_on_combined_pdb(combined_pdb_path, plip_tmp_dir, peptide_chain_id):
    file_name = os.path.basename(combined_pdb_path)
    base_name = os.path.splitext(file_name)[0]
    work_dir = os.path.join(plip_tmp_dir, f"{base_name}__work")
    os.makedirs(work_dir, exist_ok=True)

    command = ["plip", "-f", combined_pdb_path, "--peptides", peptide_chain_id, "-x", "-o", work_dir, "--silent"]
    completed = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if completed.returncode != 0:
        raise RuntimeError(f"PLIP command failed for {combined_pdb_path}")

    report_xml = find_plip_report(work_dir, base_name)
    if not report_xml:
        raise FileNotFoundError(f"Cannot find PLIP XML report in {work_dir}")

    saved_xml = os.path.join(plip_tmp_dir, f"{base_name}_report.xml")
    shutil.copyfile(report_xml, saved_xml)
    shutil.rmtree(work_dir, ignore_errors=True)
    return saved_xml


def _process_single_row(task):
    index = task["index"]
    data_id = task["data_id"]
    filename = task["filename"]
    tag = task["tag"]
    peptide_dir = task["peptide_dir"]
    protein_dir = task["protein_dir"]
    combined_dir = task["combined_dir"]
    plip_tmp_dir = task["plip_tmp_dir"]
    remove_data_ids = task["remove_data_ids"]

    result = {
        "index": index,
        "combined_pdb": None,
        "plip_xml": None,
        "pep_chain_id": None,
        "error_code": None,
    }
    for header in FIXED_HEADERS[1:]:
        result[header] = 0
    result["num_interactions"] = 0

    if tag == tag:
        return result
    if data_id in remove_data_ids:
        return result

    pep_pred_path = os.path.join(peptide_dir, filename)
    rec_path = os.path.join(protein_dir, f"{data_id}_pro.pdb")
    if not os.path.exists(pep_pred_path):
        result["error_code"] = f"Missing pred peptide: {pep_pred_path}"
        return result
    if not os.path.exists(rec_path):
        result["error_code"] = f"Missing receptor: {rec_path}"
        return result

    filename_stem, _ = os.path.splitext(filename)
    combined_pdb_path = os.path.join(combined_dir, f"{filename_stem}_combined.pdb")
    try:
        # pep_chain_ids = get_peptide_chain_ids(pep_pred_path)
        # validate_peptide_chain_ids(pep_chain_ids, pep_pred_path)
        # result["pep_chain_id"] = ";".join(pep_chain_ids)
        result["pep_chain_id"] = 'L'

        merge_protein_peptide_pdb(rec_path, pep_pred_path, combined_pdb_path)
        report_xml = run_plip_on_combined_pdb(combined_pdb_path, plip_tmp_dir, 'L') # only one peptide chain is supported
        standardized_counts = standardize_interaction_types(count_interactions(report_xml))
        for header in FIXED_HEADERS[1:]:
            result[header] = int(standardized_counts.get(header, 0))
        result["num_interactions"] = int(sum(standardized_counts.values()))
        result["combined_pdb"] = combined_pdb_path
        result["plip_xml"] = report_xml
    except Exception as e:
        result["error_code"] = str(e)

    return result


def evaluate_plip_df(
    df_gen,
    peptide_dir,
    protein_dir,
    combined_dir,
    plip_tmp_dir,
    check_repeats=0,
    remove_data_ids=None,
    n_cores=1,
):
    if remove_data_ids is None:
        remove_data_ids = []

    data_id_list = df_gen["data_id"].unique()
    print("Find %d generated mols with %d unique data_id" % (len(df_gen), len(data_id_list)))
    if check_repeats > 0:
        assert len(df_gen) / len(data_id_list) == check_repeats, (
            f"Repeat {check_repeats} not match: {len(df_gen)}:{len(data_id_list)}"
        )

    df_gen = df_gen.copy()
    for header in FIXED_HEADERS[1:]:
        df_gen[header] = 0
    df_gen["num_interactions"] = 0
    df_gen["combined_pdb"] = None
    df_gen["plip_xml"] = None
    df_gen["pep_chain_id"] = None
    df_gen["error_code"] = None
    df_gen.reset_index(inplace=True, drop=True)

    if n_cores == -1:
        n_cores = max(1, cpu_count() - 1)
    n_cores = max(1, int(n_cores))
    print(f"Running PLIP scoring with {n_cores} process(es)")

    tasks = []
    for index, line in df_gen.iterrows():
        tasks.append(
            {
                "index": int(index),
                "data_id": line["data_id"],
                "filename": line["filename"],
                "tag": line["tag"] if "tag" in df_gen.columns else float("nan"),
                "peptide_dir": peptide_dir,
                "protein_dir": protein_dir,
                "combined_dir": combined_dir,
                "plip_tmp_dir": plip_tmp_dir,
                "remove_data_ids": set(remove_data_ids),
            }
        )

    iterator = None
    pool = None
    if n_cores == 1:
        iterator = map(_process_single_row, tasks)
    else:
        pool = Pool(processes=n_cores)
        iterator = pool.imap_unordered(_process_single_row, tasks)

    try:
        for row_result in tqdm(iterator, total=len(tasks), desc="calc plip interactions"):
            idx = row_result.pop("index")
            for key, val in row_result.items():
                df_gen.loc[idx, key] = val
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    return df_gen


def evaluate_plip_df_gt(
    df_gen,
    peptide_dir,
    protein_dir,
    combined_dir,
    plip_tmp_dir,
    remove_data_ids=None,
    n_cores=1,
):
    # keep one sample per data_id for GT scoring
    df_gt = df_gen.drop_duplicates(subset=["data_id"], keep="first").copy()

    def _to_gt_filename(filename):
        basename, ext = os.path.splitext(filename)
        if ext == "":
            ext = ".pdb"
        return f"{basename}_gt{ext}"

    df_gt["filename"] = df_gt["filename"].map(_to_gt_filename)
    df_gt["tag"] = float("nan")

    return evaluate_plip_df(
        df_gen=df_gt,
        peptide_dir=peptide_dir,
        protein_dir=protein_dir,
        combined_dir=combined_dir,
        plip_tmp_dir=plip_tmp_dir,
        check_repeats=0,
        remove_data_ids=remove_data_ids,
        n_cores=n_cores,
    )


def main():
    parser = argparse.ArgumentParser(description="Compute PLIP interaction counts for peptide docking results.")
    parser.add_argument("--exp_name", type=str, default="msel_base_fixendresbb")
    parser.add_argument("--result_root", type=str, default="./outputs_paper/dock_pepbdb")
    # Keep arg name aligned with calc_peptide_ca_rmsd.py:
    # here gt_dir is used as protein(receptor) pdb directory.
    parser.add_argument("--gt_dir", type=str, default="data/pepbdb/files/proteins")
    parser.add_argument("--check_repeats", type=int, default=0)
    parser.add_argument("--remove_data_ids", type=str, default=None)
    parser.add_argument("--n_cores", type=int, default=1, help="Number of parallel processes (-1: all cores minus 1)")
    args = parser.parse_args()

    result_root = args.result_root
    exp_name = args.exp_name
    protein_dir = args.gt_dir

    if args.remove_data_ids is not None and args.remove_data_ids != "":
        remove_data_ids = [data_id.strip() for data_id in args.remove_data_ids.split(",")]
        print("remove_data_ids:", remove_data_ids)
    else:
        remove_data_ids = []

    start_time = time.time()

    gen_path = get_dir_from_prefix(result_root, exp_name)
    print("gen_path:", gen_path)

    peptide_dir = os.path.join(gen_path, "SDF")
    combined_dir = os.path.join(gen_path, "SDF_combined")
    plip_tmp_dir = os.path.join(gen_path, "plip")
    os.makedirs(combined_dir, exist_ok=True)
    os.makedirs(plip_tmp_dir, exist_ok=True)

    df_gen = pd.read_csv(os.path.join(gen_path, "gen_info.csv"))

    output_csv = os.path.join(gen_path, "plip_interactions.csv")
    if not os.path.exists(output_csv):
        df_plip = evaluate_plip_df(
            df_gen=df_gen,
            peptide_dir=peptide_dir,
            protein_dir=protein_dir,
            combined_dir=combined_dir,
            plip_tmp_dir=plip_tmp_dir,
            check_repeats=args.check_repeats,
            remove_data_ids=remove_data_ids,
            n_cores=args.n_cores,
        )
        df_plip.to_csv(output_csv, index=False)
        print("Saved to", output_csv)
    else:
        print("plip_interactions.csv already exists")

    output_gt_csv = os.path.join(gen_path, "plip_interactions_gt.csv")
    if not os.path.exists(output_gt_csv):
        df_plip_gt = evaluate_plip_df_gt(
            df_gen=df_gen[['data_id', 'filename']],
            peptide_dir=peptide_dir,
            protein_dir=protein_dir,
            combined_dir=combined_dir,
            plip_tmp_dir=plip_tmp_dir,
            remove_data_ids=remove_data_ids,
            n_cores=args.n_cores,
        )
        df_plip_gt.to_csv(output_gt_csv, index=False)
        print("Saved to", output_gt_csv)
    else:
        print("plip_interactions_gt.csv already exists")

    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
