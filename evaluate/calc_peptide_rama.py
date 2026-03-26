from __future__ import division, print_function
import math
import os
import sys
import argparse
import numpy as np
from Bio import PDB
from multiprocessing import Pool
from tqdm import tqdm
import gzip
import shutil
from functools import partial
import csv

RAMA_PREF_VALUES = None
TARGET_CHAIN_ID = None

# change path: 'data' to 'data_rama'
RAMA_PREFERENCES = {
    "General": {
        "file": os.path.join('data_rama', 'pref_general.data'),
        "bounds": [0, 0.0005, 0.02, 1],
    },
    "GLY": {
        "file": os.path.join('data_rama', 'pref_glycine.data'),
        "bounds": [0, 0.002, 0.02, 1],
    },
    "PRO": {
        "file": os.path.join('data_rama', 'pref_proline.data'),
        "bounds": [0, 0.002, 0.02, 1],
    },
    "PRE-PRO": {
        "file": os.path.join('data_rama', 'pref_preproline.data'),
        "bounds": [0, 0.002, 0.02, 1],
    }
}


def _cache_RAMA_PREF_VALUES():
    global RAMA_PREF_VALUES 
    RAMA_PREF_VALUES = {}
    for key, val in RAMA_PREFERENCES.items():
        RAMA_PREF_VALUES[key] = np.full((360, 360), 0, dtype=np.float64)
        with open(val["file"]) as fn:
            for line in fn:
                if line.startswith("#"):
                    continue
                else:
                    x = int(float(line.split()[1]))
                    y = int(float(line.split()[0]))
                    RAMA_PREF_VALUES[key][x + 180][y + 180] \
                        = RAMA_PREF_VALUES[key][x + 179][y + 179] \
                        = RAMA_PREF_VALUES[key][x + 179][y + 180] \
                        = RAMA_PREF_VALUES[key][x + 180][y + 179] \
                        = float(line.split()[2])
    return RAMA_PREF_VALUES


def process_one(file_name):

    global RAMA_PREF_VALUES, TARGET_CHAIN_ID

    if RAMA_PREF_VALUES is None:
        RAMA_PREF_VALUES = _cache_RAMA_PREF_VALUES()
        
    tmp_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp_rama')
    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)


    if file_name.endswith('.pdb.gz'):
        tmp_pdb_file = os.path.join(tmp_folder, os.path.basename(file_name).replace('.pdb.gz', '.pdb'))
        with gzip.open(file_name, 'rb') as f_in:
            with open(tmp_pdb_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        file_to_process = tmp_pdb_file
    else:
        file_to_process = file_name

    accept_count = 0
    favoured_count = 0
    total_count = 0
    accept_count_nc = 0
    favoured_count_nc = 0
    total_count_nc = 0

    structure = PDB.PDBParser().get_structure('input_structure', file_to_process)
    first_model = next(structure.get_models(), None)
    if first_model is not None:
        chains = list(first_model.get_chains())
        if TARGET_CHAIN_ID is None:
            target_chain = chains[0] if chains else None  # default: only process the first chain
        else:
            target_chain = next((chain for chain in chains if chain.id == TARGET_CHAIN_ID), None)

        if target_chain is not None:
            chain = target_chain
            polypeptides = PDB.PPBuilder().build_peptides(chain) # ignore HETATM
            for poly_index, poly in enumerate(polypeptides): # iterate each polypeptide
                phi_psi = poly.get_phi_psi_list() # get phi, psi angles for each residue
                poly_len = len(poly)
                nc_indices_to_check = [1, 2, poly_len - 1, poly_len - 2] # terminal residues
                indices_to_check = range(len(poly))

                for res_index in indices_to_check: # check each residue
                    residue = poly[res_index]
                    res_name = "{}".format(residue.resname)
                    phi, psi = phi_psi[res_index]
                    if phi and psi: # 不考虑Peptide首尾的residue, 即phi或psi不为None
                        if str(poly[res_index + 1].resname) == "PRO" if res_index < poly_len - 1 else False: # check if the next residue is a PRO residue
                            aa_type = "PRE-PRO"
                        elif res_name == "PRO":
                            aa_type = "PRO"
                        elif res_name == "GLY":
                            aa_type = "GLY"
                        else:
                            aa_type = "General"
                        phi_deg = math.degrees(phi) # radian to degrees, 弧度换算成角度
                        psi_deg = math.degrees(psi)
                        acceptable = RAMA_PREF_VALUES[aa_type][int(psi_deg) + 180][int(phi_deg) + 180] >= \
                                     RAMA_PREFERENCES[aa_type]["bounds"][1]
                        favoured = RAMA_PREF_VALUES[aa_type][int(psi_deg) + 180][int(phi_deg) + 180] >= \
                                     RAMA_PREFERENCES[aa_type]["bounds"][2]
                        total_count += 1
                        if acceptable:
                            accept_count += 1
                        if favoured:
                            favoured_count += 1

                for res_index in nc_indices_to_check: # for res_index in indices_to_check: # 与上一个for循环重复, 疑似写错, 改为nc_indices_to_check
                    if res_index < 0 or res_index >= poly_len:
                        continue
                    residue = poly[res_index]
                    res_name = "{}".format(residue.resname)
                    phi, psi = phi_psi[res_index]
                    if phi and psi:
                        if str(poly[res_index + 1].resname) == "PRO" if res_index < poly_len - 1 else False:
                            aa_type = "PRE-PRO"
                        elif res_name == "PRO":
                            aa_type = "PRO"
                        elif res_name == "GLY":
                            aa_type = "GLY"
                        else:
                            aa_type = "General"
                        phi_deg = math.degrees(phi)
                        psi_deg = math.degrees(psi)
                        acceptable = RAMA_PREF_VALUES[aa_type][int(psi_deg) + 180][int(phi_deg) + 180] >= \
                                     RAMA_PREFERENCES[aa_type]["bounds"][1]
                        favoured = RAMA_PREF_VALUES[aa_type][int(psi_deg) + 180][int(phi_deg) + 180] >= \
                                     RAMA_PREFERENCES[aa_type]["bounds"][2]
                        total_count_nc += 1
                        if acceptable:
                            accept_count_nc += 1
                        if favoured:
                            favoured_count_nc += 1

    if file_name.endswith('.pdb.gz'):
        os.remove(tmp_pdb_file)
    return os.path.basename(file_name), total_count, accept_count, favoured_count, total_count_nc, accept_count_nc, favoured_count_nc


def callback(result, tbar, csvfile):
    file_name, total_count, accept_count, favoured_count, total_count_nc, accept_count_nc, favoured_count_nc = result
    accept_rate = accept_count / total_count if total_count > 0 else 0
    accept_rate_nc = accept_count_nc / total_count_nc if total_count_nc > 0 else 0
    favoured_rate = favoured_count / total_count if total_count > 0 else 0
    favoured_rate_nc = favoured_count_nc / total_count_nc if total_count_nc > 0 else 0

    writer = csv.DictWriter(csvfile, fieldnames=['input', 'total_angle_numbers', 'total_accept', 'accept_rate', 'total_favoured', 'favoured_rate', 'total_terminal_angle_numbers',
                                                'total_accept_nc', 'accept_rate_nc', 'total_favoured_nc', 'favoured_rate_nc'])
    row = {
        'input': file_name,
        'total_angle_numbers': total_count,
        'total_accept': accept_count,
        'accept_rate': accept_rate,
        'total_favoured': favoured_count,
        'favoured_rate': favoured_rate,
        'total_terminal_angle_numbers': total_count_nc,
        'total_accept_nc': accept_count_nc,
        'accept_rate_nc': accept_rate_nc,
        'total_favoured_nc': favoured_count_nc,
        'favoured_rate_nc': favoured_rate_nc
    }
    writer.writerow(row)

    tbar.update(1)

def write_rama_stat(pdb_files, rama_path, cores):
    csvfile = open(rama_path, 'w', newline='')
    fieldnames = ['filename', 'total_angle_numbers', 'total_accept', 'accept_rate', 'total_favoured', 'favoured_rate', 'total_terminal_angle_numbers',
                'total_accept_nc', 'accept_rate_nc', 'total_favoured_nc', 'favoured_rate_nc']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    tbar = tqdm(total=len(pdb_files))
    partial_callback = partial(callback, tbar=tbar, csvfile=csvfile)
    pool = Pool(processes=cores)
    for pdb in pdb_files:
        pool.apply_async(func=process_one, args=(pdb,), callback=partial_callback)
    pool.close()
    pool.join()

    csvfile.close()
    print(f"RAMA statistics saved to {rama_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process PDB files with multi - processing.')
    parser = argparse.ArgumentParser(description="Compute aa recovery rate for peptide inverse-folding results.")
    parser.add_argument('--exp_name', type=str, default='msel_base_fixendresbb')
    parser.add_argument('--result_root', type=str, default='./outputs_paper/pepinv_pepbdb')

    # parser.add_argument('-i', '--input', type=str, help='Path to the file containing PDB file paths.')
    parser.add_argument('-c', '--cores', type=int, default=1, help='Number of CPU cores to use.')
    parser.add_argument('--chain_id', type=str, default=None, help='Chain ID to process. If not set, only the first chain is processed.')
    # parser.add_argument('-o', '--output', type=str, help='Path to the output file.')
    parser.add_argument('--terminal', action='store_true', help='Only calculate angles for the terminal amino acids.')
    args = parser.parse_args()
    TARGET_CHAIN_ID = args.chain_id

    gen_path = os.path.join(args.result_root, args.exp_name)
    sdf_dir = os.path.join(gen_path, 'SDF')
    pdb_files = [
        os.path.join(sdf_dir, f)
        for f in os.listdir(sdf_dir)
        if f.endswith('.pdb') and not f.endswith('_gt.pdb')
    ]

    pdb_gt_files = [
        os.path.join(sdf_dir, f)
        for f in os.listdir(sdf_dir)
        if f.endswith('_gt.pdb')
    ]

    rama_path = os.path.join(gen_path, 'rama_stat.csv')
    rama_gt_path = os.path.join(gen_path, 'rama_stat_gt.csv')

    print(f"Calculating RAMA statistics for {len(pdb_files)} generated files...")
    write_rama_stat(pdb_files, rama_path, args.cores)
    print(f"Calculating RAMA statistics for {len(pdb_gt_files)} ground truth files...")
    write_rama_stat(pdb_gt_files, rama_gt_path, args.cores)
