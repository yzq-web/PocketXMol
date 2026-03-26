"""
Note: PATH_TMALIGN should be set to your TMalign installation path for peptide evaluation.
      Download TMalign from: https://aideepmed.com/TM-align/
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
import subprocess
from tqdm import tqdm

import sys
sys.path.append('.')
from evaluate.evaluate_mols import get_dir_from_prefix

PATH_TMALIGN = '/home/yangziqing/software/TMalign'
assert PATH_TMALIGN is not None, "Please set PATH_TMALIGN to your TMalign installation path."

def get_tmscore(pep_pred_path, pep_gt_path, task=None):
    """
    Calculate TM-score for peptide docking results
    """
    cmd = [f'{PATH_TMALIGN}/TMalign', pep_pred_path, pep_gt_path] # default for task = 'pepdesign' (inverse folding)
    
    if task == 'dock':
        cmd += ['-byresi', '1', '-het', '1'] # align by residue index; include hetero atoms
    elif task == 'pepdesign':
        cmd += ['-het', '1'] # include hetero atoms
    
    output = subprocess.run(cmd, capture_output=True, text=True)
    if output.returncode != 0:
        raise ValueError('TM-align errored:' + output.stderr)
    
    results = output.stdout.split('\n')
    print(output.stdout)

    rmsd_ca_ba = np.nan
    tm_scores = np.nan
    
    for line in results:
        if line.startswith('Aligned length') and 'RMSD' in line:
            rmsd_ca_ba = float(re.search(r"RMSD=\s*([\d.]+)", line).group(1))
        if line.startswith('TM-score') and 'Chain_2' in line:
            tm_scores = float(re.search(r"TM-score=\s*([\d.]+)", line).group(1)) # normalized by Chain_2 (native.pdb)
    return {
        'tmscore': tm_scores,
        'rmsd_ca_ba': rmsd_ca_ba
    }

def evaluate_tmscore_df(df_gen, gen_dir, gt_dir, check_repeats=10, task=None, remove_data_ids=[]):

    data_id_list = df_gen['data_id'].unique()
    print('Find %d generated mols with %d unique data_id' % (len(df_gen), len(data_id_list)))
    if check_repeats > 0:
        assert len(df_gen) / len(data_id_list) == check_repeats, f'Repeat {check_repeats} not match: {len(df_gen)}:{len(data_id_list)}'

    # # load gt mols
    gt_files = {data_id: os.path.join(gt_dir, data_id+'_pep.pdb')
                for data_id in data_id_list}
    # rec_files = {data_id: os.path.join(rec_dir, data_id+'_pro.pdb')
    #              for data_id in data_id_list}

    # # calc tm-score for each gen mol
    df_gen['tmscore'] = np.nan
    df_gen['rmsd_ca_ba'] = np.nan
    df_gen.reset_index(inplace=True, drop=True)
    for index, line in tqdm(df_gen.iterrows(), total=len(df_gen), desc='calc tm-score'):

        data_id = line['data_id']
        gen_file = os.path.join(gen_dir, line['filename'])

        if data_id in remove_data_ids:
            # print('skipped: %s' % data_id)
            continue
        
        if not os.path.exists(gen_file):
            raise ValueError('Not exist: %s' % gen_file)
        
        try:
            tmscore_dict = get_tmscore(gen_file, gt_files[data_id], task)
            df_gen.loc[index, tmscore_dict.keys()] = tmscore_dict.values()
        except Exception as e:
            print('error: %s for %s' % (e, data_id))
            continue

    return df_gen


def main():
    parser = argparse.ArgumentParser(description="Compute TM-score for peptide docking results.")
    parser.add_argument('--exp_name', type=str, default='msel_base_fixendresbb')
    parser.add_argument('--result_root', type=str, default='./outputs_paper/dock_pepbdb')
    parser.add_argument('--gt_dir', type=str, default='data/pepbdb/files/peptides')
    # parser.add_argument('--rec_dir', type=str, default='data/pepbdb/files/proteins')
    parser.add_argument('--check_repeats', type=int, default=0)
    parser.add_argument('--task', type=str, default='dock')
    parser.add_argument('--remove_data_ids', type=str, default=None)

    args = parser.parse_args()

    result_root = args.result_root
    exp_name = args.exp_name
    
    gt_dir = args.gt_dir # ground truth peptide.pdb (ligand)
    # rec_dir = args.rec_dir # protein.pdb (receptor)
    
    # # remove data_ids to avoid error
    if args.remove_data_ids is not None and args.remove_data_ids != '':
        remove_data_ids = [data_id.strip() for data_id in args.remove_data_ids.split(',')]
        print('remove_data_ids:', remove_data_ids)
    else:
        remove_data_ids = []

    # # generate dir
    gen_path = get_dir_from_prefix(result_root, exp_name)
    print('gen_path:', gen_path)
    pdb_path = os.path.join(gen_path, 'SDF') # predicted peptides.pdb
    
    # # load gen df
    df_gen = pd.read_csv(os.path.join(gen_path, 'gen_info.csv'))

    # # make tm-score df
    tmscore_path = os.path.join(gen_path, 'tmscore_pdb.csv')
    
    df_gen = evaluate_tmscore_df(df_gen, pdb_path, gt_dir, args.check_repeats, args.task, remove_data_ids)
    df_gen.to_csv(tmscore_path, index=False)
    print("Saved to", tmscore_path)


if __name__ == "__main__":
    main()
