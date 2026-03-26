import os
import argparse
import numpy as np
import pandas as pd

import sys
sys.path.append('.')
from evaluate.evaluate_mols import get_dir_from_prefix

def seq_identity(seq1, seq2):
    if len(seq1) != len(seq2):
        return np.nan
        # raise ValueError('Length of two sequences are not equal.')
    return sum((s1 == s2) and s2 != 'X' for s1, s2 in zip(seq1, seq2)) / len(seq1)

def main():
    parser = argparse.ArgumentParser(description="Compute aa recovery rate for peptide inverse-folding results.")
    parser.add_argument('--exp_name', type=str, default='msel_base_fixendresbb')
    parser.add_argument('--result_root', type=str, default='./outputs_paper/pepinv_pepbdb')
    # parser.add_argument('--gt_dir', type=str, default='data/pepbdb/files/peptides')
    # parser.add_argument('--rec_dir', type=str, default='data/pepbdb/files/proteins')

    args = parser.parse_args()

    result_root = args.result_root
    exp_name = args.exp_name
    
    # gt_dir = args.gt_dir # ground truth peptide.pdb (ligand)
    # rec_dir = args.rec_dir # protein.pdb (receptor)

    # # generate dir
    gen_path = get_dir_from_prefix(result_root, exp_name)
    print('gen_path:', gen_path)
    
    # # load sc_metrics
    df_sc_metrics = pd.read_csv(os.path.join(gen_path, 'sc_metrics.csv'))

    if 'aa_recovery_rate' not in df_sc_metrics.columns:
        print("Calculating aa recovery rate...")
        df_sc_metrics['aa_recovery_rate'] = df_sc_metrics.apply(lambda x: seq_identity(x['gt_seq'], x['aaseq']), axis=1)
        df_sc_metrics.to_csv(os.path.join(gen_path, 'sc_metrics.csv'), index=False)
        print("Saved to", os.path.join(gen_path, 'sc_metrics.csv'))
    else:
        print("aa_recovery_rate already exists")

if __name__ == "__main__":
    main()