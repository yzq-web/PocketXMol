"""
Calculate CA-RMSD (TA: target-aligned, BA: binder-aligned)

- TA: 以受体为参考叠合两个复合物, 再计算肽段 CA 的 RMSD
- BA: 以肽段为参考叠合预测肽段与 GT 肽段, 得到的 CA RMSD

"""

import os
import argparse
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, Superimposer
from tqdm import tqdm

import sys
sys.path.append('.')
from evaluate.evaluate_mols import get_dir_from_prefix

def get_ca_atoms(structure, chain_id=None):
    """
    从 Bio.PDB Structure 中取出 CA 原子坐标与列表
    每个residue中只有一个 CA 原子, 因此只需保证pred.pdb和gt.pdb的residue顺序一致即可
    """
    ca_list = []
    for model in structure:
        for chain in model:
            if chain_id is not None and chain.id != chain_id:
                continue
            for res in chain: # include: ATOM, HETATM
                if "CA" in res:
                    ca_list.append(res["CA"])
            if chain_id is not None: # chain_id = None: extract all chains
                break
        break # only extract the first model
    return ca_list


def get_ca_coords(ca_list):
    return np.array([a.get_coord() for a in ca_list])


def ca_rmsd_ba(pep_pred_path, pep_gt_path):
    """
    CA-RMSD (BA): 将pep_pred叠合至pep_gt上, 返回叠合后的CA-RMSD
    要求: pep_pred与pep_gt的residue 顺序一致: 长度相同, 且一一对应
    """
    parser = PDBParser(QUIET=True)
    s_pred = parser.get_structure("pred", pep_pred_path)
    s_gt = parser.get_structure("gt", pep_gt_path)
    ca_pred = get_ca_atoms(s_pred)
    ca_gt = get_ca_atoms(s_gt)

    assert len(ca_pred) != 0 and len(ca_gt) != 0, 'no CA atoms found in pred or gt: %d, %d' % (len(ca_pred), len(ca_gt))
    assert len(ca_pred) == len(ca_gt), 'mismatched num of CA atoms: %d, %d' % (len(ca_pred), len(ca_gt))

    sup = Superimposer()
    sup.set_atoms(ca_gt, ca_pred)
    return sup.rms


def ca_rmsd_ta(pep_pred_path, pep_gt_path):
    """
    CA-RMSD (TA): 在受体固定坐标系下, 不叠合肽段, 直接计算pep_pred与pep_gt的RMSD
    要求: pep_pred与pep_gt均在同一受体坐标系下 (docking至同一受体)
    """
    parser = PDBParser(QUIET=True)
    s_pred = parser.get_structure("pred", pep_pred_path)
    s_gt = parser.get_structure("gt", pep_gt_path)
    ca_pred = get_ca_atoms(s_pred)
    ca_gt = get_ca_atoms(s_gt)
    
    assert len(ca_pred) != 0 and len(ca_gt) != 0, 'no CA atoms found in pred or gt: %d, %d' % (len(ca_pred), len(ca_gt))
    assert len(ca_pred) == len(ca_gt), 'mismatched num of CA atoms: %d, %d' % (len(ca_pred), len(ca_gt))

    pred_coords = get_ca_coords(ca_pred)
    gt_coords = get_ca_coords(ca_gt)
    rms = np.sqrt(np.mean(np.sum((pred_coords - gt_coords) ** 2, axis=1)))
    return rms


def get_rmsd_ca(pep_pred_path, pep_gt_path):
    ba = ca_rmsd_ba(pep_pred_path, pep_gt_path)
    ta = ca_rmsd_ta(pep_pred_path, pep_gt_path)
    return {"ca_rmsd_ta": ta, "ca_rmsd_ba": ba}


def evaluate_rmsd_ca_df(df_gen, gen_dir, gt_dir, check_repeats=10, remove_data_ids=[]):

    data_id_list = df_gen['data_id'].unique()
    print('Find %d generated mols with %d unique data_id' % (len(df_gen), len(data_id_list)))
    if check_repeats > 0:
        assert len(df_gen) / len(data_id_list) == check_repeats, f'Repeat {check_repeats} not match: {len(df_gen)}:{len(data_id_list)}'

    # # load gt mols
    gt_files = {data_id: os.path.join(gt_dir, data_id+'_pep.pdb')
                for data_id in data_id_list}
    # rec_files = {data_id: os.path.join(rec_dir, data_id+'_pro.pdb')
    #             for data_id in data_id_list}
    
    df_gen['rmsd_ca_ta'] = np.nan
    df_gen['rmsd_ca_ba'] = np.nan
    df_gen['error_code'] = np.nan
    df_gen.reset_index(inplace=True, drop=True)
    for index, line in tqdm(df_gen.iterrows(), total=len(df_gen), desc='calc ca-rmsd (TA/BA)'):

        data_id = line['data_id']
        gen_file = os.path.join(gen_dir, line['filename'])

        if data_id in remove_data_ids:
            # print('skipped: %s' % data_id)
            continue

        if not os.path.exists(gen_file):
            raise ValueError('Not exist: %s' % gen_file)
        
        # rec_path = rec_files[data_id]
        try:
            rmsd_ca = get_rmsd_ca(gen_file, gt_files[data_id])
            df_gen.loc[index, 'rmsd_ca_ta'] = rmsd_ca['ca_rmsd_ta']
            df_gen.loc[index, 'rmsd_ca_ba'] = rmsd_ca['ca_rmsd_ba']
        except Exception as e:
            print('error: %s for %s' % (e, data_id))
            df_gen.loc[index, 'error_code'] = str(e)
            continue

    return df_gen


def main():
    parser = argparse.ArgumentParser(description="Compute CA-RMSD (TA/BA) for peptide docking results.")
    parser.add_argument('--exp_name', type=str, default='msel_base_fixendresbb')
    parser.add_argument('--result_root', type=str, default='./outputs_paper/dock_pepbdb')
    parser.add_argument('--gt_dir', type=str, default='data/pepbdb/files/peptides')
    # parser.add_argument('--rec_dir', type=str, default='data/pepbdb/files/proteins')
    parser.add_argument('--check_repeats', type=int, default=0)
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

    # # make rmsd df
    rmsd_ca_path = os.path.join(gen_path, 'rmsd_ca_pdb.csv')
    
    df_gen = evaluate_rmsd_ca_df(df_gen, pdb_path, gt_dir, args.check_repeats, remove_data_ids)
    df_gen.to_csv(rmsd_ca_path, index=False)
    print("Saved to", rmsd_ca_path)


if __name__ == "__main__":
    main()
