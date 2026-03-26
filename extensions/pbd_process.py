import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem

sys.path.append('.')
from utils.parser import PDBProtein, parse_pdb_peptide, parse_conf_list
from utils.data import torchify_dict

def process_peptide(df, peptides_dir, mols_dir):
    # Extract peptide sequence, filter peptide with non-continuous residue id

    bad_data_ids = []
    data_dict = {}
    num_skipped = 0
    for _, line in tqdm(df.iterrows(), total=len(df)):
        try:
            data_id = line['data_id']
            peptide_path = os.path.join(peptides_dir, data_id+'_pep.pdb')
            mol_path = os.path.join(mols_dir, data_id + '_mol.sdf')
            
            # load peptide pdb file
            pep_dict = parse_pdb_peptide(peptide_path)
            data_dict[data_id] = pep_dict

            # load peptide mol file
            mol = Chem.MolFromMolFile(mol_path)
            ligand_dict = parse_conf_list([mol], smiles=line['smiles'])
            
            # check atom consistency
            atom_mismatch(pep_dict, ligand_dict)
        except Exception as e:
            bad_data_ids.append(data_id)
            num_skipped += 1
            print('Skipping %d Num: %s' % (num_skipped, data_id))
            print(e) # 1. pdb: "residue id is not continuous", 2. pdb and sdf: "atom mismatch, delta pos"
            continue
    # print(bad_data_ids)
    return data_dict, bad_data_ids

def process_protein(df, proteins_dir):
    # Extract protein sequence

    data_dict = {}
    for _, line in tqdm(df.iterrows(), total=len(df)):
        data_id = line['data_id']
        protein_path = os.path.join(proteins_dir, data_id+'_pro.pdb')
        with open(protein_path, 'r') as f:
            pdb_block = f.read()
            protein = PDBProtein(pdb_block)
        rec_seqs = protein.get_chain_seqs(line['pro_chainid'].split(';'))
        data_dict[data_id] = {
            'rec_seqs': rec_seqs
        }
    return data_dict

def atom_mismatch(pdb_data, mol_data):
    """
    Check atom consistency between peptide.pdb and mol.sdf
    将process/process_peptide_allinone.py中的质控提前, 用于生成过滤之后的meta_uni.csv
    Input:
        pdb_data: dict, peptide data, from parse_pdb_peptide
        mol_data: dict, mol data, from parse_conf_list
    Output:
        True: atom consistency
        False: atom inconsistency
    """
    pdb_data = torchify_dict(pdb_data)
    mol_data = torchify_dict(mol_data)

    pos_ref = mol_data['pos_all_confs'][0] # 以第一个构象为reference
    assert pdb_data['pos'].shape[0] == pos_ref.shape[0], 'Num of atom mismatch'
    assert torch.allclose(pdb_data['pos'], pos_ref), 'Atom mismatch, delta pos: %s' % (pdb_data['pos'] - pos_ref).abs().max() # 判断peptide.pdb和mol.sdf的原子坐标是否一致, 相对误差 rtol=1e-05, 绝对误差 atol=1e-08

def process_meta(df, peptide_data, protein_data, bad_data_ids):
    df = df.copy()
    df['pep_seq'] = ''
    df['bad_peptide'] = False
    df['pro_seq'] = ''
    for idx, line in tqdm(df.iterrows(), total=len(df)):
        data_id = line['data_id']
        # Peptide data
        if data_id in peptide_data:
            df.loc[idx, 'pep_seq'] = peptide_data[data_id]['seq']
        if data_id in bad_data_ids:
            df.loc[idx, 'pep_seq'] = ''
            df.loc[idx, 'bad_peptide'] = True

        # Protein data
        if data_id in protein_data:
            df.loc[idx, 'pro_seq'] = protein_data[data_id]['rec_seqs']
    return df

def filter_meta(df):
    df = df.copy()
    df['len_pep'] = df['pep_seq'].fillna('').str.len() # 使用pep_seq重新计算len_pep
    mask = (
        (df['broken'] == False)
        & (df['pass_element'] == True)
        & (df['pass_bond'] == True)
        & (df['error_mol'] == False)
        & (df['bad_peptide'] == False)
        & (df['len_pep'] > 3)
        & (df['len_pep'] < 20)
    )
    print(f"Filtered {df[~mask].shape[0]} out of {df.shape[0]} entries")
    return df[mask]

def main():
    parser = argparse.ArgumentParser(description="Filter PDB files for Peptide-Protein complexes.")
    parser.add_argument('--db_name', type=str, required=True, help="Database name (e.g., cpep)")
    args = parser.parse_args()
    
    # Path definition
    db_name = args.db_name
    root = f'./data_train/{db_name}/files'
    df_dir = f'./data_train/{db_name}/dfs'
    peptides_dir = os.path.join(root, 'peptides')
    proteins_dir = os.path.join(root, 'proteins')
    mols_dir = os.path.join(root, 'mols')
    
    df = pd.read_csv(os.path.join(df_dir, "meta_uni_full.csv"))
    
    print('Processing peptide data...')
    peptide_data, bad_data_ids = process_peptide(df, peptides_dir, mols_dir)
    
    print('Processing protein data...')
    protein_data = process_protein(df, proteins_dir)
    
    print('Processing meta data...')
    df = process_meta(df, peptide_data, protein_data, bad_data_ids)
    
    # Meta output
    col_use = ['data_id', 'pdbid', 'pep_chainid', 'pro_chainid', 'pep_seq', 'pro_seq', 'smiles', 'isomeric_smiles']
    df_full = df[col_use].copy()
    # Meta filter
    df_filter = filter_meta(df)
    df_full['pass'] = df_full['data_id'].isin(df_filter['data_id']).values

    df_full.to_csv(os.path.join(df_dir, "meta_uni_full.csv"), index=False)
    df_filter.to_csv(os.path.join(df_dir, "meta_uni.csv"), index=False)

if __name__ == "__main__":
    main()



