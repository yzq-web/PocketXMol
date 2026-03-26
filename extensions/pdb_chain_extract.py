import os
import argparse
import pandas as pd

STANDARD_AA = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
}

def extract_pdb_info(input_pdb, peptides_path, proteins_path, db_name, peptide_threshold=50):
    filename = os.path.basename(input_pdb)
    pdbid = filename.split('.')[0]
    
    # data struct: {chain_id: {'lines': [], 'residues': set()}}
    segments = {}
    chain_check = []
    
    with open(input_pdb, 'r') as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                # 1. extract info
                res_name = line[17:20].strip()
                element = line[76:78].strip().upper() # atom
                chain_id = line[21:22]
                res_seq = line[22:26].strip() # residue index

                # 2. Ignore repeated chain IDs
                # discard TER-separated HETATM sections as they are all non-amino acids.
                # 部分PDB末尾含有大量的HETATM, 且chain id与前面的肽段chain id相同, biopython无法识别, 因此改用文本识别
                if chain_id in chain_check:
                    continue
                
                # 3. filtering
                # remove H atoms
                if element == 'H':
                    continue
                # remove water
                if res_name in ['HOH', 'WAT']:
                    continue
                
                # 4. initialize new chain
                if chain_id not in segments:
                    segments[chain_id] = {'lines': [], 
                                          'residues': set(),
                                          'has_nonstd': False,
                                          'all_nonstd': True,
                                          'all_hetatm': True}

                # 5. judge non-standard amino acids
                if res_name in STANDARD_AA:
                    segments[chain_id]['all_nonstd'] = False
                elif res_name not in ['HOH', 'WAT']:
                    segments[chain_id]['has_nonstd'] = True

                # 6. judge if all HETATM
                if line.startswith('ATOM'):
                    segments[chain_id]['all_hetatm'] = False
                
                # 7. record chain
                segments[chain_id]['lines'].append(line)
                segments[chain_id]['residues'].add(res_seq)
            
            # End of the chain
            elif line.startswith('TER'):
                chain_check.append(chain_id)
                continue
            
            # End of the model (only extract MODEL 1)
            elif line.startswith('ENDMDL'):
                break
    
    # Save and Meta Aggregation
    os.makedirs(peptides_path, exist_ok=True)
    os.makedirs(proteins_path, exist_ok=True)
    
    pep_entries = []
    pro_entries = []
    has_nonstd = 0
    
    for chain_id, data in segments.items():

        # Remove chains: 
        # 1. all residues are non-standard residues
        # 2. all atoms are HETATM
        if data['all_nonstd'] or data['all_hetatm']:
            continue

        res_count = len(data['residues'])
        atom_count = len(data['lines'])
        has_nonstd = 1 if data['has_nonstd'] else 0
        
        is_peptide = res_count <= peptide_threshold
        
        if is_peptide:
            pep_entries.append({'id': chain_id, 'len': res_count, 'atoms': atom_count, 'has_nonstd': has_nonstd})
        else:
            pro_entries.append({'id': chain_id, 'atoms': atom_count})

        suffix = "pep" if is_peptide else "pro"
        out_path = os.path.join(peptides_path if is_peptide else proteins_path, 
                                f"{db_name}_{pdbid}_{suffix}.pdb")
        
        with open(out_path, 'w') as f:
            f.writelines(data['lines'])

    # 5. Build data_entry
    data_entry = {
        "data_id": f'{db_name}_{pdbid}',
        "pdbid": pdbid,
        "pep_chainid": ";".join([p['id'] for p in pep_entries]),
        "len_pep": ";".join(str(p['len']) for p in pep_entries),
        "n_atoms_pep": ";".join(str(p['atoms']) for p in pep_entries),
        "pro_chainid": ";".join([p['id'] for p in pro_entries]),
        "n_atoms_pro": ";".join(str(p['atoms']) for p in pro_entries),
        "nonstd_res_pep": ";".join(str(p['has_nonstd']) for p in pep_entries)
    }
        
    return data_entry

def main():
    parser = argparse.ArgumentParser(description="Process PDB files for Peptide-Protein complexes.")
    parser.add_argument('--db_name', type=str, required=True, help="Database name (e.g., cpep)")
    args = parser.parse_args()
    
    # Path definition
    db_name = args.db_name
    root = f'./data_train/{db_name}/files'
    df_path = f'./data_train/{db_name}/dfs'
    
    pdbs_path = os.path.join(root, 'pdbs')
    peptides_path = os.path.join(root, 'peptides')
    proteins_path = os.path.join(root, 'proteins')
    
    os.makedirs(df_path, exist_ok=True)
    os.makedirs(peptides_path, exist_ok=True)
    os.makedirs(proteins_path, exist_ok=True)
    
    # PDBs processing
    all_results = []
    pdb_files = [f for f in os.listdir(pdbs_path) if f.endswith('.pdb')]
    
    print(f"Starting processing {len(pdb_files)} files")
    
    for filename in pdb_files:
        input_path = os.path.join(pdbs_path, filename)
        
        try:
            entry = extract_pdb_info(input_path, peptides_path, proteins_path, db_name)
            all_results.append(entry)
            print(f"Successfully processed: {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
    # Meta output
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(df_path, "meta_uni_full.csv"), index=False)
    
    print(f"Processing complete. Meta data saved to {df_path}")

if __name__ == "__main__":
    main()