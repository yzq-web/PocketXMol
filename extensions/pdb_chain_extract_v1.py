import os
import argparse
from Bio import PDB
import pandas as pd

def extract_pdb_info(input_pdb, peptides_path, proteins_path, db_name, peptide_threshold=50):
    """
    Processes a Peptide-Protein complex PDB file to isolate chains and extract structural metadata.
    """

    parser = PDB.PDBParser(QUIET=True)
    io = PDB.PDBIO()

    filename = os.path.basename(input_pdb)
    pdbid = filename.split('.')[0]
    structure = parser.get_structure(pdbid, input_pdb)
    # extract Model 0
    model = structure[0] # 只保留了一个model

    # Structure cleaning
    for chain in model:
        for residue in list(chain):
            # Remove H2O
            if residue.resname in ['HOH', 'WAT']:
                chain.detach_child(residue.id)
                continue
            # Remove H atmos
            for atom in list(residue): # 使用 list() 创建副本
                if atom.element.strip().upper() == 'H':
                    residue.detach_child(atom.id)
    
    # judge non-standard residue (ignore water)
    def is_nonstd(res):
        return not PDB.is_aa(res) and res.resname not in ['HOH', 'WAT']

    # 1. Pep/Pro identification and extraction
    pep_chain_objs = []
    pro_chain_objs = []
    has_nonstd = 0
    
    for chain in model:
        residues = [res for res in chain if PDB.is_aa(res)] # or is_nonstd(res) 是否计入non-standard residue
        if not residues: continue
        
        n_atoms = sum(len(res) for res in residues)
        
        if len(residues) <= peptide_threshold:
            pep_chain_objs.append(chain)
            if any(is_nonstd(res) for res in chain):
                has_nonstd = 1
        else:
            pro_chain_objs.append(chain)

    # 2. PDB chain select
    class ChainSelect(PDB.Select):
        def __init__(self, target_chains):
            self.target_chains = target_chains
        def accept_chain(self, chain):
            return 1 if chain in self.target_chains else 0

    # 3. Chain split and save
    if not os.path.exists(peptides_path):
        os.makedirs(peptides_path)
        
    if not os.path.exists(proteins_path):
        os.makedirs(proteins_path)
        
    if pep_chain_objs:
        io.set_structure(structure)
        io.save(os.path.join(peptides_path, f"{db_name}_{pdbid}_{pep_chain_objs[0].id}_pep.pdb"), ChainSelect(pep_chain_objs))
        
    if pro_chain_objs:
        io.set_structure(structure)
        io.save(os.path.join(proteins_path, f"{db_name}_{pro_chain_objs[0].id}_{pdbid}_pro.pdb"), ChainSelect(pro_chain_objs))

    # 4. Meta stat
    pep_data = []
    for c in pep_chain_objs:
        aa_residues = [r for r in c if PDB.is_aa(r)] # or is_nonstd(res) 
        pep_data.append({
            'id': c.id,
            'len': len(aa_residues),
            'atoms': sum(len(r) for r in aa_residues)
        })

    pro_data = []
    for c in pro_chain_objs:
        aa_residues = [r for r in c if PDB.is_aa(r)]
        pro_data.append({
            'id': c.id,
            'atoms': sum(len(r) for r in aa_residues)
        })

    data_entry = {
        "data_id": f'{db_name}_{pdbid}_{pep_chain_objs[0].id}',
        "pdbid": pdbid,
        "pep_chainid": ",".join([p['id'] for p in pep_data]),
        "len_pep": ",".join(str(p['len']) for p in pep_data),
        "n_atoms_pep": ",".join(str(p['atoms']) for p in pep_data),
        "pro_chainid": ",".join([p['id'] for p in pro_data]),
        "n_atoms_pro": ",".join(str(p['atoms']) for p in pro_data),
        "nonstd_res_pep": has_nonstd
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
    df.to_csv(os.path.join(df_path, "meta_uni.csv"), index=False)
    
    print(f"Processing complete. Meta data saved to {df_path}")

if __name__ == "__main__":
    main()