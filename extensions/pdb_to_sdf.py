import os
import sys
import argparse
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.append('.')
from process.make_mol_meta import make_meta

def pdb_to_sdf(input_pdb, output_sdf):
    """
    Converts a single-chain peptide PDB file to SDF format.
    """
    # 1. Load molecule from PDB file
    # removeHs=True removes hydrogen atoms if present in the PDB
    # sanitize=True performs standard chemical validity checks
    mol = Chem.MolFromPDBFile(input_pdb, removeHs=True, sanitize=True)
    
    if mol is None:
        print(f"Error: Could not parse file {input_pdb}. Please check the file path or format.")
        return

    # 2. Check if the molecule contains atoms
    if mol.GetNumAtoms() == 0:
        print("Warning: The loaded molecule contains no atoms.")
        return

    # 3. Write to SDF file
    # SDWriter will preserve the 3D coordinates from the PDB
    writer = Chem.SDWriter(output_sdf)
    try:
        writer.write(mol)
        print(f"Conversion successful! File saved to: {output_sdf}")
    except Exception as e:
        print(f"An error occurred during writing: {e}")
    finally:
        writer.close()

    return mol

def main():
    parser = argparse.ArgumentParser(description="Convert peptide .pdb into .sdf")
    parser.add_argument('--db_name', type=str, required=True, help="Database name (e.g., cpep)")
    args = parser.parse_args()
    
    db_name = args.db_name
    root = f'./data_train/{db_name}/files'
    df_path = f'./data_train/{db_name}/dfs'

    peptides_path = os.path.join(root, 'peptides')
    mols_path = os.path.join(root, 'mols')
    os.makedirs(mols_path, exist_ok=True)

    # Convert pdb to sdf
    peptide_files = [f for f in os.listdir(peptides_path) if f.endswith('.pdb')]
    mol_dict = {}
    for pbd_file in peptide_files:
        data_id = pbd_file.replace('_pep.pdb', '')
        sdf_name = data_id + '_mol.sdf'
        input_pdb = os.path.join(peptides_path, pbd_file)
        output_sdf = os.path.join(mols_path, sdf_name)

        try:
            mol = pdb_to_sdf(input_pdb, output_sdf)
            print(f"Successfully processed: {pbd_file}")
            mol_dict[data_id] = mol
        except Exception as e:
            print(f"Error processing {pbd_file}: {e}")

    # Make mol meta
    print("Extracting mol meta")
    result_list = make_meta(mol_dict, num_workers=8)
    df_mol = pd.DataFrame(result_list)
    df = pd.read_csv(os.path.join(df_path,'meta_uni_full.csv'))
    df = pd.merge(df, df_mol, how='inner', on='data_id')

    df.to_csv(os.path.join(df_path, "meta_uni_full.csv"), index=False)
    
    print(f"Processing complete. Meta data saved to {df_path}")


if __name__ == "__main__":
    main()