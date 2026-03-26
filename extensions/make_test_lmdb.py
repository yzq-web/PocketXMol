import os
import sys
import argparse
from tqdm import tqdm

sys.path.append('.')
from utils.dataset import LMDBDatabase

def test_lmdb(peptide_lmdb_path):

    peptide_lmdb = LMDBDatabase(peptide_lmdb_path, readonly=False)

    for data_id in tqdm(peptide_lmdb.get_all_keys(), total=len(peptide_lmdb), desc='Updating peptide_pep_path'):
        data = peptide_lmdb[data_id]
        if data is None or "peptide_pep_path" not in data:
            continue
        # update peptide_pep_path
        pep_path = data['peptide_pep_path']
        if pep_path.split('/')[0] == 'data_train':
            new_pep_path = pep_path.replace('data_train', 'data')
        else:
            new_pep_path = pep_path
        data['peptide_pep_path'] = new_pep_path
        peptide_lmdb.add_one(data_id, data)

    peptide_lmdb.close()

def main():
    parser = argparse.ArgumentParser(description="Generate test lmdb.")
    parser.add_argument('--db_name', type=str, required=True, help="Database name (e.g., cpep)")
    args = parser.parse_args()
    
    db_name = args.db_name

    lmdb_dir = f'./data/{db_name}/lmdb'
    peptide_lmdb_path = os.path.join(lmdb_dir, 'peptide.lmdb')
    
    test_lmdb(peptide_lmdb_path)

    
if __name__ == "__main__":
    main()

