import os
import sys
import argparse
import numpy as np
import pandas as pd

def test_assemblies(df_meta_path, df_assembly_path, data_task, db_name, split='test'):
    df = pd.read_csv(df_meta_path)
    df['data_task'] = data_task
    df['db'] = db_name
    df['split'] = split
    df_assembly = df[['data_id', 'data_task', 'db', 'split']]
    df_assembly.to_csv(df_assembly_path, index=False)

    print(f"Saved {df_assembly.shape[0]} assemblies to {df_assembly_path}")

    return df_assembly

def main():
    parser = argparse.ArgumentParser(description="Generate test assemblies.")
    parser.add_argument('--db_name', type=str, required=True, help="Database name (e.g., cpep)")
    parser.add_argument('--data_task', type=str, required=True, help="Data task (e.g., dock)")
    args = parser.parse_args()
    
    data_task = args.data_task
    db_name = args.db_name

    df_dir = f'./data_train/{db_name}/dfs'
    assembly_dir = f'./data/test/assemblies/'

    df_meta_path=os.path.join(df_dir, "meta_uni.csv")
    df_assembly_path=os.path.join(assembly_dir, f"{data_task}_{db_name}.csv")
    
    test_assemblies(df_meta_path, df_assembly_path, data_task=data_task, db_name=db_name)

    
if __name__ == "__main__":
    main()



