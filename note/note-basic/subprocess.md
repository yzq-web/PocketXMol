subprocess

```python
import subprocess

cmd = [tmalign_bin, pep_pred_path, pep_gt_path] # default for task = 'pepdesign' (inverse folding)

if task == 'dock':
    cmd += ['-byresi', '1', '-het', '1'] # align by residue index; include hetero atoms
elif task == 'pepdesign':
    cmd += ['-het', '1'] # include hetero atoms

output = subprocess.run(cmd, capture_output=True, text=True)
if output.returncode != 0:
    raise ValueError('TM-align errored:' + output.stderr)

results = output.stdout.split('\n')
print(output.stdout)
```



```python
ligand_cmd = [
    pythonsh_path,
    prepare_ligand_path,
    "-l",
    pep_pred_path,
    "-o",
    lig_pdbqt,
    "-F" # 如果Peptide断裂, 则取最大片段
]
try:
    ligand_result = subprocess.run(
        ligand_cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
```



multiprocess

```python
from multiprocessing import Pool, cpu_count

def _process_one_vina(task):
    index = task["index"]
    data_id = task["data_id"]
    pep_path = task["pep_gen_path"]
    pro_path = task["pro_path"]
    tag = task["tag"]
    tmp_root = task["tmp_root"]
    remove_data_ids = task["remove_data_ids"]
    pythonsh_path = task["pythonsh_path"]
    prepare_ligand_path = task["prepare_ligand_path"]
    prepare_receptor_path = task["prepare_receptor_path"]
    vina_bin_path = task["vina_bin_path"]
    vina_cpu = task["vina_cpu"]
    
    ...
    
    return index, vina_score, message
    
def evaluate_vina_df(
    df_gen,
    tmp_root,
    check_repeats=0,
    remove_data_ids=None,
    pythonsh_path="",
    prepare_ligand_path="",
    prepare_receptor_path="",
    vina_bin_path="",
    vina_cpu=1,
    n_cores=1,
):

    df_gen = df_gen.copy()
    df_gen["vina_score"] = np.nan
    df_gen["error_code"] = np.nan
    df_gen.reset_index(inplace=True, drop=True)

    n_cores = max(1, int(n_cores))
    print(f"Running vina scoring with {n_cores} process(es)")

    tasks = []
    for index, line in df_gen.iterrows():
        tasks.append(
            {
                "index": index,
                "data_id": line["data_id"],
                "pep_gen_path": line["pep_gen_path"],
                "pro_path": line["pro_path"],
                "tag": line["tag"] if "tag" in df_gen.columns else np.nan,
                "tmp_root": tmp_root,
                "remove_data_ids": set(remove_data_ids),
                "pythonsh_path": pythonsh_path,
                "prepare_ligand_path": prepare_ligand_path,
                "prepare_receptor_path": prepare_receptor_path,
                "vina_bin_path": vina_bin_path,
                "vina_cpu": vina_cpu,
            }
        )

    iterator = None
    pool = None
    if n_cores == 1:
        iterator = map(_process_one_vina, tasks)
    else:
        pool = Pool(processes=n_cores)
        iterator = pool.imap_unordered(_process_one_vina, tasks)

    try:
        for index, vina_score, error_code in tqdm(iterator, total=len(tasks), desc="calc vina score"):
            df_gen.loc[index, "vina_score"] = vina_score
            df_gen.loc[index, "error_code"] = error_code
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    return df_gen
```

