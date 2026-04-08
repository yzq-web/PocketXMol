import os
import pandas as pd
from tqdm import tqdm
from rdkit import Chem

def identify_peptide_cyclization(pdb_path, log_path):
    """
    1. 如果存在多条链，仅分析 'L' 链。
    2. 如果仅有一条链，分析该唯一链。
    """
    log_lines = []
    # 1. 加载分子
    # proximityBonding=True 会根据原子距离自动建立化学键（这对识别PDB中的环至关重要）
    # sanitize=False 避开 "Explicit valence greater than permitted" 错误
    mol = Chem.MolFromPDBFile(pdb_path, sanitize=False, removeHs=False, proximityBonding=True)
    
    if not mol:
        log_lines.append(f"无法读取文件: {pdb_path}")
        with open(log_path, "a", encoding="utf-8") as fout:
            fout.write("\n".join(log_lines) + "\n")
        return

    # 获取所有链 ID
    all_chains = sorted(list(set(
        atom.GetPDBResidueInfo().GetChainId().strip() 
        for atom in mol.GetAtoms() if atom.GetPDBResidueInfo()
    )))

    if not all_chains:
        log_lines.append(f"PDB 文件中没有残基信息: {pdb_path}")
        with open(log_path, "a", encoding="utf-8") as fout:
            fout.write("\n".join(log_lines) + "\n")
        return

    log_lines.append(f"文件分析: {pdb_path}")
    if len(all_chains) > 1:
        if 'A' in all_chains:
            target_chain = 'A'
            log_lines.append(f"检测到多条链 {all_chains}，已锁定分析目标: 'A' 链。")
        else:
            log_lines.append(f"检测到多条链 {all_chains}，但未找到 'A' 链。跳过分析。")
            with open(log_path, "a", encoding="utf-8") as fout:
                fout.write("\n".join(log_lines) + "\n")
            return
    else:
        target_chain = all_chains[0]
        log_lines.append(f"检测到单条链: '{target_chain}' 链，开始全链分析。")

    # 2. 获取target_chain的残基编号
    target_res_nums = []
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info and info.GetChainId().strip() == target_chain:
            target_res_nums.append(info.GetResidueNumber())
    
    abs_min, abs_max = min(target_res_nums), max(target_res_nums)
    log_lines.append(f"全链残基范围: {abs_min} - {abs_max}")

    # 3. 获取PDB中最小环的集合 (SSSR)
    ssr = Chem.GetSymmSSSR(mol)
    found_cycle = False

    for i, ring in enumerate(ssr):
        ring_res = set() # target_chain中成环的残基编号
        has_other_chain = False # 是否与其他链成环
        has_sulfur = False # 是否存在硫原子
        
        for idx in ring:
            atom = mol.GetAtomWithIdx(idx)
            info = atom.GetPDBResidueInfo()
            if info:
                cid = info.GetChainId().strip()
                if cid == target_chain:
                    ring_res.add(info.GetResidueNumber())
                else:
                    has_other_chain = True
            if atom.GetSymbol() == 'S':
                has_sulfur = True

        # 过滤: 必须包含target_chain原子，且涉及至少2个残基 (忽略残基内部环, 如 Proline, Phe, Tyr, Trp, His 的侧链)
        if not ring_res or len(ring_res) < 2:
            continue

        found_cycle = True
        r_min, r_max = min(ring_res), max(ring_res)
        
        log_lines.append(f"[环 #{i+1} 详情]")
        if has_other_chain:
            log_lines.append(f"  类型: 跨链连接 (Inter-chain)")
            log_lines.append(f"  范围: 涉及 {target_chain} 链的残基 {r_min}-{r_max} 与其他链")
        elif r_min == abs_min and r_max == abs_max:
            log_lines.append(f"  类型: 首尾成环 (Head-to-Tail Cyclic)")
            log_lines.append(f"  范围: {target_chain} 链全长 {r_min}-{r_max}")
        else:
            bridge = "二硫键/侧链桥" if has_sulfur else "局部环化"
            log_lines.append(f"  类型: 链内局部成环 ({bridge})")
            log_lines.append(f"  范围: {target_chain} 链残基 {r_min}-{r_max}")

    if found_cycle:
        log_lines.append(f"结论: {target_chain} 链含有环状结构(Cyclic)")
    else:
        log_lines.append(f"结论: {target_chain} 链是线性的(Linear)")

    log_lines.append('-'*50)
    with open(log_path, "a", encoding="utf-8") as fout:
        fout.write("\n".join(log_lines) + "\n")

if __name__ == "__main__":

    # # CPSea
    # cpsea_path = '/home/yangziqing/CPSea/data/CPSea_PDB/cpsea_tmp'
    # log_path = 'cyclic_check/cpsea_cyclic_check.txt'
    # if os.path.exists(log_path):
    #     os.remove(log_path)
    # for pdb_file in tqdm(os.listdir(cpsea_path)[0:1000]):
    #     pdb_path = os.path.join(cpsea_path, pdb_file)
    #     identify_peptide_cyclization(pdb_path, log_path)
    #     # result_list.append(result)

    # # cpep
    # df_meta = pd.read_csv('data_train/cpep/dfs/meta_uni.csv')
    # log_path = 'cyclic_check/cpep_cyclic_check.txt'
    # if os.path.exists(log_path):
    #     os.remove(log_path)
    # for _, line in df_meta.iterrows():
    #     data_id = line['data_id']
    #     pdb_path = f'data_train/cpep/files/peptides/{data_id}_pep.pdb'
    #     identify_peptide_cyclization(pdb_path, log_path)

    # # bpep
    # df_meta = pd.read_csv('data_train/bpep/dfs/meta_uni.csv')
    # log_path = 'cyclic_check/bpep_cyclic_check.txt'
    # if os.path.exists(log_path):
    #     os.remove(log_path)
    # for _, line in df_meta.iterrows():
    #     data_id = line['data_id']
    #     pdb_path = f'data_train/bpep/files/peptides/{data_id}_pep.pdb'
    #     identify_peptide_cyclization(pdb_path, log_path)

    # # pepbd
    # pepbdb_path = '/home/yangziqing/PocketXMol/data/pepbdb/files/peptides'
    # log_path = 'cyclic_check/pepbdb_cyclic_check.txt'
    # if os.path.exists(log_path):
    #     os.remove(log_path)
    # for pdb_file in tqdm(os.listdir(pepbdb_path)):
    #     pdb_path = os.path.join(pepbdb_path, pdb_file)
    #     identify_peptide_cyclization(pdb_path, log_path)

    # pepbdb cyclic
    pepbdb_cyclic_path = 'outputs_test/pepdesign_pepbdb/base_pxm_20260407_190131/SDF'
    log_path = 'cyclic_check/pepbdb_cyclic_gt_check.txt'
    if os.path.exists(log_path):
        os.remove(log_path)
    for pdb_file in tqdm(os.listdir(pepbdb_cyclic_path)):
        if '_gt.pdb' in pdb_file:
        # if 'pdb' in pdb_file and '_gt' not in pdb_file:
            pdb_path = os.path.join(pepbdb_cyclic_path, pdb_file)
            identify_peptide_cyclization(pdb_path, log_path)