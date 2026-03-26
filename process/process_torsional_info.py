from itertools import product
import os
import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
import networkx as nx
import argparse
import torch

import sys
sys.path.append('.')
from utils.fragment import find_rotatable_bond_mat
from process.unmi.process_mols import get_unmi_raw_db, unmi_data_to_rdmol


def get_mol_from_data(data_dict, mols_dir=None, train_txn=None, val_txn=None, root_dir=None):
    if mols_dir is None:
        assert root_dir is not None, 'either mols_dir or root_dir should not be None'
        db = data_dict['db']
        if db in ['geom', 'qm9', 'cremp']:
            mols_dir = os.path.join(root_dir, db, 'mols')
        elif db in ['moad', 'pbdock', 'csd', 'apep', 'pepbdb', 'poseb', 'poseboff', 'bpep', 'cpep', 'peptest']:
            mols_dir = os.path.join(root_dir, db, 'files/mols')
        else:
            assert db == 'unmi', f'Unknown db {db} to get gt mols'

    data_id = data_dict['data_id']
    db_name = data_id.split('_')[0]
    
    if db_name in ['geom', 'qm9', 'cremp']:
        mol_fn = data_id + '.sdf'
    elif db_name in ['moad', 'pbdock', 'csd', 'apep', 'pepbdb', 'poseb', 'poseboff', 'bpep', 'cpep', 'peptest']:
        mol_fn = data_id + '_mol.sdf'
    elif db_name == 'unmi':
        key = data_id[10:].encode()
        if 'train' in data_id:
            data = train_txn.get(key)
        elif 'valid' in data_id:
            data = val_txn.get(key)
        data = pickle.loads(data)
        mol = unmi_data_to_rdmol(data, add_confs=True)
        return mol
    else:
        raise NotImplementedError(f'Unknown db_name: {db_name}')
    mol = Chem.MolFromMolFile(os.path.join(mols_dir, mol_fn))
    return mol


def get_torsional_info_mol(mol, bond_index, data_id=None):
    # # find rotatable bond
    if isinstance(bond_index, torch.Tensor):
        bond_index = bond_index.numpy() # shape: (2, num_bonds)

    rot_mat = find_rotatable_bond_mat(mol) # shape: (num_atoms, num_atoms), 1: rotable bond, 0: non-rotable bond
    bond_rotatable = rot_mat[bond_index[0], bond_index[1]] # shape: (num_bonds,)
    rotatable_bond_index = bond_index[:, bond_rotatable==1] # 只保留可旋转的键索引, shape: (2, num_rotatable_bonds)
    
    # # find tor_twist_pairs - 图论视角下, 按"断键"找扭转原子两侧的原子集合
    G_base = nx.from_edgelist(bond_index.T) # 构建基础图, 键索引为边, 原子为节点
    tor_twist_pairs = {}
    for rot_bond in rotatable_bond_index.T:
        if rot_bond[0] > rot_bond[1]: # note: bond is symmetric (bond 是无向的，只保留一个方向避免重复计算)
            continue
        G_break = G_base.copy()
        G_break.remove_edge(*rot_bond) # 移除可旋转的键 (atom_index_i, atom_index_j)
        connected_components = list(nx.connected_components(G_break)) # 获取连通分量component: [{node_set_1}, {node_set_2}, ...], node即为atom index
        if len(connected_components) == 2: # 只移除一个键, 因此连通分量数量为2
            component_0, component_1 = connected_components
            if rot_bond[0] in component_0: # 移除旋转键上的两个原子, 两边的component可整体转动
                component_0.remove(rot_bond[0])
                component_1.remove(rot_bond[1])
                tor_twist_pairs[rot_bond[0], rot_bond[1]] = [
                    component_0, component_1] # 存储: 旋转键两端的原子索引 -> 两边要整体转动的component原子索引集合
            else:
                component_1.remove(rot_bond[0])
                component_0.remove(rot_bond[1])
                tor_twist_pairs[rot_bond[0], rot_bond[1]] = [
                    component_1, component_0]
        else:
            raise ValueError(f'Skip: {data_id} does not have two connected components.')
        
    # # make fixed_dist - 固定距离矩阵
    # 值为1: 在任何扭转条件下, 该atom pair之间的距离都是固定的, 用于掩码 / 约束
    # way 1: initial all fixed (not right if there are multiple components)
    n_atoms = mol.GetNumAtoms()
    fixed_dist = np.ones((n_atoms, n_atoms)) # shape: (num_atoms, num_atoms), 初始化为1, 默认所有原子对之间的距离都"固定"
    for tor_edge, twisted_edges in tor_twist_pairs.items():
        for not_fixed_pair in product(twisted_edges[0], twisted_edges[1]): # product: node_set_1 * node_set_2, 笛卡尔积, 遍历所有可能的组合
            fixed_dist[not_fixed_pair[0], not_fixed_pair[1]] = 0 # 两个component之间的原子距离不"固定", 即设置为0
            fixed_dist[not_fixed_pair[1], not_fixed_pair[0]] = 0
            
    # # make nbh info - 邻阶信息
    path_mat = Chem.GetDistanceMatrix(mol) # 计算原子之间的最短路径距离矩阵, shape: (num_atoms, num_atoms)
    nbh_dict = {}
    for i in range(n_atoms):
        nbh_dict[i] = np.where(path_mat[i] == 1)[0].tolist() # 获取原子i的邻居原子索引列表, distance=1
    
    # # add mol symmetries - 分子对称性信息（自同构 / 同构映射）
    matches_list = []
    for isomeric in [False, True]:
        # matches: 找出分子对自身的所有同构映射, shape: (num_matches, num_atoms); 行索引: 第几种映射, 列索引: 原始的atom index, 元素值: 映射后的atom index
        # e.g. 对称分子 A–B–C–B–A, 原始映射为[0,1,2,3,4]; 左右翻转后为同种构型, 映射为[4,3,2,1,0]
        matches = np.array(mol.GetSubstructMatches(mol, uniquify=False, useChirality=isomeric, maxMatches=10000)) # useChirality: 是否考虑手性, False: 不考虑手性, 只看图结构; True: 严格考虑手性(分子构型完全相同)
        natural_order = np.arange(n_atoms) # shape: (num_atoms,); 原始映射: 原始的atom index == 映射后的atom index
        is_natural = (matches == natural_order).all(-1) # 判断matches是否存在原始映射, shape: (num_matches,), 值为True/False
        if is_natural.sum() == 0: # 全为False, RDKit没有采样到原始映射, 则在第0行添加原始映射
            is_natural = np.array([True] + [False] * len(matches))
            matches = np.concatenate([natural_order[None], matches], axis=0)
        inconsistent = (matches != matches[is_natural]).any(axis=0) # 哪些原子在不同映射下"会被挪动": 判断matches的每一列是否与原始映射不一致, shape: (num_atoms,), 值为True/False
        matches = matches[:, inconsistent] # 删除与原始映射不一致的列: 去掉在所有自同构映射下都不发生变化的原子列，只保留“参与对称操作”的原子
        matches_list.append(matches)
    matches_graph = matches_list[0] # 图结构同构的映射: isomeric=False，忽略手性，只看图结构对称性
    matches_isom = matches_list[1] # 分子构型同构的映射: isomeric=True，考虑手性，严格同构
    
    result = {
            'bond_rotatable': np.array(bond_rotatable, dtype=np.int64), # True: rotatable bond, False: non-rotatable bond
            'tor_twisted_pairs': tor_twist_pairs,
            'fixed_dist_torsion': fixed_dist,
            
            'tor_bond_mat': rot_mat,
            'path_mat': path_mat,
            'nbh_dict': nbh_dict,
            
            'matches_graph': matches_graph,
            'matches_iso': matches_isom,
        }
        
    return result


def get_torsional_info(df, mol_path, save_path, mols_dir):
    from utils.dataset import LMDBDatabase
    mol_lmdb = LMDBDatabase(mol_path, readonly=True) # lmdb/pocmol10.lmdb
    tor_lmdb = LMDBDatabase(save_path, readonly=False)
    
    if 'unmi' in mols_dir:
        train_txn, val_txn = get_unmi_raw_db()
    else:
        train_txn, val_txn = None, None
    
    for _, line in tqdm(df.iterrows(), total=len(df)):
        data_id = line['data_id']
        result = {}

        mol_data = mol_lmdb[data_id]
        if mol_data is None:
            print(f'Skip: {data_id} does not have mol data.')
            continue
        bond_index = mol_data['bond_index'].numpy()
        
        # # find rotatable bonds
        mol = get_mol_from_data(mol_data, mols_dir, train_txn, val_txn) # load mol from sdf
        mol = Chem.RemoveAllHs(mol)
        
        result = get_torsional_info_mol(mol, bond_index, data_id)
        
        tor_lmdb.add_one(data_id, result)
    tor_lmdb.close()
    # train_txn.close()
    # val_txn.close()


def get_db_config(db_name, save_name='torsion', root='data_train'):
    data_dir = f'{root}/{db_name}'
    df_path = os.path.join(data_dir, 'dfs/meta_uni.csv')
    mols_dir = os.path.join(data_dir, 'mols')
    mol_path = os.path.join(data_dir, 'lmdb/mols.lmdb')
    save_path = os.path.join(data_dir, f'lmdb/{save_name}.lmdb')

    if db_name in ['pbdock', 'csd']:
        df_path = df_path.replace('meta_uni.csv', 'meta_filter_w_pocket.csv')
    if db_name in ['pbdock', 'csd', 'moad', 'apep', 'bpep', 'cpep', 'peptest']:  # with pocket
        mols_dir = os.path.join(data_dir, 'files/mols')
        mol_path = os.path.join(data_dir, 'lmdb/pocmol10.lmdb')
    
    return df_path, mol_path, save_path, mols_dir
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--from_preset', type=bool, default=True)
    parser.add_argument('--db_name', type=str, default='geom')
    args = parser.parse_args()

    if args.from_preset:
        df_path, mol_path, save_path, mols_dir = get_db_config(args.db_name)
    else:
        raise NotImplementedError
    df_use = pd.read_csv(df_path)
        
    get_torsional_info(df_use, mol_path, save_path, mols_dir, )
    print('Done processing torsional info.')
    