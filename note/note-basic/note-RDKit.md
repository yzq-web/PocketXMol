## 读取sdf

```python
mol = Chem.MolFromMolFile(os.path.join(mols_dir, data_id + '_mol.sdf'))
```

## GetPeriodicTable - 元素周期表对象

`Chem.GetPeriodicTable()` 是 RDKit 中 `Chem` 模块提供的一个函数，它返回一个表示元素周期表的对象。这个对象提供了元素周期表中各种元素的相关信息，比如元素的符号、原子序数、质量等。通过这个对象，你可以方便地获取化学元素的信息。

**简单示例：**

```python
from rdkit import Chem

# 获取元素周期表对象
periodic_table = Chem.GetPeriodicTable()

# 获取元素符号为 'H' 的原子序号
atomic_number = periodic_table.GetAtomicNumber('H')
print(atomic_number)  # 输出: 1

# 获取原子序号为 1 的元素符号
element_symbol = periodic_table.GetElementSymbol(1)
print(element_symbol)  # 输出: 'H'
```

**主要功能：**

1. **获取原子符号**：可以通过原子序号获取元素符号。
2. **获取原子序号**：可以通过元素符号获取原子序号。
3. **获取元素的其他属性**：例如原子质量、价电子数、是否是金属等。

## 获取原子坐标

```python
# 返回一个list, 包含各个原子的xyz坐标
mol_pos = mol.GetConformer().GetPositions()
```

常见用途

```python
# 计算分子的几何中心
centroid = positions.mean(axis=0)

# 计算原子 i 和原子 j 之间的欧几里得距离
import numpy as np
dist = np.linalg.norm(positions[i] - positions[j], ord=2) # 默认ord=2
```

## 获取原子对象

```python
num_atoms = mol.GetNumAtoms()
num_bonds = mol.GetNumBonds()

conf = mol.GetConformer()
ele_list = []
pos_list = []
for i, atom in enumerate(mol.GetAtoms()):
    ele = atom.GetAtomicNum()
    pos = conf.GetAtomPosition(i)
    pos_list.append(list(pos))
    ele_list.append(ele)
```

## 获取键索引

- `PocketXMol/utils/parser.py`中的`parse_3d_mol`
- 被`PocketXMol/process/process_pocmol.py`调用

```python
row, col = [], []
bond_type = []
for bond in mol.GetBonds():
    b_type = int(bond.GetBondType())
    assert b_type in [1, 2, 3, 12], 'Bond can only be 1,2,3,12 bond'
    b_type = b_type if b_type != 12 else 4 # 12 -> 4, aromatic bond
    b_index = [ # bond index, 键索引, 键的起点和终点的原子索引
        bond.GetBeginAtomIdx(),
        bond.GetEndAtomIdx()
    ]
    bond_type += 2*[b_type] # 键类型, 存储两条有向边 (i,j) and (j,i)
    row += [b_index[0], b_index[1]]
    col += [b_index[1], b_index[0]]

bond_type = np.array(bond_type, dtype=np.int64)
bond_index = np.array([row, col],dtype=np.int64)

perm = (bond_index[0] * num_atoms + bond_index[1]).argsort() # 按atom index排序, 优先按row index排序(bond_index[0] * num_atoms), 再按col index排序(bond_index[1])
bond_index = bond_index[:, perm]
bond_type = bond_type[perm]
```

## Rotable bond

- `PocketXMol/utils/fragment.py`
- 被`PocketXMol/process/process_torsional_info.py`调用

```python
def find_rotatable_bond_mat(mol):
    """Find groups of contiguous rotatable bonds and return as a matrix
    from https://github.com/rdkit/rdkit/discussions/3776"""

    # rotable atom pairs, ((atom_index_1, atom_index_2), ...), shape: (num_rotatable_bonds, 2)
    rot_atom_pairs = mol.GetSubstructMatches(RotatableBondSmarts) 
    
    # rotable bond matrix, shape: (num_atoms, num_atoms), 1: rotable bond, 0: non-rotable bond
    rot_mat = np.zeros((mol.GetNumAtoms(), mol.GetNumAtoms()), dtype=int) 
    for i, j in rot_atom_pairs:
        rot_mat[i, j] = 1
        rot_mat[j, i] = 1
    return rot_mat
```

## 全原子RMSD

- `PocketXMol/evaluate/evaluate_dockpep.py`

读取mols

```python
# load mols
mol = Chem.MolFromPDBFile(peptide.pdb)
```

计算RMSD

```python
from rdkit import Chem
from rdkit.Chem import AllChem # RMSD计算需要

def get_rmsd(mol_prob, mol_gt):
    """
    Calculate the symm rmsd between two mols. to move them.
    """
    # mol_prob, mol_gt = mol_pair
    mol_prob = deepcopy(mol_prob)
    mol_gt = deepcopy(mol_gt)

    # try:
    #     # rmsd = Chem.rdMolAlign.CalcRMS(mol_prob, mol_gt, maxMatches=30000)  # NOT move mol_prob. for docking
    #     rmsd = get_rmsd_timeout(mol_prob, mol_gt)
    # except Exception as e:
    #     if isinstance(e, RuntimeError):
    #         print('matches error. Retry with direct atom mapping')
    #     elif isinstance(e, TimeoutError):
    #         print('timeout. Retry with direct atom mapping')
    #     else:
    #         raise e
    # rmsd use direct atom mapping
    assert mol_prob.GetNumAtoms() == mol_gt.GetNumAtoms(), 'mismatched num of atoms'
    assert all([mol_prob.GetAtomWithIdx(i_atom).GetSymbol() == mol_gt.GetAtomWithIdx(i_atom).GetSymbol()
                for i_atom in range(mol_prob.GetNumAtoms())]), 'mismatched atom element types'
    # try:
    #     rmsd = Chem.rdMolAlign.CalcRMS(mol_prob, mol_gt, maxMatches=30000)
    # except Exception as e:
    atom_map = [[(i, i) for i in range(mol_prob.GetNumAtoms())]] # atom index pair for alignment
    rmsd = Chem.rdMolAlign.CalcRMS(mol_prob, mol_gt, map=atom_map)
    return rmsd
```

