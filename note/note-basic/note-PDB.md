## PDB文件信息提取

- Ref: `PocketXMol/utils/parser.py`

```python
import numpy as np
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from utils.parser import PDBProtein

def parse_pdb_peptide(pdb_path):
    """
    Parse pdb file to peptide feature dictionary

    Input: pdb_path
    - pdb_path: str, path to pdb file
    
    Output: pep_feature_dict
    - pos: np.array, atom coordinates
    - atom_name: list, atom names
    - is_backbone: np.array, whether the atom is a backbone atom
    - res_id: np.array, residue index in PBD
    - atom_to_aa_type: np.array, atom to amino acid type
    - res_index: np.array, residue index (0 - len(peptide))
    - seq: str, peptide sequence
    - pep_len: int, peptide length
    - pep_path: str, peptide path
    """

    parser = PDBParser(PERMISSIVE=0)
    structure = parser.get_structure('structure', pdb_path)
    
    pep_feature_dict = {'pos': [], 'atom_name': [], 'is_backbone': [], 'res_id': [], 'atom_to_aa_type': []}
    for atom in structure.get_atoms():
        element = atom.element
        if element == 'H':
            continue
        element = element if len(element) == 1 else element[0] + element[1].lower() # element: H, C, N, O, etc.
        pep_feature_dict['pos'].append(atom.get_coord()) # atom pos
        pep_feature_dict['atom_name'].append(atom.get_name()) # atom name: CA, C, N, O, NH1, NH2, etc.
        res = atom.get_parent()
        pep_feature_dict['res_id'].append(res.get_id()[1]) # residue index, 保留非标准氨基酸
        resname = res.get_resname() # residue name: ARG, LYS, etc.
        aa_type = PDBProtein.AA_NAME_NUMBER[resname] if resname in PDBProtein.AA_NAME_NUMBER else len(PDBProtein.AA_NAME_NUMBER) # aa type: 0-20, 定义非标准氨基酸的number为20
        pep_feature_dict['atom_to_aa_type'].append(aa_type)
        pep_feature_dict['is_backbone'].append(
            (atom.get_name() in PDBProtein.BACKBONE_NAMES) and (aa_type < len(PDBProtein.AA_NAME_NUMBER))) # 1. atom类型为backbone原子(CA, C, N, O); 2. aa类型为20种标准氨基酸

    unique_res_id, res_index = np.unique(pep_feature_dict['res_id'], return_inverse=True) # 去重排序
    pep_feature_dict['res_index'] = res_index
    assert (np.diff(unique_res_id) == 1).all(), 'residue id is not continuous' # 相邻差分, 确保residue id连续

    pep_feature_dict = {k: np.array(v) for k, v in pep_feature_dict.items()} # list -> np.array
    
    pep_feature_dict['seq'] = ''.join([seq1(residue.get_resname()) or 'X' for residue in structure.get_residues()]) # 非标准氨基酸用X表示
    pep_feature_dict['pep_len'] = len(pep_feature_dict['seq'])
    pep_feature_dict['pep_path'] = pdb_path
    pep_feature_dict['atom_name'] = list(pep_feature_dict['atom_name']) # np.array -> list
    
    if 'X' in pep_feature_dict['seq']:
        print('Warning: X in peptide sequence:', pdb_path)
    
    return pep_feature_dict
```

