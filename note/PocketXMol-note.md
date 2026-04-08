# Data Method

**Data Filtering**

We removed data that violated any of the following criteria: 

1. successful loading by RDKit;
2. inclusion of only pre-defined atom types including C, N, O, F, P, S, Cl, B, Br, I, and Se;
3. being a complete molecule;
4. For small molecules, the count of heavy atoms should be within the range of 4 to 122;
5. For peptides, the length should be shorter than 15 and the count of heavy atoms should not exceed 150.

After filtering for individual datasets, we eliminated duplicate entries from each dataset to compile a comprehensive dataset.

**Data Processing**

1. Structure processing
   1. Use RDKit to process small molecules and peptides. 
   2. Remove hydrogen atoms.
   3. Collected atom types, bond types, and 3D conformations (atom coordinates).
   4. If multiple conformations existed, all conformations were reserved, from which one conformation was sampled at the training step.
2. Structure annotation
   1. Use RDKit to annotate the rotatable bonds
   2. Use the BRICS and MMPA algorithms to calculate the decompositions
3. Peptide annotation
   1. Use BioPython to annotate the backbone atoms and side-chain atoms
4. Pocket processing
   1. We parsed the protein receptor files and defined the pocket as the amino acids whose distances with any atom in the molecule were less than 10A.
   2. The pocket was defined by a set of atoms, each characterized by its coordinates, atom type, parent amino acid, and a binary variable to indicate it is in the backbone.

The generated peptide sequences were annotated by Open Babel based on side-chain atoms.

# Data Filtering

1. PDB数据下载

```Bash
python extensions/pdb_download.py 
```

1. 对原始PDB数据进行处理，整理成PocketXMol中`apep` / `pepbdb`数据集的数据格式

```Bash
data_train/
└── apep/
    ├── dfs/
    │   └── meta_uni.csv
    └── files/                   
        ├── proteins/            # pdb of protein (receptor)
        ├── mols/                # sdf of peptides
        └── peptides/            # pdb of peptides (ligand)
```

1. PDB预处理：`extensions/pdb_chain_extract.py`
   1. Remove
      1. 去除PDB中的氢原子
      2. 去除PDB中的水分子
      3. 去除所有原子类型均为`HETATM`的chain（其中`cpep`中8f0z.pdb的Peptide全都为`HETATM`，因此被过滤清除）
      4. 去除所有residue均不是20种标准氨基酸的chain
   2. Keep
      1. 保留了peptide和protein中缺失部分residue的chain（residue id 不连续）
      2. 保留了peptide和protein中的`HETATM`
      3. 部分PDB具有多种构象（含有多个`MODEL`，例如），只保留了第1个MODEL
   3. 识别PDB中的 peptide chain 和 protein chain，分别存储成2个PDB文件
      1. 根据长度区分 peptide 和 protein（Default：residue count < 50）
   4. 输出相关的meta信息
   5. Limitation
      1. `pdb_chain_extract.py`默认一个pdb中只含有一个Peptide和一个Protein，目前可以满足对`bpep`和`cpep`的分析需求
      2. 后续如需处理一个pdb中含有多个Peptide和Protein的样本，再通过contact_res进行完善
2. sdf文件处理：`extensions/pdb_to_sdf.py`
   1. 根据peptide的pdb生成sdf文件
   2. 根据sdf文件生成mol相关的meta（Source：`process/make_mol_meta.py`）
3. PDB数据质控：`extensions/pbd_process.py`
   1. 功能：
      - 将Data Processing中的部分质控前置
      - 输出多肽数据集csv文档：meta_uni_full.csv
      - 输出过滤后的多肽数据集csv文档：meta_uni.csv
        - 过滤条件
          - 多肽长度：**3 < pep_len < 20**
          - 其他质控：broken, pass_element, pass_bond, error_mol, bad_peptide
        - 数据统计
          - `bpep`：过滤33个，保留 **102**个peptide-protein complex
          - `cpep`：过滤5个，保留 **29** 个cyclic peptide-protein complex
   2. Reference：`PocketXMol/utils/parser.py`
      - 处理peptide.pdb：`parse_pdb_peptide(pdb_path)`，调用于`process/process_peptide_allinone.py`
      - 处理peptide.sdf：`parse_conf_list(conf_list, smiles=None)`，调用于`process/process_pocmol.py`
      - 处理protein.pdb：`PDBProtein`，调用于`process/extract_pockets.py`

- **注意个别特殊情况**
  - **bpep_8iqm**（见3.16工作记录的Log & Debug）
    - peptide.pdb中的第39位CYS存在两个结果：ACYS和BCYS
    - RDKit把.pdb转换成.sdf时，只识别了其中的ACYS，没有保留BCYS，导致peptide的.pdb和.sdf不一致
    - **在运行****`python extensions/pdb_chain_extract.py --db_name bpep`****之后，手动将bpep_8iqm_pep.pdb中的BCYS删除**
    - **待更新：****`meta_uni.csv`**
      - atom_num是按.pdb中ATOM的行数进行统计，修改bpep_8iqm_pep.pdb后未进行更新
- **完整的Data Filtering流程**

```Bash
# bpep
python extensions/pdb_download.py 
python extensions/pdb_chain_extract.py --db_name bpep # 注意修改个例：bpep_8iqm
python extensions/pdb_to_sdf.py --db_name bpep
python extensions/pbd_process.py --db_name bpep

# PepMerge
python ~/PocketXMol/extensions/convert_pepmerge_to_bpep.py --overwrite # 转换成PocketXMol中的PepBDB,bpep数据结构
python extensions/pdb_chain_extract.py --db_name pepmerge
python extensions/pdb_to_sdf.py --db_name pepmerge --from_meta
python extensions/pbd_process.py --db_name pepmerge --no_len_filter
```

# Data Processing

## apep

**代码核对及修改**

- 经检查，`apep`预处理相关的代码可以直接应用于`bpep`和`cpep`数据集的预处理，只需要做以下调整，添加`bpep`和`cpep`：

1. `process/extract_pockets.py`和`process/process_pocmol.py`：将`if db_name == 'apep'`改为：`if db_name in ['apep', 'bpep', 'cpep']`
2. `process_torsional_info.py`：向`get_mol_from_data`和`get_db_config`中的`db`和`db_name`添加相关数据集：`'bpep', 'cpep'`

**`apep`** **数据预处理流程**

- bpep和cpep：修改--db_name即可运行

```Bash
# apep
python process/extract_pockets.py --db_name apep
python process/process_pocmol.py --db_name apep
python process/process_peptide_allinone.py --db_name apep
python process/process_torsional_info.py --db_name apep
python process/process_decompose_info.py --db_name apep
```

**Explaination**

```Bash
python process/extract_pockets.py --db_name apep
- 从protein pdb里按ligand (这里为peptide) 周围半径 (r=10) Å 选残基，截出 pocket，并写成 pocket 的 PDB 文件
- Input
  - meta data (使用 data_id 列)
  - protein pdb
  - ligand sdf (peptide)
- Output
  - pocket pdb
  - pocket meta

python process/process_pocmol.py --db_name apep
- 把"pocket + 配体分子sdf"打包成模型训练用的 PocketMolData，写成 pocmol10.lmdb（这是 APEP 的基础 LMDB）
- Input
  - meta data (使用 data_id, pdbid, smiles 列)
  - pocket pdb
  - ligand sdf (peptide)
- Output
  - PocketMolData LMDB (data_train/apep/lmdb/pocmol10.lmdb)

python process/process_peptide_allinone.py --db_name apep
- 处理peptide的 PDB（残基/原子名/是否主链/序列等），将其写成 peptide.lmdb。并且强制检查：肽 PDB 的原子数与坐标要和 pocmol10.lmdb 里配体的第一构象完全一致（否则跳过）。
- Input
  - meta data (使用 data_id 列)
  - ligand pdb (peptide)
  - PocketMolData LMDB (作为reference)
- Output
  - peptide LMDB (包含peptide的特征)

python process/process_torsional_info.py --db_name apep
- 为每个样本计算"可旋转键/扭转相关的结构约束与对称匹配"等，将其写成torsion.lmdb（按 data_id 对齐）
- Input
  - meta data (使用 data_id 列)
  - PocketMolData LMDB
  - ligand sdf (真实的分子文件, 用于判断哪些键可旋转等)
- Output
  - torsion LMDB (包含扭转信息)

python process/process_decompose_info.py --db_name apep
- 对分子做分解（目前默认做 BRICS 和 MMPA 两类分解），将其写成decom.lmdb（按 data_id 对齐）
- Input
  - meta data (使用 data_id 列)
  - PocketMolData LMDB
  - ligand sdf
- Output
  - decom LMDB (包含分解信息)
```

## PepBDB

**`pepbdb`** **数据预处理流程**

- bpep和cpep：修改--db_name即可运行

```Bash
# pepbdb
python process/process_pocmol_allinone.py --db_name pepbdb
python process/process_peptide_allinone.py --db_name pepbdb

# apep
python process/extract_pockets.py --db_name apep
python process/process_pocmol.py --db_name apep
python process/process_peptide_allinone.py --db_name apep
python process/process_torsional_info.py --db_name apep
python process/process_decompose_info.py --db_name apep
```

**Data Processing比较：**`pepbdb` vs `apep`

1. `pepbdb`与`apep`的Input数据类型相同，Data Processing流程稍有不同
   1. `apep`：分步处理extract_pocket, pocmol, torsional, decompose
   2. `pepbdb`：将`apep`的上述流程命名为不同的modes，在`process_pocmol_allinone.py`中依次执行
2. 核对`process_pocmol_allinone.py`中的modes，与`apep`的处理方法比较
   1. `extract_pocket`：与extract_pockets.py的处理方法相同，都输出了`_pocket.pdb`，只是没有输出`meta_pocket.csv`
   2. `pocmol`：与process_pocmol.py的处理方法相同，得到了含有pocket_dict和mol_dict的PocketMolData
   3. `torsional`：与process_torsional_info.py的处理方法相同，得到了torsional_info
   4. `decompose`：与process_decompose_info.py的处理方法相同，得到了decompose_info
3. 结论：PepBDB预处理使用的`process_pocmol_allinone.py`，等价于`apep`预处理使用的`extract_pockets.py, process_pocmol.py, process_torsional_info.py, process_decompose_info.py`功能整合

**LMDB数据存储结构**

- `pepbdb`的`pocmol10.lmdb` = `apep`的`pocmol10.lmdb`+`torsion.lmdb`+`decom.lmdb`
- `pepbdb`的`pocmol10.lmdb` = `apep`的`peptide.lmdb`

**代码核对及修改**

经检查，向`process_pocmol_allinone.py`指定相应的路径，就可以直接应用于`bpep`和`cpep`数据集的预处理

```Python
if db_name in ['pepbdb']:
    mols_dir = f'data_train/{db_name}/files/mols'
    pro_dir = f'data_train/{db_name}/files/proteins'
    save_path = f'data_train/{db_name}/lmdb/pocmol10.lmdb'
    df_use = pd.read_csv(f'data_train/{db_name}/dfs/meta_filter.csv')
elif db_name in ['apep', 'bpep', 'cpep', 'peptest']:
    mols_dir = f'data_train/{db_name}/files/mols'
    pro_dir = f'data_train/{db_name}/files/proteins'
    save_path = f'data_train/{db_name}/lmdb/pocmol10.lmdb'
    df_use = pd.read_csv(f'data_train/{db_name}/dfs/meta_uni.csv')
```

**Explaination**

```Bash
python process/process_pocmol_allinone.py --db_name pepbdb
- 对 PepBDB 的Protein 和 Peptide (ligand) 做一体化处理
  - 从protein pdb里按ligand (这里为peptide) 周围半径 (r=10) Å 选残基，截出 pocket
  - 构建pocket + 分子图, 写入LMDB
  - 同时会进行torsion和decompositoin, 写入LMDB
- Input
  - meta data (使用 data_id 列)
  - protein pdb
  - ligand sdf (peptide)
- Output
  - pocket pdb
  - pocket+ligand lmdb

python process/process_peptide_allinone.py --db_name pepbdb
- 处理peptide的 PDB（残基/原子名/是否主链/序列等），将其写成 peptide.lmdb。并且强制检查：肽 PDB 的原子数与坐标要和 pocmol10.lmdb 里配体的第一构象完全一致（否则跳过）。
- Input
  - meta data (使用 data_id 列)
  - ligand pdb (peptide)
  - PocketMolData LMDB (作为reference)
- Output
  - peptide LMDB (包含peptide的特征)
```

## Debug & Log

Log：`process/process_peptide_allinone.py`

- **Error: residue id is not continuous**
  - `bpep`：7组数据中peptide的residue不连续，**skipped**
    - bpep_8t8r, bpep_7xuv, bpep_8fk3, bpep_8raj, bpep_8wqd, bpep_8he0
  - `cpep`：2组数据中peptide的residue不连续，**skipped**
    - cpep_8q1q, cpep_7z8o
  - 前期数据预处理（Data Filtering）时，需要去除residue不连续的peptide数据

```Python
assert (np.diff(unique_res_id) == 1).all(), 'residue id is not continuous'
```

- **Error: Atom mismatch, delta pos: xxxx**
  - `bpep`：1组数据中的peptide在pdb和sdf中的原子坐标不一致，**skipped**
    - bpep_8iqm
  - 前期数据预处理（Data Filtering）时，需要去除atom不一致的数据

```Bash
assert data['peptide_pos'].shape[0] == pos_ref.shape[0], 'Num of atom mismatch'
assert torch.allclose(data['peptide_pos'], pos_ref), 'Atom mismatch, delta pos: %s' % (data['peptide_pos'] - pos_ref).abs().max()
```

- **Warning: X in peptide sequence**
  - `bpep`：所有peptide均为标准氨基酸
  - `cpep`：23组数据中peptide的residue含非标准氨基酸，**warning**
    - `pepbdb`中也保留了peptide中的非标准氨基酸，存在较多含有X residue的peptide
  - peptide中的非标准氨基酸不影响后续代码的运行

```Bash
# 示例
Warning: X in peptide sequence: data_train/cpep/files/peptides/cpep_8ibo_pep.pdb
Warning: X in peptide sequence: data_train/cpep/files/peptides/cpep_8alx_pep.pdb
Warning: X in peptide sequence: data_train/cpep/files/peptides/cpep_8cix_pep.pdb
```

## LMDB数据结构

> 以下介绍apep的LMDB数据结构，PepBDB的LMDB数据内容与之相同，只是存储结构有所不同
>
> - `pepbdb`的`pocmol10.lmdb` = `apep`的`pocmol10.lmdb`+`torsion.lmdb`+`decom.lmdb`
> - `pepbdb`的`pocmol10.lmdb` = `apep`的`peptide.lmdb`

**`pocmol10.lmdb`****：Pocket (.pdb) + Ligand (.sdf)**

- scripts：`process/process_pocmol.py`
- 存入的数据（**PocketMolData**）
  - `ligand_dict`
    - 方法：`ligand_dict = parse_conf_list([mol], smiles=smiles)`
    - 描述：ligand（peptide）中的原子序数、化学键索引、化学键类型、各个构象的原子坐标等基本信息
    - 数据结构：

```Go
{
    'element': np.array(element),
    'bond_index': np.array(bond_index),
    'bond_type': np.array(bond_type),
    # 'bond_rotatable': np.array(data['bond_rotatable']),
    'pos_all_confs': np.array(pos_all_confs, dtype=np.float32),
    'num_atoms': num_atoms,
    'num_bonds': num_bonds,
    'i_conf_list': i_conf_list,
    'num_confs': len(i_conf_list),
}
```

- `pocket_dict`
  - 方法：`pocket_dict = PDBProtein(pdb_bloack).to_dict_atom()`
  - 描述：pocket（protein）中的原子序数、原子坐标、是否为骨架原子等基本信息
  - 数据结构：存入后，key加上**`pocket_`**前缀

```Python
{
    'element': np.array(self.element, dtype=np.int64),
    'molecule_name': self.title, # pdb HEADER
    'pos': np.array(self.pos, dtype=np.float32),
    'is_backbone': np.array(self.is_backbone, dtype=bool),
    'atom_name': self.atom_name,
    'atom_to_aa_type': np.array(self.atom_to_aa_type, dtype=np.int64)
}
```

- 其他基本信息

```Python
data.pdbid = pdbid
data.data_id = data_id
data.smiles = smiles
```

- 数据获取

```Python
from utils.dataset import LMDBDatabase
lmdb_path = f'data_train/{db_name}/lmdb/pocmol10.lmdb'
db = LMDBDatabase(lmdb_path, readonly=True)
db[data_id]
```

**`peptide.lmdb`****：peptide (.pdb)**

- scripts：`process/process_peptide_allinone.py`
- 存入的数据
  - `peptide_dict`
    - 方法：`data = process_peptide(ligand_path, data_id)`
    - 质控：peptide (.pdb)的atom num和atom pos需要与`pocmol10.lmdb`中的ligand (.sdf)保持一致
    - 描述：ligand（peptide）中的原子坐标、原子名称、残基索引、多肽序列等基本信息
    - 数据结构：存入后，key加上**`peptide_`**前缀

```Python
{
    'pos': np.array, atom coordinates,
    'atom_name': list, atom names,
    'is_backbone': np.array, whether the atom is a backbone atom,
    'res_id': np.array, residue index in PBD,
    'atom_to_aa_type': np.array, atom to amino acid type,
    'res_index': np.array, residue index (0 - len(peptide)),
    'seq': str, peptide sequence,
    'pep_len': int, peptide length,
    'pep_path': str, peptide path,
}
```

- 数据获取

```Python
from utils.dataset import LMDBDatabase
lmdb_path = f'data_train/{db_name}/lmdb/peptide.lmdb'
db = LMDBDatabase(lmdb_path, readonly=True)
db[data_id]
```

**`torsion.lmdb`****：Ligand (.sdf)**

- Scripts: `process/process_torsional_info.py`
- 存入的数据
  - torsional_info
    - 方法：`result = get_torsional_info_mol(mol, bond_index, data_id)`
    - 描述：ligand（peptide）中的可旋转键、可旋转键两侧的原子集合、固定原子的矩阵、邻近原子、同构型的映射等基本信息
    - 数据结构：

```Python
{
    'bond_rotatable': np.array(bond_rotatable, dtype=np.int64),
    'tor_twisted_pairs': tor_twist_pairs,
    'fixed_dist_torsion': fixed_dist,
    
    'tor_bond_mat': rot_mat,
    'path_mat': path_mat,
    'nbh_dict': nbh_dict,
    
    'matches_graph': matches_graph,
    'matches_iso': matches_isom,
}
```

- 数据获取

```Python
from utils.dataset import LMDBDatabase
lmdb_path = f'data_train/{db_name}/lmdb/torsion.lmdb'
db = LMDBDatabase(lmdb_path, readonly=True)
db[data_id]
```

**`decom.lmdb`****：Ligand (.sdf)**

- Scripts: `process/process_decompose_info.py`
- 存入的数据
  - decompose_info
    - 方法：`result_dict['brics'] = decompose_brics(mol)`，`result_dict['mmpa'] = decompose_mmpa(mol)`
      - BRICS（Breaking of Retrosynthetically Interesting Chemical Substructures） 一种按“合成上可切断”的键来拆分子的策略：只切断符合 BRICS 规则的那几类键（如 C–N、C–O 等），把分子拆成若干化学上有意义的子结构（类似逆合成里的 building blocks）。
      - MMPA（Matched Molecular Pairs Analysis） 这里用 MMPA 的思路做分子切断：用一条 SMARTS 规则挑出“可切断的键”（通常是单键、非环、碳–杂原子等），再按这些键把分子拆成片段（fragments）和连接子（linkers），便于做 linker/fragment 分析或生成。
    - 描述：ligand（peptide）中，根据BRICS / MMPA规则切断相关的键后，各个子图的原子索引列表、各个子图中的anchor atom索引集合、各个子图的邻居子图索引列表、subgraph和anchor atom的对应关系等基本信息
    - 数据结构：

```Python
{
    'brics': {
        'subgraphs': node_list,
        'anchors_list': anchors,
        'nbh_subgraphs': nbh_subgraphs,
        'connections': connections,
    },
    'mmpa': {
        'subgraphs': node_list,
        'anchors_list': anchors,
        'nbh_subgraphs': nbh_subgraphs,
        'connections': connections,
    },
}
```

- 数据获取

```Python
from utils.dataset import LMDBDatabase
lmdb_path = f'data_train/{db_name}/lmdb/decom.lmdb'
db = LMDBDatabase(lmdb_path, readonly=True)
db[data_id]
```

# 测试集Benchmark

## 前期准备

1. 创建config文件

   1. Peptide Docking
      1. `bpep`：configs/sample/test/dock_bpep/base.yml
      2. `cpep`：configs/sample/test/dock_cpep/base.yml

2. 调整config中的参数

   1. `sample`
      1. batch_size = 20 # Optional：降低batch_size，防止CUDA out of memory
      2. **num_repeats = 200**
   2. `data`和`task`
      1. 将db_name修改为对应的名称
   3. 其他参数与`pepbdb`保持一致
      1. Peptide Docking：configs/sample/test/dock_pepbdb/base.yml
      2. Inverse Folding：configs/sample/test/pepinv_pepbdb/base.yml
      3. Peptide Design：configs/sample/test/pepdesign_pepbdb/base.yml

3. 根据config文件（base.yml）准备相关文件

   1. config中的数据格式（以pepbdb为例：configs/sample/test/dock_pepbdb/base.yml）

      ```python
      data:
        dataset:
          root: ./data
          assembly_path: test/assemblies/dock_pepbdb.csv # test/assemblies/lmdb/dock_pepbdb.lmdb
          dbs:
          - name: pepbdb
            lmdb_root: pepbdb/lmdb
            lmdb_path:
              pocmol10: pocmol10.lmdb
              peptide: peptide.lmdb
      ```

      

   2. assembly_path

      1. `pepbdb`
         1. dock_pepbdb.csv的列名：`data_id`,`data_task`,`db`,`split`
      2. `bpep`和`cpep`按相同的格式准备
         1. `bpep`：test/assemblies/dock_bpep.csv
         2. `cpep`：test/assemblies/dock_cpep.csv
      3. 运行：`extensions/make_test_assemblies.py`

      ```python
      # peptide docking
      python extensions/make_test_assemblies.py --db_name bpep --data_task dock
      python extensions/make_test_assemblies.py --db_name cpep --data_task dock
      
      # inverse folding / peptide design
      python extensions/make_test_assemblies.py --db_name bpep --data_task pepdesign
      python extensions/make_test_assemblies.py --db_name cpep --data_task pepdesign
      ```

   3. dbs

      1. **`pepbdb`**：位于**`./data`**

         1. `proteins_combchain`：非必须
            1. 来自proteins中的pdb
            2. 变更：chain id变成R，atom index和residue index变成从1开始
         2. `peptide.lmdb`
            1. 与`./data_train/pepbdb/lmdb/peptide.lmdb`比较
               1. 相同：数据类型和格式相同（keys和values）
               2. 不同：'peptide_pep_path'指定的路径由./data_train更改为./data
         3. `pocmol10.lmdb`
            1. 与`./data_train/pepbdb/lmdb/pocmol10.lmdb`比较
               1. 相同：数据类型和格式相同（keys和values）

         ```python
         pepbdb
         ├── files
         │   ├── peptides
         │   ├── proteins
         │   └── proteins_combchain
         └── lmdb
             ├── peptide.lmdb
             └── pocmol10.lmdb
         ```

      2. `bpep`和`cpep`

         1. 将./data_train中的相关数据复制至./data
         2. 更改peptide.lmdb以适配模型测试
            1. `peptide_pep_path`：将其中的`./data_train`更改为`./data`

         ```bash
         cp -r ./data_train/bpep ./data
         cp -r ./data_train/cpep ./data
         
         python extensions/make_test_lmdb.py --db_name bpep
         python extensions/make_test_lmdb.py --db_name cpep
         ```

4. 调整代码

   1. 添加db_name：`bpep`和`cpep`
      1. `utils/dataset.py`中的`get_data_key`函数

```Python
elif db_name in ['apep']:
    key = f'pocmol10/{data_id};torsion/{data_id};decom/{data_id};peptide/{data_id}'
elif db_name in ['pepbdb', 'qbpep', 'bpep', 'cpep']:
    key = f'pocmol10/{data_id};peptide/{data_id}'
```

## 运行结果

- config中的相关参数（`configs/sample/test/dock_pepbdb/base.yml`）
  - assembly_path
    - Option 1：`assembly_path:  test/assemblies/lmdb/dock_pepbdb.lmdb`
    - Option 2：`assembly_path:  test/assemblies/dock_pepbdb.csv`
      - 通过csv检索`./data/pepbdb/lmdb/pocmol10.lmdb`和`./data/pepbdb/lmdb/peptide.lmdb`
- 输出结果
  - 输出路径
    - Terminal运行（num_repeats = 50）
      - /home/yangziqing/PocketXMol/outputs_test/dock_pepbdb/base_pxm_20260317_172745
  - **数据形式**：`pepbdb`一共79个测试数据
    - **`gen_info.csv`**：generated info
      - data_id
      - filename：生成的pdb文件名
      - i_repeat：num_repeats的index
    - **SDF目录**
      - `xx_mol.sdf`：模型输出的结构
      - `xx.pdb`：模型输出的结构，由`xx_mol.sdf`转化得到
      - `xx_gt.pdb`：ground truth结构

```Bash
# docking
python scripts/sample_pdb.py \
    --config_task configs/sample/test/dock_pepbdb/base.yml \
    --outdir outputs_test/dock_pepbdb \
    --device cuda:0
    
# inverse-folding
python scripts/sample_pdb.py \
    --config_task configs/sample/test/pepinv_pepbdb/base.yml \
    --outdir outputs_test/pepinv_pepbdb \
    --device cuda:0

# peptide design
## pepbdb
python scripts/sample_pdb.py \
    --config_task configs/sample/test/pepdesign_pepbdb/base.yml \
    --outdir outputs_test/pepdesign_pepbdb \
    --device cuda:0
## bpep
python scripts/sample_pdb.py \
    --config_task configs/sample/test/pepdesign_bpep/base.yml \
    --outdir outputs_test/pepdesign_bpep \
    --device cuda:0
## cpep
python scripts/sample_pdb.py \
    --config_task configs/sample/test/pepdesign_cpep/base.yml \
    --outdir outputs_test/pepdesign_cpep \
    --device cuda:1
```