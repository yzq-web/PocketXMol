- Ref：`utils/dataset.py`中的`class LMDBDatabase(Dataset)`

## 1. LMDB 是什么

LMDB（Lightning Memory-Mapped Database）是一个键值对数据库，特点包括：

- 内存映射：数据通过 mmap 访问，读多写少场景下很快
- ACID：支持事务，数据一致性好
- 单写多读：适合“预处理一次、训练时多进程/多机读”的流程
- 零拷贝：读时可直接从映射内存取，适合存二进制（如 pickle 的 PyTorch 数据）

在 PocketXMol 里，LMDB 用来存预处理好的 pocket–分子对（`PocketMolData`），避免训练时反复读 SDF/PDB 和重复解析。

## 2. `import lmdb` 提供什么

`lmdb` 是 LMDB 的 Python 绑定，主要用到：


| 功能                            | 说明                         |
| ----------------------------- | -------------------------- |
| `lmdb.open(path, ...)`        | 打开/创建 LMDB 环境（数据库目录）       |
| `env.begin(write=True/False)` | 开启读或写事务                    |
| `txn.put(key, value)`         | 写入键值（key/value 需为 bytes）   |
| `txn.get(key)`                | 按 key 读取                   |
| `txn.stat()['entries']`       | 当前库中的条目数                   |
| `txn.cursor()`                | 遍历所有 key（如 `get_all_keys`） |


也就是说：`import lmdb` 让你在 Python 里打开 LMDB、做事务、读写字节数据；项目里再在上面包一层 `LMDBDatabase`，用 `pickle` 把 Python 对象序列化成 bytes 再存进去。

## 3. 项目里的 `LMDBDatabase`（`utils/dataset.py`）

`LMDBDatabase` 在 `dataset.py` 里封装了 `lmdb` 的常用操作，方便统一读写和与 PyTorch `Dataset` 配合：

- 打开方式：
  - 只读：`lmdb.open(..., readonly=True, lock=False, readahead=False, meminit=False)`，适合多 worker 读。
  - 可写：`lmdb.open(..., readonly=False)`，用于 `process_pocmol.py` 这种写库脚本。
- 写入：
  - `add_one(key, value)`：单条写入，key 转 bytes，value 用 `pickle.dumps`。
  - `add(data_dict)`：批量写入，同样 key 编码 + value pickle。
- 读取：
  - `__getitem__(key)`：`txn.get(key.encode())` 后 `pickle.loads` 返回 Python 对象。
  - `__len__`：用 `txn.stat()['entries']` 得到条数。
  - `get_all_keys()`：用 cursor 遍历得到所有 key 的字符串列表。

也就是说：LMDB 只存 bytes；`import lmdb` 负责和 LMDB 交互，`LMDBDatabase` 负责把 `data_id`、`PocketMolData` 等 Python 对象变成 bytes 再交给 lmdb。

## 4. 小结

- LMDB：高性能、内存映射的 key-value 库，用来存预处理好的 pocket–分子数据，训练时多进程/多机只读，速度快、省重复解析。
- `import lmdb`：在 Python 里打开 LMDB、做事务、按 key 读写字节；项目里所有 LMDB 的底层操作都依赖它。
- `process_pocmol.py`：用 `LMDBDatabase`（内部用 `lmdb`）把 SDF + PDB 处理成的 `PocketMolData` 按 `data_id` 写入第一个 LMDB（pocmol），供后续训练/评估直接按 key 读取。

关于`self.env`

在这段代码里，`self.env` 是 **LMDB 的“环境对象（Environment）”**，由 `lmdb.open(...)` 返回。

更具体一点：

- `lmdb.open(...)` 会打开（或创建）一个 LMDB 数据库文件/目录，并返回一个 `Environment` 实例。
- 这个 `Environment` 就是整个 LMDB 数据库的“句柄”，你后面所有的读写操作都要通过它来开启事务。

在你的类里：

```
 self.env = lmdb.open(...)
```

之后：

- **写入时**（`add` / `add_one`）：
  ```
   with self.env.begin(write=True) as txn:
       txn.put(...)
  ```
  这里通过 `self.env.begin(write=True)` 从环境 `env` 创建一个**写事务** `txn`，用它向数据库里 `put` 数据。
- **读取单条时**（`__getitem__`）：
  ```
   with self.env.begin() as txn:
       value = txn.get(key.encode())
  ```
  这里用 `self.env.begin()` 创建**读事务**，在这个事务里按 key 取数据。
- **统计条目数**（`__len__`）：
  ```
   with self.env.begin() as txn:
       return txn.stat()['entries']
  ```
  通过环境创建的事务拿到 `stat()`，得到当前库里有多少条记录。
- **遍历所有 key**（`get_all_keys`）：
  ```
   with self.env.begin() as txn:
       return [k.decode() for k in txn.cursor().iternext(values=False)]
  ```
  同样先从 `self.env` 开一个事务，再通过 `cursor()` 遍历所有 key。

所以 **一句话**：  
`self.env` **是对某个 LMDB 数据库文件的“连接/环境”，你用它来开启事务（**`begin`**）、获得统计信息（**`stat`**）、获取游标（**`cursor`**），本类所有对 LMDB 的读写都是通过这个** `self.env` **完成的。**