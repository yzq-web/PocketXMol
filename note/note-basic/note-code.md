## multiprocessing多线程操作示例

- process/extract_pockets.py

```python
import multiprocessing as mp

pool = mp.Pool(args.num_workers)
result_list = []
for item_pocket in tqdm(pool.imap_unordered(
        partial(process_item, radius=args.radius), data_list), total=len(data_list)):
    result_list.append(item_pocket)
```

简单来说，它就像是一个**工厂流水线**：你有一大堆零件（`data_list`），雇佣了一群工人（`pool`），让他们并排坐着处理这些零件，最后把做好的成品收集起来。

### 核心组件拆解

- `**mp.Pool(args.num_workers)`**: 创建一个“进程池”。`num_workers` 是工人的数量（通常对应 CPU 的核心数）。
- `**partial(process_item, radius=args.radius)`**: 这是一个“预设参数”的技巧。因为 `imap` 每次只能传一个参数，如果你的函数需要多个参数（如 `radius`），就用 `partial` 把不变的参数先“固定”住。
- `**imap_unordered`**: 这是多进程的核心动作。
  - `i` 代表 `iterable`（可迭代）。
  - `unordered` 表示**不按顺序**返回结果。谁先做完，谁的结果就先传回来。这比按顺序返回（`imap`）效率更高，因为不需要等待走得慢的进程。
- `**tqdm(...)`**: 这是一个进度条工具，让你能实时看到任务完成了百分之几。
- `**pool.close()`**: 告诉工厂不要再接收新任务了，准备收工。

### 更简单的示例：帮数字“翻倍”

为了让你看清底层逻辑，我们去掉 `partial` 和 `tqdm`，写一个最纯粹的多进程代码：

```Python
import multiprocessing as mp

# 1. 定义你要重复做的任务
def double_number(n):
    return n * 2

if __name__ == "__main__":
    # 准备数据
    data_list = [1, 2, 3, 4, 5]

    # 2. 开启进程池（假设用 2 个工人）
    pool = mp.Pool(processes=2)

    # 3. 分派任务并收集结果
    # imap_unordered 会把 data_list 里的数字一个个喂给 double_number
    results = []
    for res in pool.imap_unordered(double_number, data_list):
        results.append(res)
        print(f"收到结果: {res}")

    # 4. 清理
    pool.close()
    pool.join() # 等待所有工人打扫完战场离开

    print(f"最终列表: {results}")
```

### 为什么要这么写？（优势对比）


| **特性**      | **普通 for 循环**   | **mp.Pool (多进程)** |
| ----------- | --------------- | ----------------- |
| **速度**      | 慢（一个接一个做）       | **快**（多个人同时做）     |
| **CPU 利用率** | 只用到一个核心（可能 10%） | **跑满所有核心** (100%) |
| **顺序**      | 始终是 1, 2, 3...  | **随机**（取决于谁算得快）   |


## @staticmethod

`@staticmethod` 是一个**装饰器**，用于定义类中的**静态方法**

调用静态方法时，不需要创建类的实例（对象），可以直接通过 `类名.方法名()` 调用

```python
class Robot:
    brand = "TechCorp"

    def __init__(self, name):
        self.name = name

    # 1. 实例方法：需要访问实例属性 (self.name)
    def say_hello(self):
        print(f"你好，我是 {self.name}")

    # 2. 类方法：需要访问类属性 (cls.brand)
    @classmethod
    def get_brand(cls):
        print(f"品牌是: {cls.brand}")

    # 3. 静态方法：既不需要 self 也不需要 cls
    @staticmethod
    def add_numbers(a, b):
        return a + b

# 使用静态方法
result = Robot.add_numbers(5, 10)  # 直接调用，无需实例化
print(result) # 输出 15
```

## Load config

```python
from easydict import EasyDict
import yaml
def load_config(path):
    with open(path, 'r') as f:
        return EasyDict(yaml.safe_load(f))
```

## Numpy

### 去重排序

- `PocketXMol/utils/parser.py`中的`parse_pdb_peptide`函数
- 被`process/process_peptide_allinone.py`调用

```python
unique_res_id, res_index = np.unique(pep_feature_dict['res_id'], return_inverse=True)
```

`p.unique(arr, return_inverse=True)` 会返回两个数组：

- **第一个**：`arr` 去重后**排序**得到的唯一值数组（这里叫 `unique_res_id`）。
- **第二个**：`inverse`（这里叫 `res_index`），长度和原始 `arr` 一样，**每个元素表示“该位置上的值在唯一值数组中的下标”**。

因此：

- `unique_res_id`：所有出现过的残基 ID，去重且排好序，例如 `[1, 2, 3, 4, ...]`。
- `res_index`：每个原子对应的残基在 `unique_res_id` 里的索引（0, 1, 2, ...），把“按原子”的 `res_id` 映射成“按残基”的连续下标。

**简单例子**：若 `res_id = [2, 2, 3, 2, 3]`（每个原子一个 ID），则：

- `unique_res_id = [2, 3]`
- `res_index = [0, 0, 1, 0, 1]`  
即：第 0、1、3 个原子属于残基 2（索引 0），第 2、4 个原子属于残基 3（索引 1）。这样后面用 `res_index` 就可以按残基分组或做残基级操作。

### 相邻差分

- `PocketXMol/utils/parser.py`中的`parse_pdb_peptide`函数
- 被`process/process_peptide_allinone.py`调用

```python
assert (np.diff(unique_res_id) == 1).all(), 'residue id is not continuous'
```

- `np.diff(unique_res_id)`：对 `unique_res_id` 做**相邻差分**，即  
`unique_res_id[1] - unique_res_id[0]`, `unique_res_id[2] - unique_res_id[1]`, …  
得到的是“相邻残基 ID 的差值”组成的数组。
- `== 1`：检查这些差值是否**全部等于 1**。若等于 1，说明残基 ID 是**严格连续**的（如 1, 2, 3, 4…），没有缺号、没有跳号。
- `.all()`：要求每一个差值都是 1。
- `assert ..., 'residue id is not continuous'`：若不连续（存在差值不为 1），就抛出断言错误，提示 `'residue id is not continuous'`。

## assert

assert

- 当 条件为 True: 什么都不做，代码继续往下执行
- 当 条件为 False: 立刻抛出 AssertionError 异常

```python
# 用法
assert 条件, '错误信息'
# 示例
assert (np.diff(unique_res_id) == 1).all(), 'residue id is not continuous'
```

如果需要在弹出error之后继续运行, 则需配合try...except...

```python
for ...:
    try:
        data = process_peptide(ligand_path, data_id) # 内含assert语句
    except AssertionError as e:
        print(f'{ligand_path} 出错: {e}')
        # 你可以选择：
        # 1. 跳过这条数据继续下一个
        #    continue
        # 2. 给 data 一个默认值
        data = None
        continue    # 或者不写，看你后面怎么用 data
    ...
```

## logging

- Ref: PocketXMol/utils/misc.py
- 用于: PocketXMol/scripts/sample_pdb.py

这个函数是一个非常实用的工具，它帮你完成了日志系统的“三要素”设置：

- 谁来记？ (logger)
- 怎么记？ (formatter)
- 记到哪？ (StreamHandler 到屏幕，FileHandler 到文件)

定义函数

```python
import logging
from logging import Logger

def get_logger(name, log_dir=None):
    logger = logging.getLogger(name) # 初始化Logger: 接收log
    logger.setLevel(logging.DEBUG) # 日志级别: DEBUG, debug/info/warning/error 都会输出
    formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s') # 定义Formatter: 格式化log
    # e.g. '[2026-03-17 10:00:00::sample::DEBUG] This is a debug message'

    stream_handler = logging.StreamHandler() # 控制台输出(StreamHandler): 输出log至控制台
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        file_handler = logging.FileHandler(os.path.join(log_dir, 'log.txt')) # 文件输出(FileHandler): 输出log到文件
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
```

记录log

```python
logger = get_logger('sample', log_dir)
logger.info('Load from %s...' % config.model.checkpoint) # 同时输出至控制台和log.txt
```

## 语法糖

- PocketXMol/utils/[transforms.py](http://transforms.py)

_TRAIN_DICT是一个空字典，为什么可以直接进行`_TRAIN_DICT[name](config, *args, **kwargs)`

原因在于 执行顺序：`_TRAIN_DICT` 在定义时是空的，但在你第一次调用 `get_transforms` 之前，整个 `transforms.py` 已经 import 完，字典早就被装饰器填满了。

**发生了什么**

1. 加载模块时
  Python 从上到下执行 `transforms.py`。
   执行到 `_TRAIN_DICT = {}` 时，字典确实是空的。
2. 接着执行后面的类定义
  每遇到一个带 `@register_transforms('xxx')` 的类，装饰器会执行类似： `_TRAIN_DICT['xxx'] = CutPeptide`（类对象）。
   文件里所有被装饰的类都会在 import 阶段 注册进去。
3. 别的代码 `from utils.transforms import get_transforms` 时
  必须先跑完整个模块，所以此时 `_TRAIN_DICT` 已经不是空字典。
4. 之后 `get_transforms` 里 `_TRAIN_DICT[name](...)` 取的是已经注册好的类，再实例化。

**小结**

- “一开始是空字典” 只对模块刚执行到那一行的瞬间成立。
- 调用 `get_transforms` 时，前提是已经 `import` 过该模块，注册已完成。
- 若有人只执行了文件前几行、或动态删了键、或 `name` 从未被注册，那时才会 `KeyError`，而不是因为“字典生来就该是空的”。

请给出一个具体示例，说明装饰器是如何执行注册的

下面用你项目里的真实写法，加上一个最小可运行小例子，说明装饰器是怎么在 import 时把类写进 `_TRAIN_DICT` 的。

### 1. 项目里的机制（等价于「语法糖」）

```python
def register_transforms(name: str):
    """Decorator to register transform classes by name.
    ...
    """
    def decorator(cls):
        _TRAIN_DICT[name] = cls
        return cls
    return decorator
```

当你写：

```python
@register_transforms('cut_peptide')
class CutPeptide(object):
    ...
```

Python 会做一件事：先定义类 `CutPeptide`，再调用
`CutPeptide = register_transforms('cut_peptide')(CutPeptide)`。

展开后等价于：

```python
class CutPeptide(object):
    ...

CutPeptide = register_transforms('cut_peptide')(CutPeptide)
```

执行顺序是：

1. `register_transforms('cut_peptide')` 被调用 → 返回内层函数 `decorator`（此时 `name` 已经绑定为 `'cut_peptide'`）。
2. `decorator(CutPeptide)` 被调用 → 执行 `_TRAIN_DICT['cut_peptide'] = CutPeptide`，然后 `return cls`（类本身不变）。
3. 所以注册发生在 类定义完成之后、赋值给名字 `CutPeptide` 之前；且发生在 import 该模块的过程中。

因此等你 `from utils.transforms import get_transforms` 时，文件里所有带 `@register_transforms(...)` 的类都已经执行过一遍， `_TRAIN_DICT` 里已有对应条目。

---

### 2. 一个最小独立示例（看打印顺序就懂）

```python
_REGISTRY = {}

def register(name):
    def decorator(cls):
        print(f"注册: name={name!r}, class={cls.__name__}")
        _REGISTRY[name] = cls
        return cls
    return decorator

print("1. 定义装饰器后，_REGISTRY =", _REGISTRY)

@register("foo")
class Foo:
    pass

print("2. 定义完 Foo 后，_REGISTRY =", _REGISTRY)
print("3. 通过名字实例化:", _REGISTRY["foo"]())
```

典型输出：

```python
1. 定义装饰器后，_REGISTRY = {}
注册: name='foo', class=Foo
2. 定义完 Foo 后，_REGISTRY = {'foo': <class '__main__.Foo'>}
3. 通过名字实例化: <__main__.Foo object at 0x...>
```

要点：`{}` 只在最开始出现一次；一旦出现 `@register("foo") class Foo`，装饰器体立刻运行，字典里就有 `'foo'` 了。你项目里的 `_TRAIN_DICT[name](config, ...)` 和这里的 `_REGISTRY["foo"]()` 是同一类用法：字典里存的是类，后面再加括号就是构造实例。