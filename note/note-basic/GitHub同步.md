## PocketXMol 仓库初始化与远程同步全流程

### 第一阶段：本地仓库初始化

在你的 Linux 服务器（`master00`）终端执行：

```
# 1. 进入项目目录
cd ~/PocketXMol

# 2. 初始化 Git 仓库
git init

# 3. 创建并配置忽略文件（非常重要，避免上传冗余的缓存和大型中间数据）
# 如果有大型模型权重或临时数据，建议也加上：
# checkpoints/
# data/test_results/
cat > .gitignore <<EOF
__pycache__/
*.pyc
.DS_Store
.ipynb_checkpoints/
data
data_zip
data_train
outputs_test
outputs_examples
rosetta_tmp
EOF

# 4. 将所有代码文件添加到暂存区
git add .

git add configs data_rama environment.yml LICENSE metric_stat notebooks process scripts docs evaluate models README.md utils environment_cu128_base.yml extensions metrics_stat.ipynb note

# 5. 提交到本地（确保已设置过 user.name 和 user.email）
git config --global user.email "470886632@qq.com"
git config --global user.name "yzq-web"
git commit -m "Update PepDesign Metrics"
```

------

### 第二阶段：远程仓库关联

在执行此步前，请确保你已经在 GitHub 网页端创建了名为 `PocketXMol` 的空仓库。

```
# 6. 关联到你自己的 GitHub 远程仓库
# 如果之前关联错了，先执行：git remote remove origin
git remote add origin https://github.com/yzq-web/PocketXMol.git

# 7. (可选) 如果需要关联原作者仓库作为 upstream 以便同步更新
git remote add upstream https://github.com/pengxingang/PocketXMol.git

# 8. 验证远程连接情况
git remote -v
```

------

### 第三阶段：推送代码至 GitHub

```
# 9. 将本地 master 分支的代码推送到 GitHub origin 仓库
# -u 参数会建立追踪关系，以后只需输入 git push 即可
git push -u origin master
```

**添加密钥：登录新的集群，push之前需要设置密钥**

**第一步：检查是否已有现成的密钥**

在当前集群运行：

```
ls -al ~/.ssh
```

看是否存在 `id_rsa.pub` 或 `id_ed25519.pub`。如果没有，请执行第二步。

**第二步：生成新的 SSH 密钥**

运行以下命令（一路按 **Enter** 键即可，无需设置密码）：

```
ssh-keygen -t ed25519 -C "你的邮箱@example.com"
```

**第三步：复制公钥 (Public Key)**

生成后，查看并复制这一串以 `ssh-ed25519` 开头的字符：

```
cat ~/.ssh/id_ed25519.pub
```

**第四步：将密钥添加到 GitHub**

1. 登录 GitHub，点击右上角头像 -> **Settings**。
2. 在左侧菜单找到 **SSH and GPG keys**。
3. 点击 **New SSH key**。
4. **Title**: 填入这个集群的名字（例如：`Cluster-B`），方便以后管理。
5. **Key**: 粘贴刚才 `cat` 命令查看到的完整内容。
6. 点击 **Add SSH key**。

**第五步：验证连接**

运行测试命令：

```
ssh -T git@github.com
```

如果看到 `Hi [你的用户名]! You've successfully authenticated...`，就说明连接成功了。

------

## 💡 进阶：日常更新维护流程

以后如果你在 `PocketXMol` 中修改了代码（例如优化了蛋白质配体结合的分析脚本），只需重复以下“三板斧”：

1. **查看状态**：`git status`（看看改了哪些文件）。
2. **暂存修改**：`git add .`。
3. **提交说明**：`git commit -m "Update: optimized binding affinity calculation"`。
4. **推送到云端**：`git push`。

------

### 常见问题小贴士

- **输入密码麻烦？** 如果频繁推送，建议在 GitHub 设置 **SSH Key** 或者使用 **Personal Access Token (PAT)**，这样就不需要每次都输入 GitHub 账号密码了。
- **分支名称**：目前你使用的是 `master`，如果 GitHub 仓库默认显示的是 `main`，你可以通过 `git branch -M main` 将本地也改为 `main` 以保持一致。





## GitLab

### 第一阶段：本地仓库初始化

```bash
# 1. 进入项目目录
cd ~/affinity_rank

# 2. 初始化 Git 仓库
git init

# 3. 创建并配置忽略文件（非常重要，避免上传冗余的缓存和大型中间数据）
# 如果有大型模型权重或临时数据，建议也加上：
# checkpoints/
# data/test_results/
cat > .gitignore <<EOF
__pycache__/
*.pyc
.DS_Store
.ipynb_checkpoints/
data
data_zip
data_train
outputs_test
EOF

# 4. 将所有代码文件添加到暂存区
git add .

# 5. 提交到本地（确保已设置过 user.name 和 user.email）
git config --global user.email "ziqing.yang@synerontech.com"
git config --global user.name "ZiqingYang"
git commit -m "Initial commit for affinity_rank"
```

### 第二阶段：远程仓库关联

在执行此步前，请确保你已经在 GitHub 网页端创建了名为 `PocketXMol` 的空仓库。

```
# 6. 关联到你自己的 GitHub 远程仓库
# 如果之前关联错了，先执行：git remote remove origin
git remote add rank-origin https://jihulab.com/ZiqingYang/affinity_rank.git

# 7. (可选) 如果需要关联原作者仓库作为 upstream 以便同步更新
git remote add rank-upstream https://jihulab.com/xiongyp/affinity_rank.git

# 8. 验证远程连接情况
git remote -v
```

### 第三阶段：推送代码至 GitHub

```
# 9. 将本地 main 分支的代码推送到 GitLab 仓库
# -u 参数会建立追踪关系，以后只需输入 git push 即可
git push -u origin main
```

