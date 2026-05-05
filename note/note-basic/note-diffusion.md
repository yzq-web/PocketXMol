
- PocketXMol/utils/info_level.py
```python
class AdvanceScaler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        scale_start = nn.Parameter(torch.tensor(config['scale_start']), requires_grad=False)
        scale_end = nn.Parameter(torch.tensor(config['scale_end']), requires_grad=False)
        width = nn.Parameter(torch.tensor(config['width']), requires_grad=False)
        self.setup(scale_start, scale_end, width)
        
    def setup(self, scale_start, scale_end, width):
        self.k = width
        A0 = scale_end
        A1 = scale_start

        self.a = (A0-A1)/(torch.sigmoid(-self.k) - torch.sigmoid(self.k))
        self.b = 0.5 * (A0 + A1 - self.a)
        
    def __call__(self, x):
        x = 2 * x - 1  # step [0, 1] -> x [-1, 1]
        return self.a * torch.sigmoid(- self.k * x) + self.b
```
这段代码实现了一个基于 **Sigmoid 函数** 的非线性缩放器，通常用于扩散模型（Diffusion Models）中控制噪声调度（Noise Schedule）或时间步（Timestep）的权重分配。

在扩散模型中，随着时间步 $t$ 从 $0$ 增加到 $1$，我们通常希望某个参数（如信号强度或噪声比例）从一个起始值 `scale_start` 平滑地过渡到 `scale_end`。

------

### 1. 数学逻辑推导

该类核心的数学表达式为：

$$f(x) = a \cdot \sigma(-k \cdot x) + b$$

其中 $x \in [-1, 1]$（由输入 $t \in [0, 1]$ 线性映射而来），$\sigma$ 是 Sigmoid 函数 $\sigma(z) = \frac{1}{1 + e^{-z}}$。

为了让函数在指定的端点取得特定值，代码通过 `setup` 计算了 $a$ 和 $b$。设目标端点值为：

- 当 $x = -1$ 时（对应 $t=0$），期望输出 $f(-1) = A_1$ (`scale_start`)
- 当 $x = 1$ 时（对应 $t=1$），期望输出 $f(1) = A_0$ (`scale_end`)

带入公式得到方程组：

1. $A_1 = a \cdot \sigma(k) + b$
2. $A_0 = a \cdot \sigma(-k) + b$

#### $a$ 的含义：振幅/缩放系数 (Amplitude/Scaling Factor)

通过两式相减 $(2) - (1)$：

$$A_0 - A_1 = a(\sigma(-k) - \sigma(k)) \implies a = \frac{A_0 - A_1}{\sigma(-k) - \sigma(k)}$$

- **物理意义**：$a$ 决定了曲线在 $y$ 轴方向上的**跨度**。它负责将 Sigmoid 函数的标准输出范围（原本在 $k$ 影响下的有限区间）拉伸或压缩，以匹配 `scale_start` 到 `scale_end` 之间的差值。

#### $b$ 的含义：偏移量 (Bias/Offset)

通过两式相加 $(1) + (2)$，利用 Sigmoid 的对称性 $\sigma(k) + \sigma(-k) = 1$：

$$A_0 + A_1 = a(\sigma(-k) + \sigma(k)) + 2b = a(1) + 2b \implies b = \frac{A_0 + A_1 - a}{2}$$

- **物理意义**：$b$ 是一个**平移项**。它确保曲线在经过 $a$ 缩放后，其整体位置能在垂直方向上精确对齐，使得端点恰好落在指定的起始和终止数值上。

------

### 2. 为什么这样设计时间步？

这种设计相比于简单的线性插值（Linear Interpolation）有几个显著特点：

- **控制过渡节奏 ($k$)**：参数 `width`（即 $k$）控制了过渡的“陡峭”程度。
  - 如果 $k$ 很大，变化会集中在 $t=0.5$ 附近，两头相对平稳。
  - 如果 $k$ 较小，曲线会趋近于线性。
- **S 型曲线 (S-Curve)**：在扩散模型中，通常需要在 $t$ 较小时（靠近原图）和 $t$ 较大时（靠近纯噪声）保持较慢的变化速率，而在中间阶段快速过渡。Sigmoid 恰好提供了这种“两头平缓、中间急促”的特性。
- **归一化处理**：代码中 `x = 2 * x - 1` 将原始的 $[0, 1]$ 时间步映射到 $[-1, 1]$，是为了利用 Sigmoid 函数关于原点 $(0, 0.5)$ 的对称性，使得计算 $a$ 和 $b$ 的过程更加简洁且数值稳定。

### 总结

- **$x$**：将物理时间步转化为对称的计算空间。
- **$k$ (width)**：形状控制器，决定过渡是“突变”还是“渐变”。
- **$a$**：缩放比例，解决“变多大”的问题。
- **$b$**：基准偏移，解决“从哪变到哪”的问题。