# Online Softmax增量计算

> 在不知道完整序列的情况下增量计算Softmax

---

## 1. 问题回顾

### 1.1 标准Softmax的限制

标准Softmax需要**两遍遍历**才能计算：

```
第一遍：计算 m = max(x₁, x₂, ..., xₙ)
第二遍：计算 softmax(xᵢ) = exp(xᵢ - m) / Σⱼ exp(xⱼ - m)
```

**问题：** 必须先看完所有数据才能开始计算！

### 1.2 FlashAttention的需求

FlashAttention需要**分块处理**数据：

```
┌─────────┐  ┌─────────┐  ┌─────────┐      ┌─────────┐
│ Block 1 │  │ Block 2 │  │ Block 3 │ ...  │ Block K │
└─────────┘  └─────────┘  └─────────┘      └─────────┘
     ↓            ↓            ↓               ↓
  处理完成     处理完成     处理完成         处理完成
```

**需求：** 每处理完一个块，就要能得到正确的**部分结果**，并能与后续块**正确合并**。

---

## 2. Online Softmax核心思想

### 2.1 从两遍到一遍

Online Softmax的突破在于：**单遍遍历**就能计算Softmax。

核心技巧：同时维护两个运行时变量：
- $m$：当前已见元素的最大值
- $l$：当前的（未完全归一化的）累加和

### 2.2 基本算法

```python
def online_softmax(x):
    """
    Online Softmax：单遍遍历计算
    """
    n = len(x)
    
    # 初始化
    m = -inf  # 最大值
    l = 0     # 累加和
    
    # 单遍遍历
    for i in range(n):
        m_prev = m
        m = max(m, x[i])                      # 更新最大值
        l = l * exp(m_prev - m) + exp(x[i] - m)  # 更新累加和
    
    # 计算输出
    y = []
    for i in range(n):
        y.append(exp(x[i] - m) / l)
    
    return y
```

**关键点：** 当最大值更新时，需要**缩放**之前的累加和！

### 2.3 为什么需要缩放

当新元素 $x_i$ 使最大值从 $m_{old}$ 更新到 $m_{new}$ 时：

```
之前的累加和: l_old = Σⱼ<ᵢ exp(xⱼ - m_old)
正确的累加和: l_new = Σⱼ≤ᵢ exp(xⱼ - m_new)
```

需要调整：
$$
l_{new} = l_{old} \cdot e^{m_{old} - m_{new}} + e^{x_i - m_{new}}
$$

---

## 3. 公式推导

### 3.1 初始状态

处理第一个元素 $x_0$：
- $m_0 = x_0$
- $l_0 = e^{x_0 - m_0} = e^0 = 1$

### 3.2 更新规则

处理第 $i$ 个元素 $x_i$：

**Step 1: 更新最大值**
$$
m_i = \max(m_{i-1}, x_i)
$$

**Step 2: 更新累加和**
$$
l_i = l_{i-1} \cdot e^{m_{i-1} - m_i} + e^{x_i - m_i}
$$

### 3.3 正确性证明

我们要证明 $l_i = \sum_{j=0}^{i} e^{x_j - m_i}$

**归纳基础：** $i=0$ 时，$l_0 = 1 = e^{x_0 - m_0}$ ✓

**归纳步骤：** 假设 $l_{i-1} = \sum_{j=0}^{i-1} e^{x_j - m_{i-1}}$

$$
\begin{aligned}
l_i &= l_{i-1} \cdot e^{m_{i-1} - m_i} + e^{x_i - m_i} \\
&= \sum_{j=0}^{i-1} e^{x_j - m_{i-1}} \cdot e^{m_{i-1} - m_i} + e^{x_i - m_i} \\
&= \sum_{j=0}^{i-1} e^{x_j - m_i} + e^{x_i - m_i} \\
&= \sum_{j=0}^{i} e^{x_j - m_i}
\end{aligned}
$$

证明完毕！✓

---

## 4. 块级Online Softmax

### 4.1 扩展到块处理

实际中，我们按**块**处理数据，而不是逐元素处理：

```
状态: (m_old, l_old)  +  新块数据  →  更新状态: (m_new, l_new)
```

### 4.2 块更新公式

给定当前状态 $(m_{old}, l_{old})$ 和新块的统计量 $(m_{block}, l_{block})$：

$$
m_{new} = \max(m_{old}, m_{block})
$$

$$
l_{new} = e^{m_{old} - m_{new}} \cdot l_{old} + e^{m_{block} - m_{new}} \cdot l_{block}
$$

其中：
- $m_{block} = \max(\text{block元素})$
- $l_{block} = \sum_{x \in \text{block}} e^{x - m_{block}}$

### 4.3 图解

```
┌─────────────────────────────────────────────────────────────────┐
│                     块级Online Softmax                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Block 1: [x₁, x₂, x₃]                                         │
│   ├── m₁ = max(x₁, x₂, x₃)                                      │
│   └── l₁ = exp(x₁-m₁) + exp(x₂-m₁) + exp(x₃-m₁)                │
│                                                                  │
│                      ↓                                           │
│                                                                  │
│   Block 2: [x₄, x₅, x₆]                                         │
│   ├── m₂ = max(x₄, x₅, x₆)                                      │
│   └── l₂ = exp(x₄-m₂) + exp(x₅-m₂) + exp(x₆-m₂)                │
│                                                                  │
│   合并:                                                          │
│   ├── m_new = max(m₁, m₂)                                       │
│   └── l_new = exp(m₁-m_new)·l₁ + exp(m₂-m_new)·l₂              │
│                                                                  │
│                      ↓                                           │
│                                                                  │
│   Block 3: [x₇, x₈, x₉]  ...                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. 在Attention中的应用

### 5.1 带输出的Online Softmax

在Attention中，我们不仅要计算Softmax权重，还要计算加权输出：

$$
O = \text{softmax}(S) \cdot V = P \cdot V
$$

**问题：** 如何在分块计算时正确更新输出 $O$？

### 5.2 完整的更新公式

状态变量：$(m, l, O)$

给定新块的数据，更新规则为：

$$
m_{new} = \max(m_{old}, m_{block})
$$

$$
l_{new} = e^{m_{old} - m_{new}} \cdot l_{old} + e^{m_{block} - m_{new}} \cdot l_{block}
$$

$$
O_{new} = e^{m_{old} - m_{new}} \cdot O_{old} + e^{m_{block} - m_{new}} \cdot O_{block}
$$

其中 $O_{block} = P_{block} \cdot V_{block}$ 是块的局部输出（使用局部Softmax权重）。

### 5.3 最终归一化

处理完所有块后，最终输出需要归一化：

$$
O_{final} = \frac{O_{accumulated}}{l_{final}}
$$

---

## 6. 实现细节

### 6.1 伪代码

```python
def flash_attention_forward(Q, K, V, B_c):
    """
    FlashAttention前向传播（简化版）
    
    B_c: K/V的块大小
    """
    N, d = Q.shape
    
    # 初始化输出状态
    O = zeros(N, d)
    l = zeros(N)
    m = full(N, -inf)
    
    # 按块处理K和V
    for j in range(0, N, B_c):
        K_j = K[j:j+B_c]
        V_j = V[j:j+B_c]
        
        # 计算当前块的注意力分数
        S_j = Q @ K_j.T  # (N, B_c)
        
        # 块内统计量
        m_j = S_j.max(dim=-1)  # (N,)
        P_j = exp(S_j - m_j)    # 未归一化的权重
        l_j = P_j.sum(dim=-1)   # (N,)
        O_j = P_j @ V_j         # 块输出 (N, d)
        
        # 更新全局状态
        m_new = maximum(m, m_j)
        l = exp(m - m_new) * l + exp(m_j - m_new) * l_j
        O = exp(m - m_new) * O + exp(m_j - m_new) * O_j
        m = m_new
    
    # 最终归一化
    O = O / l.unsqueeze(-1)
    
    return O
```

### 6.2 数值稳定性考虑

1. **初始化 m 为 -inf**：保证第一个块能正确设置最大值

2. **缩放因子的计算**：
   - 当 $m_{old} = m_{new}$ 时，$e^{m_{old} - m_{new}} = 1$
   - 当 $m_{old} < m_{new}$ 时，$e^{m_{old} - m_{new}} < 1$（缩小旧结果）
   - 这保证了所有exp运算的参数都 ≤ 0

3. **避免除以零**：$l$ 始终 > 0（因为至少有一个 $e^0 = 1$）

---

## 7. 总结

### 核心要点

1. **Online Softmax的关键突破**
   - 同时维护最大值 $m$ 和累加和 $l$
   - 当最大值更新时，缩放之前的累加和
   - 实现单遍遍历计算Softmax

2. **块级扩展**
   - 计算每个块的局部统计量 $(m_{block}, l_{block})$
   - 与全局状态合并
   - 支持分块处理

3. **在Attention中的应用**
   - 额外维护输出状态 $O$
   - 同样使用缩放因子更新
   - 最后进行归一化

### 关键公式

$$
\begin{aligned}
m_{new} &= \max(m_{old}, m_{block}) \\
l_{new} &= e^{m_{old} - m_{new}} \cdot l_{old} + e^{m_{block} - m_{new}} \cdot l_{block} \\
O_{new} &= e^{m_{old} - m_{new}} \cdot O_{old} + e^{m_{block} - m_{new}} \cdot O_{block}
\end{aligned}
$$

---

**下一节：** [核心公式推导](../3_核心公式推导/) - 完整的数学推导过程


