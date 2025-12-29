# 标准Softmax数值稳定性

> 理解Softmax的计算过程及其数值稳定性问题

---

## 1. Softmax函数回顾

### 1.1 定义

Softmax函数将一个实数向量转换为概率分布：

$$
\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}
$$

**性质：**
- 输出值在 $(0, 1)$ 之间
- 所有输出值之和为 1
- 保持相对大小顺序（单调性）

### 1.2 为什么需要Softmax

在Attention机制中，Softmax用于将注意力分数转换为注意力权重：

```
Q × K^T → 注意力分数 S → Softmax(S) → 注意力权重 P → P × V → 输出
```

Softmax确保：
1. 每个位置的权重都是正数
2. 权重之和为1（概率解释）
3. 较大的分数获得较大的权重

---

## 2. 直接计算的问题

### 2.1 溢出问题

直接按定义计算Softmax会导致数值问题：

```python
# 危险的实现！
def naive_softmax(x):
    exp_x = np.exp(x)           # 可能溢出！
    return exp_x / np.sum(exp_x)
```

**问题分析：**

1. **上溢（Overflow）**：当 $x_i$ 很大时，$e^{x_i}$ 会超出浮点数范围
   ```
   np.exp(1000) → inf
   ```

2. **下溢（Underflow）**：当 $x_i$ 很小（负数）时，$e^{x_i}$ 趋近于0
   ```
   np.exp(-1000) → 0
   ```

3. **NaN传播**：`inf / inf = NaN`

### 2.2 实际例子

```python
import numpy as np

# 正常情况
x1 = np.array([1.0, 2.0, 3.0])
print(np.exp(x1) / np.sum(np.exp(x1)))  # [0.09, 0.24, 0.67] ✓

# 大数情况
x2 = np.array([1000.0, 1001.0, 1002.0])
print(np.exp(x2))  # [inf, inf, inf] ✗
print(np.exp(x2) / np.sum(np.exp(x2)))  # [nan, nan, nan] ✗
```

在实际应用中，这种情况很容易发生：
- Attention分数 $QK^T$ 可能很大
- 序列越长，问题越严重

---

## 3. 数值稳定的Softmax

### 3.1 减去最大值技巧

**核心思想：** 利用Softmax的平移不变性

$$
\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}} = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}}
$$

其中 $m = \max_j(x_j)$

**证明：**
$$
\frac{e^{x_i - m}}{\sum_j e^{x_j - m}} = \frac{e^{x_i} \cdot e^{-m}}{\sum_j e^{x_j} \cdot e^{-m}} = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$

### 3.2 稳定实现

```python
def stable_softmax(x):
    """数值稳定的Softmax实现"""
    m = np.max(x)           # 找最大值
    exp_x = np.exp(x - m)   # 减去最大值后再exp
    l = np.sum(exp_x)       # 求和
    return exp_x / l
```

**为什么有效：**
- 减去最大值后，最大的元素变为 0
- $e^0 = 1$，不会溢出
- 其他元素都是负数，$e^{负数} < 1$，也不会溢出

### 3.3 对比验证

```python
# 大数情况
x = np.array([1000.0, 1001.0, 1002.0])

# 不稳定版本
naive_result = np.exp(x) / np.sum(np.exp(x))  # [nan, nan, nan]

# 稳定版本
stable_result = stable_softmax(x)  # [0.09, 0.24, 0.67] ✓

# 验证：与较小数字的结果一致
x_small = x - 1000  # [0, 1, 2]
print(stable_softmax(x_small))  # [0.09, 0.24, 0.67] ✓
```

---

## 4. 标准Softmax的计算流程

### 4.1 三遍遍历

标准的数值稳定Softmax需要**三次遍历**数据：

```
┌─────────────────────────────────────────────────────────────────┐
│                  标准Softmax计算流程                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   输入: x = [x₁, x₂, x₃, ..., xₙ]                               │
│                                                                  │
│   ┌──────────────────────────────────────────────┐              │
│   │ Pass 1: 求最大值                              │              │
│   │ m = max(x₁, x₂, ..., xₙ)                     │              │
│   │ 遍历所有元素                                  │              │
│   └──────────────────────────────────────────────┘              │
│                       ↓                                          │
│   ┌──────────────────────────────────────────────┐              │
│   │ Pass 2: 计算归一化因子                        │              │
│   │ l = Σᵢ exp(xᵢ - m)                           │              │
│   │ 遍历所有元素                                  │              │
│   └──────────────────────────────────────────────┘              │
│                       ↓                                          │
│   ┌──────────────────────────────────────────────┐              │
│   │ Pass 3: 计算输出                              │              │
│   │ yᵢ = exp(xᵢ - m) / l                         │              │
│   │ 遍历所有元素                                  │              │
│   └──────────────────────────────────────────────┘              │
│                       ↓                                          │
│   输出: y = [y₁, y₂, y₃, ..., yₙ]                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 伪代码

```python
def standard_softmax(x):
    n = len(x)
    
    # Pass 1: 找最大值
    m = x[0]
    for i in range(1, n):
        m = max(m, x[i])
    
    # Pass 2: 计算归一化因子
    l = 0
    for i in range(n):
        l += exp(x[i] - m)
    
    # Pass 3: 计算输出
    y = []
    for i in range(n):
        y.append(exp(x[i] - m) / l)
    
    return y
```

### 4.3 问题：无法分块处理

**关键限制：** 第一遍遍历必须看完所有数据才能确定最大值。

这对FlashAttention造成问题：
- FlashAttention按块处理Q、K、V
- 处理第一个块时，不知道后面块的数据
- 无法提前计算正确的最大值 $m$

```
Block 1    Block 2    Block 3    ...    Block K
[x₁...x_b] [...]      [...]              [...]
   ↓                                       
 m₁ = max(Block 1)    ← 这不是全局最大值！
```

**这就是为什么我们需要Online Softmax！**

---

## 5. Safe Softmax in Attention

### 5.1 Attention中的应用

在Attention中，Softmax作用于分数矩阵 $S = QK^T / \sqrt{d}$：

```python
def attention(Q, K, V):
    d = Q.shape[-1]
    S = Q @ K.T / sqrt(d)
    P = softmax(S, dim=-1)  # 对每一行做Softmax
    O = P @ V
    return O
```

### 5.2 逐行处理

Softmax是**逐行独立**的：

```
S = [s₁₁, s₁₂, ..., s₁ₙ]    →  P = [p₁₁, p₁₂, ..., p₁ₙ]  ← 第1行独立
    [s₂₁, s₂₂, ..., s₂ₙ]        [p₂₁, p₂₂, ..., p₂ₙ]  ← 第2行独立
    [...]                        [...]
    [sₘ₁, sₘ₂, ..., sₘₙ]        [pₘ₁, pₘ₂, ..., pₘₙ]  ← 第m行独立
```

每一行需要：
- 该行的最大值 $m^{(i)}$
- 该行的归一化因子 $l^{(i)}$

### 5.3 内存问题

标准方法需要：
1. 完整计算 $S = QK^T$（O(N²) 内存）
2. 存储 $S$ 以计算 Softmax
3. 存储 $P$ 以计算 $PV$

这就是为什么标准Attention需要 O(N²) 显存！

---

## 6. 总结

### 核心要点

1. **直接计算Softmax有数值稳定性问题**
   - 大数导致上溢
   - 小数导致下溢
   
2. **解决方案：减去最大值**
   - 利用Softmax的平移不变性
   - 确保指数运算不会溢出

3. **标准实现需要三次遍历**
   - 第一遍：找最大值
   - 第二遍：计算归一化因子
   - 第三遍：计算输出

4. **FlashAttention的挑战**
   - 需要分块处理，但标准Softmax需要完整数据
   - 解决方案：Online Softmax（下一节）

### 关键公式

$$
\text{softmax}(x_i) = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}}, \quad m = \max_j(x_j)
$$

---

**下一节：** [Online Softmax增量计算](../2_Online_Softmax增量计算/) - 如何在不知道完整序列的情况下计算Softmax


