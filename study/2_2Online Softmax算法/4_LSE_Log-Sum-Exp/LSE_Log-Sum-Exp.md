# LSE (Log-Sum-Exp)

> 理解Log-Sum-Exp在FlashAttention中的作用与高效计算

---

## 1. LSE的定义

### 1.1 基本定义

**Log-Sum-Exp (LSE)** 函数定义为：

$$
\text{LSE}(x_1, x_2, ..., x_n) = \log\left(\sum_{i=1}^{n} e^{x_i}\right)
$$

也可以写作：
$$
\text{LSE}(\mathbf{x}) = \log\left(\|\mathbf{e^x}\|_1\right)
$$

### 1.2 与Softmax的关系

Softmax可以用LSE表示：
$$
\text{softmax}(x)_i = e^{x_i - \text{LSE}(\mathbf{x})}
$$

**证明：**
$$
e^{x_i - \text{LSE}(\mathbf{x})} = e^{x_i - \log(\sum_j e^{x_j})} = \frac{e^{x_i}}{\sum_j e^{x_j}} = \text{softmax}(x)_i
$$

---

## 2. 数值稳定的LSE

### 2.1 直接计算的问题

```python
# 危险的实现！
def naive_lse(x):
    return np.log(np.sum(np.exp(x)))  # exp可能溢出
```

当 $x$ 中有大值时，$e^{x_i}$ 会溢出。

### 2.2 稳定的计算方法

**关键技巧：** 先减去最大值

$$
\text{LSE}(\mathbf{x}) = m + \log\left(\sum_{i} e^{x_i - m}\right)
$$

其中 $m = \max_i(x_i)$

**证明：**
$$
\begin{aligned}
\log\left(\sum_{i} e^{x_i}\right) &= \log\left(\sum_{i} e^{x_i - m} \cdot e^{m}\right) \\
&= \log\left(e^{m} \sum_{i} e^{x_i - m}\right) \\
&= m + \log\left(\sum_{i} e^{x_i - m}\right)
\end{aligned}
$$

### 2.3 稳定实现

```python
def stable_lse(x):
    """数值稳定的Log-Sum-Exp"""
    m = np.max(x)
    return m + np.log(np.sum(np.exp(x - m)))
```

**为什么稳定：**
- $x_i - m \leq 0$，所以 $e^{x_i - m} \leq 1$
- 求和结果 ≥ 1（因为有一项是 $e^0 = 1$）
- 对数的参数始终 ≥ 1，不会有数值问题

---

## 3. LSE在FlashAttention中的作用

### 3.1 存储Softmax统计量

在反向传播中，FlashAttention需要重新计算Softmax。为此，需要存储足够的信息。

**选择1：存储 $(m, l)$**
- $m$：最大值
- $l$：累加和 $l = \sum_i e^{x_i - m}$

**选择2：存储 LSE**
$$
\text{LSE} = m + \log(l)
$$

**优势：** LSE将两个量压缩为一个量！

### 3.2 从LSE恢复Softmax

给定 $x_i$ 和 $\text{LSE}$，可以计算：
$$
\text{softmax}(x)_i = e^{x_i - \text{LSE}}
$$

### 3.3 在反向传播中的应用

```
前向传播:
1. 计算 S = QK^T
2. 计算每行的 LSE
3. 存储 LSE（而不是完整的P矩阵）

反向传播:
1. 重新计算 S = QK^T
2. 使用存储的 LSE 恢复 P = softmax(S)
3. 计算梯度
```

**内存节省：**
- 存储 $P$：$O(N^2)$ 内存
- 存储 LSE：$O(N)$ 内存

---

## 4. LSE的Online更新

### 4.1 两块LSE的合并

给定两块的LSE：
- 块1：$\text{LSE}_1 = m_1 + \log(l_1)$
- 块2：$\text{LSE}_2 = m_2 + \log(l_2)$

**目标：** 计算全局的 $\text{LSE}_{1:2}$

### 4.2 推导

$$
\begin{aligned}
\text{LSE}_{1:2} &= \log\left(\sum_{\text{所有} i} e^{x_i}\right) \\
&= \log\left(\sum_{i \in \text{块1}} e^{x_i} + \sum_{i \in \text{块2}} e^{x_i}\right) \\
&= \log\left(e^{\text{LSE}_1} + e^{\text{LSE}_2}\right) \\
&= \text{LSE}(\text{LSE}_1, \text{LSE}_2)
\end{aligned}
$$

**结论：** 两块LSE的合并就是对两个LSE值再做一次LSE！

### 4.3 数值稳定的合并

```python
def merge_lse(lse1, lse2):
    """合并两个LSE值"""
    m = max(lse1, lse2)
    return m + np.log(np.exp(lse1 - m) + np.exp(lse2 - m))
```

### 4.4 递推公式

更一般地：
$$
\text{LSE}_{1:k} = \text{LSE}(\text{LSE}_{1:k-1}, \text{LSE}^{(k)})
$$

---

## 5. LSE与在线算法的关系

### 5.1 从 $(m, l)$ 到 LSE

$$
\text{LSE} = m + \log(l)
$$

### 5.2 从 LSE 到 $(m, l)$

这是**信息有损**的！
- 从单个LSE值无法唯一确定 $(m, l)$
- 但可以选择一个特定的表示，例如令 $m = \text{LSE}$，则 $l = 1$

### 5.3 在实际实现中的选择

FlashAttention代码中可能：

**方案A：直接存储 $(m, l)$**
```cpp
// 优点：更新计算简单
// 缺点：需要两个数
float m = rowMax;
float l = sumExp;
```

**方案B：存储 LSE**
```cpp
// 优点：只需一个数
// 缺点：合并时需要额外计算
float lse = m + log(l);
```

实际代码通常使用方案A进行计算，最后转换为LSE存储。

---

## 6. LSE的梯度

### 6.1 梯度公式

$$
\frac{\partial \text{LSE}(\mathbf{x})}{\partial x_i} = \frac{e^{x_i}}{\sum_j e^{x_j}} = \text{softmax}(\mathbf{x})_i
$$

**意义：** LSE对输入的梯度就是Softmax！

### 6.2 证明

$$
\begin{aligned}
\frac{\partial}{\partial x_i} \log\left(\sum_j e^{x_j}\right) 
&= \frac{1}{\sum_j e^{x_j}} \cdot \frac{\partial}{\partial x_i} \sum_j e^{x_j} \\
&= \frac{1}{\sum_j e^{x_j}} \cdot e^{x_i} \\
&= \text{softmax}(\mathbf{x})_i
\end{aligned}
$$

### 6.3 在反向传播中的应用

这个性质使得LSE成为Softmax的"对数空间"表示，有利于梯度计算。

---

## 7. 代码实现示例

### 7.1 基本实现

```python
import numpy as np

def lse(x, axis=None, keepdims=False):
    """
    数值稳定的Log-Sum-Exp
    
    Args:
        x: 输入数组
        axis: 沿哪个轴计算
        keepdims: 是否保持维度
    
    Returns:
        LSE值
    """
    m = np.max(x, axis=axis, keepdims=True)
    result = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    
    if not keepdims:
        result = np.squeeze(result, axis=axis)
    
    return result

def softmax_from_lse(x, lse_val):
    """使用LSE计算Softmax"""
    return np.exp(x - lse_val)
```

### 7.2 Online LSE更新

```python
def online_lse_update(lse_old, x_new):
    """
    在线更新LSE
    
    Args:
        lse_old: 之前的LSE值
        x_new: 新的数据块
    
    Returns:
        更新后的LSE
    """
    lse_new = lse(x_new)
    return merge_lse(lse_old, lse_new)

def merge_lse(lse1, lse2):
    """合并两个LSE"""
    m = np.maximum(lse1, lse2)
    return m + np.log(np.exp(lse1 - m) + np.exp(lse2 - m))
```

### 7.3 FlashAttention中的使用

```python
def flash_attention_with_lse(Q, K, V, block_size):
    """
    FlashAttention，保存LSE用于反向传播
    """
    seq_len, d = Q.shape
    
    # 初始化
    O_tilde = np.zeros((seq_len, d))
    m = np.full(seq_len, -np.inf)
    l = np.zeros(seq_len)
    
    # 按块处理
    for j in range(0, seq_len, block_size):
        K_j = K[j:min(j+block_size, seq_len)]
        V_j = V[j:min(j+block_size, seq_len)]
        
        S_j = Q @ K_j.T
        m_j = S_j.max(axis=1)
        P_j = np.exp(S_j - m_j[:, None])
        l_j = P_j.sum(axis=1)
        O_j = P_j @ V_j
        
        # 更新
        m_new = np.maximum(m, m_j)
        alpha = np.exp(m - m_new)
        beta = np.exp(m_j - m_new)
        l = alpha * l + beta * l_j
        O_tilde = alpha[:, None] * O_tilde + beta[:, None] * O_j
        m = m_new
    
    # 计算LSE用于存储
    lse = m + np.log(l)
    
    # 归一化输出
    O = O_tilde / l[:, None]
    
    return O, lse  # 返回LSE供反向传播使用
```

---

## 8. 总结

### 关键公式

| 公式 | 描述 |
|------|------|
| $\text{LSE}(\mathbf{x}) = \log(\sum_i e^{x_i})$ | LSE定义 |
| $\text{LSE}(\mathbf{x}) = m + \log(\sum_i e^{x_i - m})$ | 稳定计算 |
| $\text{softmax}(x)_i = e^{x_i - \text{LSE}}$ | 从LSE计算Softmax |
| $\text{LSE}_{1:2} = \text{LSE}(\text{LSE}_1, \text{LSE}_2)$ | LSE合并 |
| $\partial \text{LSE} / \partial x_i = \text{softmax}(x)_i$ | LSE梯度 |

### LSE的优势

1. **紧凑存储**：一个标量代替 $(m, l)$ 两个量
2. **代数性质好**：合并操作简单（对LSE值再做LSE）
3. **与Softmax的美妙关系**：LSE的梯度就是Softmax

### 在FlashAttention中的角色

1. **前向传播**：计算并存储每行的LSE
2. **反向传播**：使用LSE重计算Softmax矩阵
3. **内存效率**：$O(N)$ 替代 $O(N^2)$

---

**本章完成！** 你现在应该对Online Softmax算法有了深入的理解。

**回顾：** [章节概述](../README.md)


