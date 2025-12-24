# Tiling分块技术

> 将大矩阵分成小块，在高速SRAM中完成计算

---

## 1. 为什么需要Tiling

### 1.1 问题回顾

标准Attention需要存储 N×N 的中间矩阵（S 和 P），当序列长度 N 很大时：

```
N = 4096, d = 64:
  - S矩阵: 4096 × 4096 × 2 bytes = 32 MB
  - P矩阵: 4096 × 4096 × 2 bytes = 32 MB
  - 总计: 64 MB

GPU SRAM (共享内存) 容量:
  - A100: ~192 KB / SM
  - 整个GPU: ~20 MB

结论: 64 MB >> 20 MB，无法将完整矩阵放入SRAM！
```

### 1.2 Tiling的核心思想

**Tiling（分块）** 是将大矩阵分成小块，每次只处理一小块：

```
┌─────────────────────────────────────────────────────────────────┐
│                      Tiling的核心思想                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   原始矩阵 (N×N)                    分块处理                     │
│   ┌───────────────┐                 ┌───┬───┬───┬───┐           │
│   │               │                 │ 1 │ 2 │ 3 │ 4 │           │
│   │   太大！      │    ────→        ├───┼───┼───┼───┤           │
│   │   放不进      │                 │ 5 │ 6 │ 7 │ 8 │           │
│   │   SRAM       │                 ├───┼───┼───┼───┤           │
│   │               │                 │ 9 │10 │11 │12 │           │
│   └───────────────┘                 ├───┼───┼───┼───┤           │
│                                     │13 │14 │15 │16 │           │
│                                     └───┴───┴───┴───┘           │
│                                                                  │
│   每个小块可以完整放入SRAM，在SRAM中完成所有计算！               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. FlashAttention的分块策略

### 2.1 Q、K、V的分块

FlashAttention将 Q、K、V 分成多个块：

```
Q 分块:
┌─────────────────┐
│    Q_1 (Br×d)   │  ← 第1块
├─────────────────┤
│    Q_2 (Br×d)   │  ← 第2块
├─────────────────┤
│    Q_3 (Br×d)   │  ← 第3块
├─────────────────┤
│       ...       │
├─────────────────┤
│    Q_Tr (Br×d)  │  ← 第Tr块 (Tr = N/Br)
└─────────────────┘

K, V 分块类似，每块大小 Bc × d
Tc = N / Bc (K,V的块数)
```

### 2.2 块大小的选择

块大小受限于SRAM容量：

```
需要同时放入SRAM的数据:
  - Q块: Br × d 个元素
  - K块: Bc × d 个元素
  - V块: Bc × d 个元素
  - S块 (局部): Br × Bc 个元素
  - O块 (累加器): Br × d 个元素
  - m, l (softmax统计量): Br 个元素

总内存约: (Br×d + 2×Bc×d + Br×Bc + Br×d + 2×Br) × bytes_per_element

SRAM容量约束:
  总内存 ≤ SRAM大小 (如 96KB)
```

### 2.3 典型块大小

| GPU架构 | SRAM/SM | 推荐 Br | 推荐 Bc |
|---------|---------|---------|---------|
| Ampere (A100) | ~164 KB | 64-128 | 64-128 |
| Hopper (H100) | ~228 KB | 128-256 | 64-128 |

---

## 3. 外循环与内循环

### 3.1 双层循环结构

FlashAttention使用**外循环遍历Q块，内循环遍历K/V块**的策略：

```python
# FlashAttention前向传播伪代码

# 将Q分成Tr块，K/V分成Tc块
Tr = ceil(N / Br)
Tc = ceil(N / Bc)

# 外循环：遍历Q的每一块
for i in range(Tr):
    # 从HBM加载Q的第i块到SRAM
    Q_i = load_from_HBM(Q[i*Br : (i+1)*Br, :])
    
    # 初始化输出累加器和softmax统计量
    O_i = zeros(Br, d)      # 在SRAM中
    m_i = -inf(Br)          # 每行的最大值
    l_i = zeros(Br)         # 每行的指数和
    
    # 内循环：遍历K/V的每一块
    for j in range(Tc):
        # 从HBM加载K,V的第j块到SRAM
        K_j = load_from_HBM(K[j*Bc : (j+1)*Bc, :])
        V_j = load_from_HBM(V[j*Bc : (j+1)*Bc, :])
        
        # 在SRAM中计算局部attention
        S_ij = Q_i @ K_j.T / sqrt(d)  # Br × Bc，在SRAM中
        
        # Online Softmax更新（下一节详解）
        m_ij = max(S_ij, axis=1)
        m_new = max(m_i, m_ij)
        
        P_ij = exp(S_ij - m_new)
        l_new = exp(m_i - m_new) * l_i + sum(P_ij, axis=1)
        
        # 更新输出
        O_i = exp(m_i - m_new) * O_i + P_ij @ V_j
        
        m_i = m_new
        l_i = l_new
    
    # 最终归一化
    O_i = O_i / l_i
    
    # 写回HBM
    store_to_HBM(O[i*Br : (i+1)*Br, :], O_i)
```

### 3.2 循环顺序的重要性

```
外循环Q，内循环K/V 的优点:
═══════════════════════════════════════════════════════════════

1. 输出O可以在内循环中累积，最后一次性写回HBM
   → 减少O的HBM写入次数

2. 每次外循环只需读取一次Q块
   → Q只被读取一次

3. K/V在内循环中被多个Q块共享（如果有多个Q块同时处理）
   → 可以利用K/V的数据重用

═══════════════════════════════════════════════════════════════
```

### 3.3 数据流图

```
┌─────────────────────────────────────────────────────────────────┐
│                   FlashAttention数据流                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   HBM                                                            │
│   ┌────┐ ┌────┐ ┌────┐                              ┌────┐     │
│   │ Q  │ │ K  │ │ V  │                              │ O  │     │
│   └─┬──┘ └─┬──┘ └─┬──┘                              └─▲──┘     │
│     │      │      │                                   │         │
│     │ 读   │ 读   │ 读                            写  │         │
│     ▼      ▼      ▼                                   │         │
│   ═══════════════════════════════════════════════════════════   │
│                         SRAM                                     │
│   ┌────────────────────────────────────────────────────────┐    │
│   │                                                         │    │
│   │   Q_i ──────┐                                          │    │
│   │             │                                          │    │
│   │   K_j ──────┼──→ S_ij = Q_i @ K_j.T                   │    │
│   │             │           │                              │    │
│   │             │           ▼                              │    │
│   │             │      Online Softmax                      │    │
│   │             │           │                              │    │
│   │   V_j ──────┼───────────┼──→ O_i (累加) ──────────────┼────┘
│   │             │           │                              │    │
│   │   m_i, l_i ◄────────────┘                              │    │
│   │                                                         │    │
│   └────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. 分块计算的数学保证

### 4.1 问题：如何分块计算Softmax？

标准Softmax需要全局信息：

$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{N} e^{x_j}}
$$

**问题：** 分块计算时，我们只能看到部分数据，如何计算正确的softmax？

**答案：** Online Softmax（在线Softmax）算法！

### 4.2 Online Softmax关键

```
分块处理时的挑战:
═══════════════════════════════════════════════════════════════

处理第1块K后:  S_1 = Q @ K_1.T → P_1 = softmax(S_1)  ❌ 不完整！
处理第2块K后:  S_2 = Q @ K_2.T → 需要结合S_1和S_2重新计算softmax

标准做法: 存储所有S，最后一起计算softmax
         → 需要O(N²)存储空间！

Online Softmax: 边处理边更新，无需存储完整S矩阵
         → 只需O(N)空间存储统计量(m, l)

═══════════════════════════════════════════════════════════════
```

### 4.3 分块输出的正确性

FlashAttention保证最终输出与标准Attention完全相同：

$$
O = \text{softmax}(QK^T)V = \frac{\sum_j \exp(S_{:,j}) \cdot V_j}{\sum_j \exp(S_{:,j})}
$$

通过维护运行时的最大值 m 和累加和 l，可以增量地计算这个结果。

---

## 5. 分块矩阵乘法示例

### 5.1 经典分块矩阵乘法

在理解FlashAttention的分块之前，先看经典的分块矩阵乘法：

```
C = A × B

将A分成行块，B分成列块:

A = [A_1]    B = [B_1, B_2, B_3]
    [A_2]
    [A_3]

C_ij = A_i × B_j

分块计算:
for i in range(num_row_blocks):
    for j in range(num_col_blocks):
        C[i,j] = A[i] @ B[j]
```

### 5.2 FlashAttention的分块特殊性

FlashAttention的分块更复杂，因为涉及Softmax：

```
标准分块矩阵乘法:  C_ij = A_i @ B_j (可以独立计算)

FlashAttention:     O = softmax(Q @ K.T) @ V
                       ↑
                    需要全局归一化！

解决方案: Online Softmax允许增量更新归一化因子
```

---

## 6. SRAM利用优化

### 6.1 双缓冲（Double Buffering）

为了隐藏内存延迟，使用双缓冲技术：

```
┌─────────────────────────────────────────────────────────────────┐
│                    双缓冲技术                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   时间 →                                                         │
│   ──────────────────────────────────────────────────────────    │
│                                                                  │
│   Buffer A:  [加载K_1] [计算用K_1] [加载K_3] [计算用K_3] ...    │
│   Buffer B:  [  闲置 ] [加载K_2]   [计算用K_2] [加载K_4] ...    │
│                                                                  │
│   效果: 加载下一块数据的同时，计算当前块                         │
│         → 计算和内存访问可以重叠！                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 寄存器分配

对于频繁访问的数据，尽量放在寄存器中：

```
寄存器级数据（最快）:
  - m_i, l_i: softmax统计量
  - 部分累加结果

共享内存级数据:
  - Q_i, K_j, V_j: 输入块
  - S_ij: 局部注意力分数
  - O_i: 输出累加器
```

---

## 7. 块大小与性能

### 7.1 块大小的权衡

| 因素 | 大块 | 小块 |
|------|------|------|
| SRAM利用 | 需要更多SRAM | 需要更少SRAM |
| 循环次数 | 更少迭代 | 更多迭代 |
| 并行度 | 每块更多并行 | 更多块可并行 |
| 矩阵乘法效率 | 更高效率 | 较低效率 |

### 7.2 最优块大小

```
经验法则:
═══════════════════════════════════════════════════════════════

Br, Bc ∈ {32, 64, 128, 256}

约束条件:
1. SRAM容量: (Br×d + 2×Bc×d + Br×Bc) × 2 ≤ SRAM_size
2. 矩阵乘法效率: Br, Bc 应是32的倍数（Tensor Core友好）
3. 占用率: 不要使用过多共享内存导致低占用率

实践中:
- A100上常用 Br=Bc=64 或 Br=Bc=128
- 可以根据序列长度动态调整

═══════════════════════════════════════════════════════════════
```

---

## 8. 代码示例：分块Attention

### 8.1 Python实现（简化版）

```python
import torch
import torch.nn.functional as F

def flash_attention_forward(Q, K, V, Br=64, Bc=64):
    """
    FlashAttention前向传播的简化Python实现
    
    Args:
        Q, K, V: [batch, seq_len, d]
        Br: Q的块大小
        Bc: K/V的块大小
    """
    batch, N, d = Q.shape
    scale = 1.0 / (d ** 0.5)
    
    # 计算块数
    Tr = (N + Br - 1) // Br
    Tc = (N + Bc - 1) // Bc
    
    # 初始化输出
    O = torch.zeros_like(Q)
    
    # 外循环：遍历Q的块
    for i in range(Tr):
        # 获取Q的第i块
        q_start = i * Br
        q_end = min((i + 1) * Br, N)
        Q_i = Q[:, q_start:q_end, :]  # [batch, Br, d]
        
        # 初始化softmax统计量
        m_i = torch.full((batch, q_end - q_start), float('-inf'))
        l_i = torch.zeros(batch, q_end - q_start)
        O_i = torch.zeros(batch, q_end - q_start, d)
        
        # 内循环：遍历K/V的块
        for j in range(Tc):
            # 获取K/V的第j块
            kv_start = j * Bc
            kv_end = min((j + 1) * Bc, N)
            K_j = K[:, kv_start:kv_end, :]  # [batch, Bc, d]
            V_j = V[:, kv_start:kv_end, :]  # [batch, Bc, d]
            
            # 计算局部注意力分数
            S_ij = torch.matmul(Q_i, K_j.transpose(-2, -1)) * scale
            
            # Online Softmax更新
            m_ij = S_ij.max(dim=-1).values
            m_new = torch.maximum(m_i, m_ij)
            
            # 更新输出
            exp_m_diff = torch.exp(m_i - m_new).unsqueeze(-1)
            P_ij = torch.exp(S_ij - m_new.unsqueeze(-1))
            l_new = torch.exp(m_i - m_new) * l_i + P_ij.sum(dim=-1)
            
            O_i = exp_m_diff * O_i + torch.matmul(P_ij, V_j)
            
            m_i = m_new
            l_i = l_new
        
        # 最终归一化
        O_i = O_i / l_i.unsqueeze(-1)
        
        # 写回输出
        O[:, q_start:q_end, :] = O_i
    
    return O
```

---

## 9. 总结

### 9.1 关键要点

| 概念 | 说明 |
|------|------|
| **Tiling** | 将大矩阵分成小块，放入SRAM计算 |
| **块大小** | 受SRAM容量约束，通常64-128 |
| **外循环** | 遍历Q的块 |
| **内循环** | 遍历K/V的块 |
| **Online Softmax** | 允许分块计算正确的softmax |
| **双缓冲** | 重叠计算和内存访问 |

### 9.2 Tiling的优势

```
┌─────────────────────────────────────────────────────────────────┐
│                      Tiling的优势                                │
│                                                                  │
│   1. 显存节省                                                    │
│      - 不需要存储完整的N×N矩阵                                   │
│      - 只需要O(Br×Bc)的SRAM空间                                 │
│                                                                  │
│   2. HBM访问减少                                                 │
│      - 中间结果在SRAM中处理                                      │
│      - 减少HBM读写次数                                          │
│                                                                  │
│   3. 计算效率提高                                                │
│      - SRAM带宽是HBM的10倍                                      │
│      - 更多时间花在计算而非等待数据                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 10. 关键术语

| 术语 | 英文 | 含义 |
|------|------|------|
| 分块 | Tiling | 将矩阵分成小块处理 |
| 块大小 | Block Size | Br×Bc |
| 外循环 | Outer Loop | 遍历Q块 |
| 内循环 | Inner Loop | 遍历K/V块 |
| 双缓冲 | Double Buffering | 重叠IO和计算 |
| 在线算法 | Online Algorithm | 增量处理数据流 |

---

## 📚 延伸阅读

- [FlashAttention论文 Algorithm 1](https://arxiv.org/abs/2205.14135)：完整算法描述
- [Tiled Matrix Multiplication](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/)：CUDA分块矩阵乘法
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)：改进的分块策略


