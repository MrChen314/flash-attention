# Recomputation重计算策略

> 用计算换空间：反向传播时重新计算中间结果

---

## 1. 为什么需要Recomputation

### 1.1 反向传播的内存问题

在神经网络训练中，反向传播需要用到前向传播的中间结果：

```
前向传播:
  x → [Layer 1] → a₁ → [Layer 2] → a₂ → ... → [Layer N] → y

反向传播:
  需要 a₁, a₂, ..., aₙ 来计算梯度
  
问题: 需要保存所有中间激活值！
```

### 1.2 Attention中的问题

标准Attention的反向传播需要：

```
前向保存:
  - S = QK^T           # N × N 矩阵
  - P = softmax(S)     # N × N 矩阵

反向需要:
  dL/dV = P^T @ dL/dO          # 需要 P
  dL/dP = dL/dO @ V^T          # 需要 V
  dL/dS = softmax_grad(dL/dP, P)  # 需要 P 和 S
  dL/dQ = dL/dS @ K            # 需要 K
  dL/dK = dL/dS^T @ Q          # 需要 Q

关键: S 和 P 都是 O(N²) 的矩阵！
```

### 1.3 内存占用对比

| 方法 | 前向保存 | 显存占用 |
|------|----------|----------|
| 标准Attention | Q, K, V, S, P, O | O(N² + Nd) |
| FlashAttention | Q, K, V, O, m, l | O(Nd) |

FlashAttention不保存S和P，节省了O(N²)的显存！

---

## 2. Recomputation的基本思想

### 2.1 核心权衡

```
┌─────────────────────────────────────────────────────────────────┐
│                   时间 vs 空间的权衡                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   传统做法:                                                      │
│   ┌────────────────────────────────────────────────────────┐    │
│   │  前向: 计算所有中间结果 → 保存到内存                    │    │
│   │  反向: 直接从内存读取中间结果 → 计算梯度                │    │
│   │                                                         │    │
│   │  优点: 反向传播快                                       │    │
│   │  缺点: 需要O(N²)额外内存                                │    │
│   └────────────────────────────────────────────────────────┘    │
│                                                                  │
│   Recomputation:                                                 │
│   ┌────────────────────────────────────────────────────────┐    │
│   │  前向: 计算中间结果 → 只保存必要信息                    │    │
│   │  反向: 重新计算中间结果 → 计算梯度                      │    │
│   │                                                         │    │
│   │  优点: 只需O(N)额外内存                                 │    │
│   │  缺点: 需要额外计算                                     │    │
│   └────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 为什么可以接受额外计算

```
关键洞察:
═══════════════════════════════════════════════════════════════

1. Attention是内存绑定的，不是计算绑定的
   → 增加计算量不一定增加总时间
   → 如果减少了内存访问，可能反而更快

2. 重计算只发生在反向传播
   → 推理时不需要反向传播
   → 只影响训练速度

3. 显存节省使得可以用更大的batch size
   → 更好的GPU利用率
   → 可能补偿重计算的开销

═══════════════════════════════════════════════════════════════
```

---

## 3. PyTorch中的Checkpoint机制

### 3.1 torch.utils.checkpoint

PyTorch提供了 `checkpoint` 函数来实现Recomputation：

```python
from torch.utils.checkpoint import checkpoint

class AttentionWithCheckpoint(nn.Module):
    def forward(self, q, k, v):
        # 使用checkpoint包装attention计算
        return checkpoint(self._attention, q, k, v)
    
    def _attention(self, q, k, v):
        # 标准attention计算
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, v)
```

### 3.2 Checkpoint的工作原理

```
前向传播 (with checkpoint):
──────────────────────────────────────────────────────────────

1. 保存输入: Q, K, V
2. 执行前向计算: S = QK^T → P = softmax(S) → O = PV
3. 丢弃中间结果: 不保存 S 和 P
4. 返回输出: O

反向传播:
──────────────────────────────────────────────────────────────

1. 从保存的输入 Q, K, V 重新计算:
   S = QK^T
   P = softmax(S)
   
2. 使用重新计算的 S, P 计算梯度:
   dL/dV = P^T @ dL/dO
   dL/dP = dL/dO @ V^T
   ... 等等

──────────────────────────────────────────────────────────────
```

---

## 4. FlashAttention的Recomputation

### 4.1 FlashAttention的保存策略

FlashAttention前向传播只保存：
- **Q, K, V**: 输入（必须保存）
- **O**: 输出（必须保存）
- **m**: 每行的最大值（用于重计算softmax）
- **l**: 每行的指数和（用于重计算softmax）

```
FlashAttention保存的数据:
═══════════════════════════════════════════════════════════════

Q, K, V: 各 N × d → 3Nd 元素
O:       N × d   → Nd 元素
m:       N       → N 元素
l:       N       → N 元素

总计: 4Nd + 2N ≈ O(Nd) 元素

对比标准Attention保存 S, P: 2N² 元素

当 N >> d 时，节省了大量显存！

═══════════════════════════════════════════════════════════════
```

### 4.2 为什么保存m和l

m和l是Online Softmax的统计量，用于在反向传播时正确重建P：

```python
# 反向传播时重建P的某个块
S_ij = Q_i @ K_j.T / sqrt(d)   # 重新计算局部S
P_ij = exp(S_ij - m_i) / l_i   # 使用保存的m和l重建P

# 这样就不需要存储完整的N×N的P矩阵！
```

### 4.3 FlashAttention反向传播

```python
# FlashAttention反向传播伪代码

def flash_attention_backward(dO, Q, K, V, O, m, l):
    """
    Args:
        dO: 输出梯度 [N, d]
        Q, K, V: 输入 [N, d]
        O: 前向输出 [N, d]
        m: 每行最大值 [N]
        l: 每行指数和 [N]
    
    Returns:
        dQ, dK, dV: 输入梯度
    """
    dQ = zeros(N, d)
    dK = zeros(N, d)
    dV = zeros(N, d)
    
    # 与前向类似的分块循环
    for i in range(Tr):
        Q_i = Q[i*Br:(i+1)*Br]
        dO_i = dO[i*Br:(i+1)*Br]
        O_i = O[i*Br:(i+1)*Br]
        m_i = m[i*Br:(i+1)*Br]
        l_i = l[i*Br:(i+1)*Br]
        
        dQ_i = zeros(Br, d)
        
        for j in range(Tc):
            K_j = K[j*Bc:(j+1)*Bc]
            V_j = V[j*Bc:(j+1)*Bc]
            
            # 重新计算 S 和 P
            S_ij = Q_i @ K_j.T / sqrt(d)
            P_ij = exp(S_ij - m_i.unsqueeze(-1)) / l_i.unsqueeze(-1)
            
            # 计算梯度
            dV_j = P_ij.T @ dO_i
            dV[j*Bc:(j+1)*Bc] += dV_j
            
            dP_ij = dO_i @ V_j.T
            
            # Softmax梯度
            dS_ij = P_ij * (dP_ij - (dP_ij * P_ij).sum(-1, keepdim=True))
            
            dQ_i += dS_ij @ K_j / sqrt(d)
            dK[j*Bc:(j+1)*Bc] += dS_ij.T @ Q_i / sqrt(d)
        
        dQ[i*Br:(i+1)*Br] = dQ_i
    
    return dQ, dK, dV
```

---

## 5. 不同Checkpoint粒度

### 5.1 层级Checkpointing

在Transformer中，可以在不同层级应用Checkpointing：

```
粒度级别:
═══════════════════════════════════════════════════════════════

细粒度 (每个操作):
  checkpoint(matmul, Q, K)
  checkpoint(softmax, S)
  checkpoint(matmul, P, V)
  → 最省内存，但重计算开销最大

中粒度 (每个Attention层):
  checkpoint(attention_layer, Q, K, V)
  → 平衡的选择

粗粒度 (每隔几层):
  checkpoint(layers[0:4], x)
  checkpoint(layers[4:8], x)
  → 开销小，但节省内存有限

═══════════════════════════════════════════════════════════════
```

### 5.2 选择性Checkpointing

```python
# 只对显存开销大的层使用checkpoint
class Transformer(nn.Module):
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i % 2 == 0:  # 每隔一层使用checkpoint
                x = checkpoint(layer, x)
            else:
                x = layer(x)
        return x
```

---

## 6. 时间 vs 空间权衡分析

### 6.1 理论分析

```
不使用Checkpointing:
  前向时间: T_fwd
  反向时间: T_bwd
  显存: M_full

使用Checkpointing:
  前向时间: T_fwd
  反向时间: T_bwd + T_fwd (需要重新计算前向)
  显存: M_reduced

权衡:
  额外时间: T_fwd
  节省显存: M_full - M_reduced
```

### 6.2 FlashAttention的实际情况

```
FlashAttention的时间开销:
═══════════════════════════════════════════════════════════════

理论上: 反向需要额外1x前向计算

实际上: 由于以下原因，开销更小

1. 内存绑定 → 减少HBM访问节省的时间 > 额外计算的时间
2. 更好的缓存利用 → 重计算时数据可能还在缓存中
3. 融合kernel → 减少kernel启动开销

实测: FlashAttention反向传播只比标准慢约30-50%，而非2x

═══════════════════════════════════════════════════════════════
```

### 6.3 总体收益

```
标准Attention + 无Checkpoint:
  显存: O(N²)
  可用batch size: 受限

FlashAttention + Recomputation:
  显存: O(N)
  可用batch size: 更大
  → 更好的GPU利用率
  → 实际训练速度可能更快
```

---

## 7. 与Gradient Checkpointing的关系

### 7.1 概念对比

| 术语 | 含义 |
|------|------|
| Recomputation | 通用术语：重新计算而非保存 |
| Gradient Checkpointing | PyTorch的实现方式 |
| Activation Checkpointing | 另一种叫法 |

### 7.2 FlashAttention vs PyTorch Checkpoint

```
PyTorch Checkpoint (通用):
  - 适用于任意计算图
  - 在Python层面工作
  - 保存/恢复整个输入

FlashAttention Recomputation (专用):
  - 专门为Attention优化
  - 在CUDA kernel内部实现
  - 只保存必要的统计量(m, l)
  - 与Tiling紧密结合
```

---

## 8. 实际使用建议

### 8.1 何时使用Recomputation

```
建议使用:
  ✓ 训练长序列模型 (N > 2048)
  ✓ 显存受限的情况
  ✓ 需要大batch size

可能不需要:
  ✗ 推理阶段（不需要反向传播）
  ✗ 短序列（N < 512，O(N²)开销小）
  ✗ 显存充足的情况
```

### 8.2 与其他技术结合

```
FlashAttention + Recomputation + 其他优化:
═══════════════════════════════════════════════════════════════

1. 混合精度训练 (FP16/BF16)
   → 进一步减少显存和提升速度

2. 梯度累积
   → 模拟大batch size

3. 模型并行
   → 处理超大模型

4. ZeRO优化
   → 优化优化器状态的显存

═══════════════════════════════════════════════════════════════
```

---

## 9. 总结

### 9.1 关键要点

| 概念 | 说明 |
|------|------|
| **Recomputation** | 反向时重新计算，而非保存中间结果 |
| **权衡** | 用计算时间换取显存空间 |
| **FlashAttention应用** | 不保存S和P，只保存m和l |
| **显存节省** | 从O(N²)降到O(N) |
| **实际效果** | 由于内存绑定，总时间可能反而更快 |

### 9.2 FlashAttention的完整优化

```
┌─────────────────────────────────────────────────────────────────┐
│                FlashAttention优化技术总结                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   1. Tiling (分块)                                               │
│      → 将大矩阵分块在SRAM中计算                                  │
│      → 减少HBM访问                                               │
│                                                                  │
│   2. Kernel Fusion (算子融合)                                    │
│      → 整个Attention在一个kernel中完成                           │
│      → 中间结果不写回HBM                                         │
│                                                                  │
│   3. Recomputation (重计算)                                      │
│      → 反向时重新计算S和P                                        │
│      → 只保存O, m, l                                             │
│      → 显存从O(N²)降到O(N)                                       │
│                                                                  │
│   4. Online Softmax                                              │
│      → 允许分块计算正确的softmax                                 │
│      → 是Tiling和Recomputation的数学基础                        │
│                                                                  │
│   综合效果:                                                      │
│   → 速度提升 2-4x                                                │
│   → 显存减少 5-20x                                               │
│   → 支持更长序列                                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 10. 关键术语

| 术语 | 英文 | 含义 |
|------|------|------|
| 重计算 | Recomputation | 反向时重新计算中间结果 |
| 检查点 | Checkpoint | 保存部分中间状态 |
| 激活值 | Activation | 神经网络的中间输出 |
| 时空权衡 | Time-Space Tradeoff | 用时间换空间 |
| 统计量 | Statistics | m(最大值)和l(指数和) |

---

## 📚 延伸阅读

- [FlashAttention论文 Section 3.2](https://arxiv.org/abs/2205.14135)：反向传播算法
- [Gradient Checkpointing](https://pytorch.org/docs/stable/checkpoint.html)：PyTorch官方文档
- [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174)：Checkpointing的理论基础


