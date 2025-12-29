# 2.2 Online Softmax算法

> FlashAttention-2 学习计划 · 第二阶段 · 算法原理 ⭐重要

---

## 📖 本章概述

Online Softmax是FlashAttention的**核心算法**，它解决了一个关键问题：**如何在不知道完整序列的情况下增量计算Softmax**。这使得FlashAttention能够分块处理长序列，而不需要将整个注意力矩阵存储在内存中。

**核心问题：** 标准Softmax需要遍历整个序列两次（一次求最大值，一次计算归一化），如何改为单次遍历？

**Online Softmax的答案：** 通过维护**运行时状态**（最大值和累加和），在每个block到来时动态更新！

```
┌─────────────────────────────────────────────────────────────────┐
│                    标准Softmax的计算过程                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   输入: x = [x₁, x₂, ..., xₙ]                                   │
│                                                                  │
│   第一次遍历: m = max(x₁, x₂, ..., xₙ)                          │
│   第二次遍历: l = Σ exp(xᵢ - m)                                  │
│   第三次遍历: softmax(xᵢ) = exp(xᵢ - m) / l                     │
│                                                                  │
│   问题：需要完整数据才能计算，无法分块处理！                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    Online Softmax的突破                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   每处理一个block，动态更新：                                    │
│                                                                  │
│   m_new = max(m_old, m_block)         ← 更新最大值               │
│   l_new = e^(m_old-m_new) · l_old +                             │
│           e^(m_block-m_new) · l_block  ← 更新累加和               │
│   O_new = e^(m_old-m_new) · O_old +                             │
│           e^(m_block-m_new) · O_block  ← 更新输出                 │
│                                                                  │
│   优势：单次遍历，支持分块计算！                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**学习目标：**
- 理解标准Softmax的计算过程及数值稳定性问题
- 掌握Online Softmax的增量计算原理
- 深入理解核心公式的数学推导
- 学习LSE (Log-Sum-Exp) 的作用与计算

**预计学习时间：** 2-3天

---

## 📚 章节目录

| 序号 | 主题 | 内容概要 | 文件 |
|------|------|----------|------|
| 1 | [标准Softmax数值稳定性](./1_标准Softmax数值稳定性/) | Softmax原理与数值溢出问题 | [文档](./1_标准Softmax数值稳定性/标准Softmax数值稳定性.md) / [实践](./1_标准Softmax数值稳定性/标准Softmax数值稳定性.ipynb) |
| 2 | [Online Softmax增量计算](./2_Online_Softmax增量计算/) | 不完整序列下的Softmax计算 | [文档](./2_Online_Softmax增量计算/Online_Softmax增量计算.md) / [实践](./2_Online_Softmax增量计算/Online_Softmax增量计算.ipynb) |
| 3 | [核心公式推导](./3_核心公式推导/) | 数学公式的完整推导过程 | [文档](./3_核心公式推导/核心公式推导.md) / [实践](./3_核心公式推导/核心公式推导.ipynb) |
| 4 | [LSE_Log-Sum-Exp](./4_LSE_Log-Sum-Exp/) | LSE的作用与高效计算 | [文档](./4_LSE_Log-Sum-Exp/LSE_Log-Sum-Exp.md) / [实践](./4_LSE_Log-Sum-Exp/LSE_Log-Sum-Exp.ipynb) |

---

## 🛠️ 环境准备

### Python环境

```bash
# 激活conda环境
conda activate ma_rlhf

# 确保安装了PyTorch和必要的库
pip install torch matplotlib numpy
```

### 运行Notebook的要求

1. **Python**：版本 3.8 或更高
2. **PyTorch**：版本 2.0 或更高
3. **可选GPU**：本章主要是算法理解，CPU也可运行

### 验证环境

```python
import torch
import numpy as np
print(f"PyTorch版本: {torch.__version__}")
print(f"NumPy版本: {np.__version__}")
```

---

## 📝 学习建议

1. **数学推导为主**：本章重点是理解数学原理，建议手推公式
2. **对比验证**：用代码验证Online Softmax与标准Softmax的结果一致
3. **数值稳定性**：特别关注exp操作的溢出问题
4. **联系FlashAttention**：理解这些公式如何应用到注意力计算中

---

## 🔗 对应代码文件

FlashAttention中的相关实现：

```
csrc/flash_attn/src/softmax.h
```

---

## 📊 关键公式预览

### 数值稳定的Softmax

$$
\text{softmax}(x_i) = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}}, \quad m = \max_j(x_j)
$$

### Online Softmax更新公式

给定当前状态 $(m_{old}, l_{old}, O_{old})$ 和新block数据，更新规则为：

$$
m_{new} = \max(m_{old}, m_{block})
$$

$$
l_{new} = e^{m_{old} - m_{new}} \cdot l_{old} + e^{m_{block} - m_{new}} \cdot l_{block}
$$

$$
O_{new} = e^{m_{old} - m_{new}} \cdot O_{old} + e^{m_{block} - m_{new}} \cdot O_{block}
$$

### LSE (Log-Sum-Exp)

$$
\text{LSE}(x_1, ..., x_n) = \log\left(\sum_i e^{x_i}\right) = m + \log\left(\sum_i e^{x_i - m}\right)
$$

---

## ✅ 学习检查点

完成本章后，你应该能够：

- [ ] 解释为什么直接计算 $e^x$ 会有数值稳定性问题
- [ ] 手推Online Softmax的更新公式
- [ ] 实现一个正确的Online Softmax算法
- [ ] 解释LSE在FlashAttention中的作用
- [ ] 说明Online Softmax如何使分块计算成为可能

---

## 📚 推荐阅读

- [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867)：Online Softmax的原始论文
- [FlashAttention论文](https://arxiv.org/abs/2205.14135)：第2.2节详细介绍了Online Softmax
- [Safe Softmax实现技巧](https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/)：数值稳定技巧

---

**上一章：** [2.1 FlashAttention核心思想](../2_1FlashAttention核心思想/)

**下一章：** [3.1 为什么需要CUTLASS/cute](../3_1为什么需要CUTLASS_cute/)


