# 1.1 CUDA编程基础

> FlashAttention-2 学习计划 · 第一阶段 · 前置知识

---

## 📖 本章概述

本章将帮助你从零开始掌握CUDA编程的核心概念。作为理解FlashAttention实现的基础，你需要深入理解GPU的并行计算模型和内存层次结构。

**学习目标：**
- 理解GPU硬件架构与并行计算原理
- 掌握CUDA编程模型的核心概念
- 了解GPU内存层次及优化策略
- 学会使用线程同步机制
- 理解异步内存拷贝在高性能计算中的作用

**预计学习时间：** 1-2周

---

## 📚 章节目录

| 序号 | 主题 | 内容概要 | 文件 |
|------|------|----------|------|
| 1 | [GPU架构基础](./1_GPU架构基础/) | SM、Warp、Thread概念 | [文档](./1_GPU架构基础/GPU架构基础.md) / [实践](./1_GPU架构基础/GPU架构基础.ipynb) |
| 2 | [CUDA编程模型](./2_CUDA编程模型/) | Grid/Block/Thread层次结构 | [文档](./2_CUDA编程模型/CUDA编程模型.md) / [实践](./2_CUDA编程模型/CUDA编程模型.ipynb) |
| 3 | [内存层次结构](./3_内存层次结构/) | Global/Shared/Register内存 | [文档](./3_内存层次结构/内存层次结构.md) / [实践](./3_内存层次结构/内存层次结构.ipynb) |
| 4 | [线程同步](./4_线程同步/) | `__syncthreads()`与竞争条件 | [文档](./4_线程同步/线程同步.md) / [实践](./4_线程同步/线程同步.ipynb) |
| 5 | [异步内存拷贝](./5_异步内存拷贝/) | cp.async与计算重叠 | [文档](./5_异步内存拷贝/异步内存拷贝.md) / [实践](./5_异步内存拷贝/异步内存拷贝.ipynb) |

---

## 🛠️ 环境准备

### 运行Notebook的要求

1. **NVIDIA GPU**：建议使用 Ampere 架构（如 RTX 30系列、A100）或更新
2. **CUDA Toolkit**：版本 11.0 或更高
3. **Python环境**：安装 `nvcc4jupyter` 扩展

```bash
# 安装 nvcc4jupyter（支持在Jupyter中编写CUDA代码）
pip install nvcc4jupyter
```

### 验证环境

```bash
# 检查CUDA版本
nvcc --version

# 检查GPU信息
nvidia-smi
```

---

## 📝 学习建议

1. **先读文档，后做实践**：每个主题先阅读 `.md` 文档理解概念，再运行 `.ipynb` 动手实践
2. **动手修改代码**：尝试修改示例代码中的参数，观察结果变化
3. **画图辅助理解**：对于线程索引、内存布局等概念，建议手绘图帮助理解
4. **关联FlashAttention**：学习时思考这些概念在FlashAttention中如何应用

---

## 🔗 与FlashAttention的关联

| CUDA概念 | 在FlashAttention中的应用 |
|----------|--------------------------|
| Warp | FlashAttention使用Warp级别的协作完成矩阵运算 |
| Shared Memory | Q、K、V的分块加载到共享内存 |
| `__syncthreads()` | 确保数据加载完成后再计算 |
| 异步拷贝 | 使用`cp.async`实现计算与数据加载的重叠 |

---

## ✅ 学习检查点

完成本章后，你应该能够：

- [ ] 解释SM、Warp、Thread的层次关系
- [ ] 计算任意线程的全局索引
- [ ] 说明Global Memory和Shared Memory的区别
- [ ] 编写使用`__syncthreads()`的正确代码
- [ ] 理解异步拷贝如何提升性能

---

**下一章：** [1.2 C++模板元编程基础](../1_2C++模板元编程/)

