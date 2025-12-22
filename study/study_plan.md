# FlashAttention-2 C/CUDA 代码学习计划

> 适用于CUDA初学者，循序渐进掌握FlashAttention-2的实现原理

---

## 📚 第一阶段：前置知识准备（1-2周）

### 1.1 CUDA编程基础
**目标：** 理解GPU并行计算的基本概念

**学习内容：**
- [ ] GPU架构基础（SM、Warp、Thread）
- [ ] CUDA编程模型（Grid → Block → Thread层次结构）
- [ ] 内存层次结构（Global Memory、Shared Memory、Registers）
- [ ] 线程同步（`__syncthreads()`）
- [ ] 异步内存拷贝（`cp.async`）

**推荐资源：**
- NVIDIA CUDA C++ Programming Guide（官方文档）
- 《CUDA by Example》入门书籍
- 实践：编写一个简单的矩阵乘法CUDA kernel

### 1.2 C++模板元编程基础
**目标：** 理解FlashAttention代码中大量使用的模板技术

**学习内容：**
- [ ] 模板函数与模板类
- [ ] 模板特化与SFINAE
- [ ] `constexpr` 编译期计算
- [ ] `if constexpr` 编译期分支

### 1.3 Attention机制原理
**目标：** 深入理解标准Attention的计算过程

**学习内容：**
- [ ] Self-Attention公式：`Attention(Q,K,V) = softmax(QK^T / √d) × V`
- [ ] Multi-Head Attention (MHA)
- [ ] Grouped-Query Attention (GQA) / Multi-Query Attention (MQA)
- [ ] Causal Mask（因果掩码）

---

## 📚 第二阶段：FlashAttention算法原理（1周）

### 2.1 FlashAttention核心思想
**目标：** 理解为什么需要FlashAttention以及它的核心优化思路

**学习内容：**
- [ ] 标准Attention的内存瓶颈分析（O(N²)显存占用）
- [ ] IO-Aware算法设计思想
- [ ] Tiling（分块）技术：将大矩阵分成小块在SRAM中计算
- [ ] Recomputation（重计算）策略

**推荐阅读：**
- FlashAttention论文：https://arxiv.org/abs/2205.14135
- FlashAttention-2论文：https://arxiv.org/abs/2307.08691

### 2.2 Online Softmax算法 ⭐重要
**目标：** 这是FlashAttention的核心算法，必须深入理解

**学习内容：**
- [ ] 标准Softmax的计算过程及数值稳定性问题
- [ ] Online Softmax：如何在不知道完整序列的情况下增量计算softmax
- [ ] 核心公式推导：
  ```
  m_new = max(m_old, m_block)           # 更新最大值
  l_new = exp(m_old - m_new) * l_old + exp(m_block - m_new) * l_block  # 更新累加和
  O_new = exp(m_old - m_new) * O_old + exp(m_block - m_new) * O_block  # 更新输出
  ```
- [ ] LSE (Log-Sum-Exp) 的作用与计算

**对应代码文件：**
```
csrc/flash_attn/src/softmax.h
```

---

## 📚 第三阶段：CUTLASS/cute库基础（1-2周）

### 3.1 为什么需要CUTLASS/cute
**目标：** 理解FlashAttention为何基于CUTLASS构建

**学习内容：**
- [ ] CUTLASS是什么：NVIDIA的高性能CUDA模板库
- [ ] cute是什么：CUTLASS 3.x中的张量抽象层
- [ ] 为什么选择CUTLASS：封装了Tensor Core操作、内存访问优化

### 3.2 cute核心概念 ⭐关键
**目标：** 掌握FlashAttention代码中大量使用的cute抽象

**学习内容：**
- [ ] **Tensor**: 多维数组抽象
  ```cpp
  Tensor mQ = make_tensor(make_gmem_ptr(ptr), shape, stride);
  ```
- [ ] **Layout**: 描述数据在内存中的排布
  ```cpp
  Layout layout = make_layout(shape, stride);
  ```
- [ ] **TiledMMA**: 封装Tensor Core矩阵乘法
- [ ] **TiledCopy**: 高效内存拷贝抽象
- [ ] **local_tile**: 获取tensor的局部tile

**推荐资源：**
- CUTLASS cute教程：https://github.com/NVIDIA/cutlass/tree/main/media/docs/cute
- 动手实践：阅读cutlass/examples中的cute示例

### 3.3 常见cute操作速查
```cpp
// 创建全局内存tensor
Tensor gQ = make_tensor(make_gmem_ptr(ptr), shape, stride);

// 创建共享内存tensor  
Tensor sQ = make_tensor(make_smem_ptr(smem_ptr), SmemLayout{});

// 获取局部tile
Tensor tile = local_tile(gQ, tile_shape, coord);

// 分区（用于线程级并行）
Tensor tQgQ = gmem_thr_copy.partition_S(gQ);

// 数据拷贝
cute::copy(src, dst);

// 异步拷贝栅栏
cute::cp_async_fence();
cute::cp_async_wait<N>();
```

---

## 📚 第四阶段：代码结构总览（3天）

### 4.1 代码目录结构
```
csrc/flash_attn/
├── flash_api.cpp              # ⭐ API入口，PyTorch绑定
└── src/
    ├── flash.h                # ⭐ 核心数据结构定义
    ├── flash_fwd_kernel.h     # ⭐⭐⭐ 前向kernel核心实现
    ├── flash_bwd_kernel.h     # 反向kernel核心实现
    ├── flash_fwd_launch_template.h  # 前向kernel启动模板
    ├── flash_bwd_launch_template.h  # 反向kernel启动模板
    ├── kernel_traits.h        # Kernel配置traits
    ├── block_info.h           # 块信息处理
    ├── softmax.h              # ⭐ Online Softmax实现
    ├── mask.h                 # Causal/Local mask实现
    ├── dropout.h              # Dropout实现
    ├── rotary.h               # RoPE旋转位置编码
    ├── alibi.h                # ALiBi位置编码
    ├── utils.h                # 工具函数
    └── flash_fwd_hdim*.cu     # 预编译的kernel实例
```

### 4.2 调用链概览
```
Python调用
    ↓
flash_attn_interface.py
    ↓
flash_api.cpp (mha_fwd / mha_bwd)
    ↓
run_mha_fwd / run_mha_bwd
    ↓
flash_fwd_kernel.h::compute_attn_1rowblock
```

---

## 📚 第五阶段：核心代码精读（2-3周）

### 5.1 第一周：数据结构与API层

#### Day 1-2: flash.h - 参数结构体
**文件路径：** `csrc/flash_attn/src/flash.h`

**学习要点：**
- [ ] `Qkv_params` 结构体：QKV矩阵指针和stride
- [ ] `Flash_fwd_params` 结构体：前向传播所需全部参数
- [ ] `Flash_bwd_params` 结构体：反向传播额外参数
- [ ] 理解各种stride的含义（batch_stride, row_stride, head_stride）

**关键代码段：**
```cpp
struct Flash_fwd_params : public Qkv_params {
    void * __restrict__ o_ptr;           // 输出指针
    void * __restrict__ softmax_lse_ptr; // LSE指针
    int b, seqlen_q, seqlen_k, d;        // 维度信息
    float scale_softmax;                 // 缩放因子
    bool is_causal;                      // 是否因果
    int window_size_left, window_size_right; // 滑动窗口
    // ...
};
```

#### Day 3-4: flash_api.cpp - API入口
**文件路径：** `csrc/flash_attn/flash_api.cpp`

**学习要点：**
- [ ] `set_params_fprop`: 如何设置前向参数
- [ ] `mha_fwd`: 标准前向传播入口
- [ ] `mha_varlen_fwd`: 变长序列前向传播
- [ ] `run_mha_fwd`: kernel调度逻辑
- [ ] Split-KV策略：何时使用、如何选择num_splits

**关键代码段：**
```cpp
void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    FP16_SWITCH(!params.is_bf16, [&] {
        HEADDIM_SWITCH(params.d, [&] {
            BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
            });
        });
    });
}
```

#### Day 5: kernel_traits.h - Kernel配置
**文件路径：** `csrc/flash_attn/src/kernel_traits.h`

**学习要点：**
- [ ] `Flash_fwd_kernel_traits`: 定义kernel的各种编译期常量
- [ ] kBlockM, kBlockN: tile大小配置
- [ ] SmemLayout: 共享内存布局
- [ ] TiledMma: 矩阵乘法配置

---

### 5.2 第二周：前向Kernel核心实现 ⭐⭐⭐最重要

#### Day 1-3: flash_fwd_kernel.h 整体流程
**文件路径：** `csrc/flash_attn/src/flash_fwd_kernel.h`

**核心函数：** `compute_attn_1rowblock`

**学习要点（按代码顺序）：**
```
1. 初始化阶段
   - [ ] 计算块索引(bidb, bidh, m_block)
   - [ ] 获取块信息(BlockInfo)
   - [ ] 计算n_block_min, n_block_max（需要处理的K/V块范围）

2. 准备阶段
   - [ ] 创建全局内存Tensor (mQ, mK, mV, gQ, gK, gV)
   - [ ] 创建共享内存Tensor (sQ, sK, sV)
   - [ ] 设置TiledCopy和TiledMma
   - [ ] 初始化累加器 acc_o

3. Q加载
   - [ ] 从全局内存加载Q到共享内存
   - [ ] cp_async异步拷贝

4. 主循环（从后向前遍历K/V块）
   for n_block = n_block_max-1 to n_block_min:
       
   a) 加载K到共享内存
   b) 计算 S = Q @ K^T (gemm)
   c) 应用softcap（可选）
   d) 应用mask
   e) 加载V到共享内存
   f) 计算在线softmax，更新acc_o
   g) 计算 O_partial = softmax(S) @ V (gemm_rs)

5. 收尾阶段
   - [ ] 归一化输出
   - [ ] 写回全局内存
   - [ ] 保存LSE
```

#### Day 4-5: 深入理解主循环
**重点关注以下模式：**

```cpp
// 1. 异步加载K
flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tKgK, tKsK, ...);
cute::cp_async_fence();

// 2. 等待K加载完成，计算S = Q @ K^T  
flash::cp_async_wait<0>();
__syncthreads();
flash::gemm(acc_s, tSrQ, tSrK, ...);  // S = Q @ K^T

// 3. 应用mask
mask.template apply_mask<Is_causal, Is_even_MN>(acc_s, ...);

// 4. 在线softmax更新
softmax.template softmax_rescale_o<Is_first, Check_inf>(acc_s, acc_o, ...);

// 5. 计算O_partial = softmax(S) @ V
Tensor rP = flash::convert_type<Element>(acc_s);
flash::gemm_rs(acc_o, tOrP, tOrVt, ...);
```

#### Day 6-7: 辅助模块深入

**softmax.h - 在线Softmax实现**
```cpp
template <int kNRows>
struct Softmax {
    // 核心方法：更新最大值和累加和，同时rescale输出
    template<bool Is_first, bool Check_inf>
    __forceinline__ __device__ void softmax_rescale_o(
        Tensor<float> &acc_s,    // 当前块的scores
        Tensor<float> &acc_o,    // 累积输出
        float scale              // softmax缩放
    );
};
```

**mask.h - Mask应用**
```cpp
template <bool Is_causal, bool Is_local, bool Has_alibi>
struct Mask {
    // 应用因果mask或局部mask
    template <bool Causal_mask, bool Is_even_MN>
    __forceinline__ __device__ void apply_mask(
        Tensor &scores,
        int col_idx_offset,
        int row_idx_offset,
        int warp_row_stride
    );
};
```

---

### 5.3 第三周：Split-KV与反向传播

#### Day 1-2: Split-KV机制
**函数：** `compute_attn_1rowblock_splitkv`

**学习要点：**
- [ ] 何时需要Split-KV：长序列推理优化
- [ ] 如何分割K/V序列
- [ ] 如何合并多个split的结果
- [ ] `combine_attn_seqk_parallel`: 合并函数

#### Day 3-5: 反向传播kernel（选学）
**文件：** `csrc/flash_attn/src/flash_bwd_kernel.h`

**学习要点：**
- [ ] 反向传播的数学推导
- [ ] dQ, dK, dV的计算
- [ ] 重计算策略：不保存P矩阵，反向时重新计算
- [ ] 原子操作处理并发写入

---

## 📚 第六阶段：进阶专题（1-2周）

### 6.1 性能优化技巧
- [ ] 共享内存bank冲突避免
- [ ] 寄存器分配优化
- [ ] 指令级并行(ILP)
- [ ] Warp级原语使用

### 6.2 扩展功能实现
- [ ] Paged KV Cache实现
- [ ] Rotary Position Embedding (RoPE)
- [ ] ALiBi位置编码
- [ ] Sliding Window Attention
- [ ] Softcap机制

### 6.3 FlashAttention-3 新特性（Hopper架构）
**目录：** `hopper/`
- [ ] TMA (Tensor Memory Accelerator)
- [ ] Warp-specialized设计
- [ ] FP8支持

---

## 📝 学习建议

### 高效学习方法
1. **先整体后局部**：先理解算法原理和代码架构，再深入细节
2. **结合调试**：使用`printf`或CUDA调试工具观察中间值
3. **画图理解**：手绘Tiling过程和数据流动
4. **对比学习**：对比标准Attention实现，理解优化点

### 关键检查点
完成以下任务说明你已掌握核心内容：
- [ ] 能用自己的话解释Online Softmax算法
- [ ] 能画出`compute_attn_1rowblock`的执行流程图
- [ ] 理解每个`__syncthreads()`的作用
- [ ] 能解释为什么使用异步内存拷贝

### 推荐学习顺序总结
```
Week 1-2: CUDA基础 + Attention原理
Week 3:   FlashAttention算法 + Online Softmax
Week 4:   CUTLASS/cute库
Week 5:   flash.h + flash_api.cpp
Week 6-7: flash_fwd_kernel.h (核心！)
Week 8:   辅助模块 + Split-KV
Week 9+:  反向传播 + 进阶优化
```

---

## 📖 参考资源

### 论文
- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)
- [FlashAttention-3](https://arxiv.org/abs/2407.08608)

### 代码库
- [FlashAttention GitHub](https://github.com/Dao-AILab/flash-attention)
- [CUTLASS](https://github.com/NVIDIA/cutlass)

### 博客文章
- [FlashAttention核心逻辑详解](https://zhuanlan.zhihu.com/p/669926191)
- [Online Softmax推导](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf)

### 视频教程
- [Tri Dao的FlashAttention讲解](https://www.youtube.com/watch?v=FThvfkXWqtE)

---

**祝学习顺利！🚀**

