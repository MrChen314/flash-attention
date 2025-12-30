# TiledCopy内存拷贝

> cute对高效内存传输的抽象

---

## 1. 什么是TiledCopy

### 1.1 背景：GPU内存层次

GPU有多级内存层次，数据需要在它们之间移动：

```
┌─────────────────────────────────────────────────────────────────┐
│                    GPU内存层次与数据传输                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   寄存器 (Registers)                                             │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  最快 | 每个线程私有 | MMA操作的输入输出                  │   │
│   └─────────────────────────────────────────────────────────┘   │
│                     ↑ S2R (SMEM → Register)                      │
│                     ↓ R2S (Register → SMEM)                      │
│   共享内存 (SMEM)                                                │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  快速 | 线程块共享 | 数据暂存、线程间通信                 │   │
│   └─────────────────────────────────────────────────────────┘   │
│                     ↑ G2S (GMEM → SMEM)                          │
│                     ↓ S2G (SMEM → GMEM)                          │
│   全局内存 (HBM)                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  大容量 | 所有线程可访问 | 输入输出数据                   │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 TiledCopy定义

**TiledCopy**是cute对高效内存拷贝操作的抽象，它定义了：

1. **Copy Atom**：单个拷贝指令的规格（如cp.async 128位）
2. **Thread Layout**：线程如何分布在Tile上
3. **Value Layout**：每个线程拷贝的值的排布

```
┌─────────────────────────────────────────────────────────────────┐
│                    TiledCopy 结构                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   TiledCopy = Copy_Atom + ThreadLayout + ValueLayout            │
│                                                                  │
│   ┌──────────────┐                                              │
│   │  Copy Atom   │  单个拷贝指令                                │
│   │              │  例: SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>    │
│   │  ┌────────┐  │  - 128位（16字节）一次                       │
│   │  │ 128bit │  │  - 异步拷贝                                  │
│   │  └────────┘  │  - 利用L2缓存                                │
│   └──────────────┘                                              │
│          ↓                                                       │
│   ┌──────────────┐                                              │
│   │ ThreadLayout │  线程如何分布                                │
│   │              │  例: Layout<Shape<_16, _8>>                  │
│   │  T0 T1 ...   │  - 16×8 = 128个线程位置                      │
│   │  T16 T17 ... │  - 定义线程网格                              │
│   └──────────────┘                                              │
│          ↓                                                       │
│   ┌──────────────┐                                              │
│   │ ValueLayout  │  每个线程拷贝的值                            │
│   │              │  例: Layout<Shape<_1, _8>>                   │
│   │  [8个FP16]   │  - 每个线程拷贝8个FP16元素                   │
│   └──────────────┘                                              │
│                                                                  │
│   总Tile大小: 16×8 线程 × 1×8 值/线程 = 16×64 元素             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Copy Atom类型

### 2.1 同步拷贝

基本的拷贝操作，执行后立即可用：

```cpp
// 自动向量化拷贝
using CopyAtom = Copy_Atom<AutoVectorizingCopy, half_t>;

// 显式128位拷贝
using CopyAtom = Copy_Atom<UniversalCopy<uint128_t>, half_t>;
```

### 2.2 异步拷贝（cp.async）

Ampere及以上架构支持，可以隐藏内存延迟：

```cpp
// 异步拷贝，使用L2缓存
using CopyAtom = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, half_t>;

// 异步拷贝，绕过L2缓存
using CopyAtom = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>;
```

### 2.3 TMA（Hopper架构）

Tensor Memory Accelerator，硬件级别的张量传输：

```cpp
// TMA拷贝（SM90+）
using CopyAtom = Copy_Atom<SM90_TMA_LOAD, half_t>;
```

---

## 3. 创建TiledCopy

### 3.1 基本创建方式

```cpp
#include <cute/atom/copy_atom.hpp>

using namespace cute;

// 方式1: make_tiled_copy
auto tiled_copy = make_tiled_copy(
    Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, half_t>{},
    Layout<Shape<_16, _8>>{},    // 线程布局: 16×8
    Layout<Shape<_1, _8>>{}      // 值布局: 每线程1×8
);

// 方式2: 使用类型别名
using GmemTiledCopy = TiledCopy<
    Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, half_t>,
    Layout<Shape<_16, _8>>,
    Layout<Shape<_1, _8>>
>;
GmemTiledCopy gmem_tiled_copy;
```

### 3.2 基于TiledMMA创建

TiledCopy可以与TiledMMA配合，自动匹配布局：

```cpp
// 创建与MMA配合的拷贝
auto smem_tiled_copy_A = make_tiled_copy_A(
    Copy_Atom<SM75_U32x4_LDSM_N, half_t>{},
    tiled_mma
);

auto smem_tiled_copy_B = make_tiled_copy_B(
    Copy_Atom<SM75_U32x4_LDSM_N, half_t>{},
    tiled_mma
);
```

---

## 4. 使用TiledCopy

### 4.1 获取线程视图

```cpp
// 在kernel中
__global__ void kernel() {
    GmemTiledCopy gmem_tiled_copy;
    
    // 获取当前线程的拷贝视图
    auto thr_copy = gmem_tiled_copy.get_thread_slice(threadIdx.x);
}
```

### 4.2 partition操作

partition将源Tensor和目标Tensor分配给线程：

```cpp
auto thr_copy = gmem_tiled_copy.get_thread_slice(threadIdx.x);

// partition_S: 分区源Tensor（Source）
Tensor tSgS = thr_copy.partition_S(gS);  // 全局内存源

// partition_D: 分区目标Tensor（Destination）
Tensor tSsS = thr_copy.partition_D(sS);  // 共享内存目标

// 执行拷贝
cute::copy(gmem_tiled_copy, tSgS, tSsS);
```

### 4.3 retile操作

当需要将数据从一种布局转换到另一种布局时：

```cpp
// 假设有MMA需要的布局
auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
Tensor tCrA_mma = thr_mma.partition_A(sA);

// SMEM拷贝的布局
auto smem_thr_copy = smem_tiled_copy.get_thread_slice(threadIdx.x);
Tensor tCsA = smem_thr_copy.partition_S(sA);

// retile将MMA布局转换为Copy布局
Tensor tCrA_copy = smem_thr_copy.retile_D(tCrA_mma);

// 现在可以用smem_tiled_copy从tCsA拷贝到tCrA_copy
cute::copy(smem_tiled_copy, tCsA, tCrA_copy);
```

---

## 5. 异步拷贝流水线

### 5.1 cp.async基础

cp.async是异步拷贝指令，发出后不等待完成：

```cpp
// 发起异步拷贝
cute::copy(gmem_tiled_copy, tSgS, tSsS);

// 插入fence，标记一组拷贝
cute::cp_async_fence();

// 等待拷贝完成
cute::cp_async_wait<0>();  // 等待所有
// 或
cute::cp_async_wait<1>();  // 保留1组未完成
```

### 5.2 流水线模式

```cpp
// 双缓冲流水线示例
constexpr int kStages = 2;

// 预取第一个stage
cute::copy(gmem_tiled_copy, tSgQ(_, _, 0), tSsQ(_, _, 0));
cute::cp_async_fence();

// 主循环
for (int k = 0; k < num_k_tiles; ++k) {
    // 等待当前stage的数据
    cute::cp_async_wait<kStages - 1>();
    __syncthreads();
    
    // 预取下一个stage
    if (k + 1 < num_k_tiles) {
        int next_stage = (k + 1) % kStages;
        cute::copy(gmem_tiled_copy, tSgQ(_, _, k+1), tSsQ(_, _, next_stage));
        cute::cp_async_fence();
    }
    
    // 使用当前stage的数据进行计算
    int curr_stage = k % kStages;
    cute::gemm(tiled_mma, tSsQ(_, _, curr_stage), tSsK(_, _, curr_stage), acc);
}
```

### 5.3 流水线时序图

```
时间 →
───────────────────────────────────────────────────────────────────

线程工作:
Stage 0: [Load Q0,K0] ──fence── [Load Q2,K2] ──fence── [Load Q4,K4]
Stage 1:              [Load Q1,K1] ──fence── [Load Q3,K3] ──fence──

计算:
         [等待]  [MMA Q0K0]  [MMA Q1K1]  [MMA Q2K2]  [MMA Q3K3] ...

数据流:
         GMEM→SMEM    GMEM→SMEM    GMEM→SMEM
              ↓            ↓            ↓
         [Stage0]     [Stage1]     [Stage0]    (交替使用)
              ↓            ↓            ↓
             MMA          MMA          MMA

优势：加载和计算重叠，隐藏内存延迟
```

---

## 6. 在FlashAttention中的应用

### 6.1 加载Q到共享内存

```cpp
// hopper/mainloop_fwd_sm80.hpp 中的典型模式

// 定义TiledCopy
using GmemTiledCopyQKV = TiledCopy<
    Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, Element>,
    Layout<Shape<_16, _8>>,
    Layout<Shape<_1, _8>>
>;

// 创建TiledCopy和分区
GmemTiledCopyQKV gmem_tiled_copy_QKV;
auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(threadIdx.x);

// 分区Q的全局和共享内存视图
Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);  // 全局内存Q
Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);  // 共享内存Q

// 异步拷贝
cute::copy(gmem_tiled_copy_QKV, tQgQ, tQsQ);
cute::cp_async_fence();
```

### 6.2 从共享内存加载到寄存器

```cpp
// SMEM到寄存器的拷贝（为MMA准备）
using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
auto smem_thr_copy_A = smem_tiled_copy_A.get_thread_slice(threadIdx.x);

// 分区
Tensor tCsQ = smem_thr_copy_A.partition_S(sQ);     // SMEM源
Tensor tCrQ = smem_thr_copy_A.retile_D(tCrQ_mma);  // 寄存器目标

// 拷贝
cute::copy(smem_tiled_copy_A, tCsQ(_, _, k), tCrQ(_, _, k));
```

### 6.3 写回输出

```cpp
// 将累加器写回全局内存

// 先写到共享内存
cute::copy(r2s_tiled_copy, tOrO, tOsO);
__syncthreads();

// 再从共享内存写到全局内存
cute::copy(s2g_tiled_copy, tOsO, tOgO);
```

---

## 7. 边界处理

### 7.1 predicate（谓词）

处理不对齐的边界情况：

```cpp
// 创建predicate tensor
Tensor cQ = make_identity_tensor(make_shape(seqlen, headdim));
auto thr_copy = gmem_tiled_copy.get_thread_slice(threadIdx.x);
Tensor tQcQ = thr_copy.partition_S(cQ);

// 带predicate的拷贝
#pragma unroll
for (int m = 0; m < size<1>(tQgQ); ++m) {
    if (get<0>(tQcQ(0, m, 0)) < actual_seqlen) {
        cute::copy(gmem_tiled_copy, tQgQ(_, m, _), tQsQ(_, m, _));
    } else {
        cute::clear(tQsQ(_, m, _));  // 越界部分清零
    }
}
```

### 7.2 FlashAttention的utils.h中的copy函数

```cpp
// utils.h 中的通用copy函数
template <bool Is_even_MN=true, bool Is_even_K=true, 
          bool Clear_OOB_MN=false, bool Clear_OOB_K=true, ...>
CUTLASS_DEVICE void copy(TiledCopy const &tiled_copy, 
                         Tensor const &S, Tensor &D,
                         Tensor const &identity_MN, 
                         Tensor const &predicate_K, 
                         const int max_MN=0) {
    // 根据模板参数处理边界情况
    if constexpr (Is_even_MN && Is_even_K) {
        cute::copy(tiled_copy, S, D);
    } else {
        // 带predicate的拷贝逻辑
        // ...
    }
}
```

---

## 8. 关键术语

| 术语 | 英文 | 含义 |
|------|------|------|
| Copy Atom | - | 单个拷贝指令的抽象 |
| TiledCopy | - | 多线程协作拷贝的抽象 |
| G2S | Global to Shared | 全局内存到共享内存 |
| S2R | Shared to Register | 共享内存到寄存器 |
| R2S | Register to Shared | 寄存器到共享内存 |
| S2G | Shared to Global | 共享内存到全局内存 |
| cp.async | - | 异步拷贝指令 |
| partition_S | - | 分区源Tensor |
| partition_D | - | 分区目标Tensor |
| retile | - | 重新排布为不同Layout |

---

## 9. 总结

### 9.1 TiledCopy的核心价值

```
┌─────────────────────────────────────────────────────────────────┐
│                    TiledCopy 核心价值                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. 封装复杂性                                                   │
│     └── 隐藏cp.async、向量化加载等细节                          │
│                                                                  │
│  2. 自动线程分配                                                 │
│     └── partition自动计算每个线程负责的数据                     │
│                                                                  │
│  3. 异步执行                                                     │
│     └── 支持cp.async，实现流水线                                │
│                                                                  │
│  4. 与TiledMMA配合                                               │
│     └── make_tiled_copy_A/B 自动匹配MMA所需布局                 │
│                                                                  │
│  5. 边界处理                                                     │
│     └── predicate支持非对齐访问                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 9.2 典型使用模式

```cpp
// 1. G2S: 全局内存 → 共享内存
auto gmem_thr_copy = gmem_tiled_copy.get_thread_slice(threadIdx.x);
Tensor tSgS = gmem_thr_copy.partition_S(gS);
Tensor tSsS = gmem_thr_copy.partition_D(sS);
cute::copy(gmem_tiled_copy, tSgS, tSsS);
cute::cp_async_fence();
cute::cp_async_wait<0>();
__syncthreads();

// 2. S2R: 共享内存 → 寄存器（为MMA准备）
auto smem_thr_copy = smem_tiled_copy.get_thread_slice(threadIdx.x);
Tensor tCsS = smem_thr_copy.partition_S(sS);
Tensor tCrS = smem_thr_copy.retile_D(tCrS_mma);
cute::copy(smem_tiled_copy, tCsS, tCrS);

// 3. MMA计算
cute::gemm(tiled_mma, tCrA, tCrB, acc);
```

---

## 📚 延伸阅读

- [cute Copy文档](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/04_copy.md)
- [cp.async编程指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-data-copies)
- FlashAttention源码：`hopper/mainloop_fwd_sm80.hpp`


