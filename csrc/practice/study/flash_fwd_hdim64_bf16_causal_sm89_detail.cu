/******************************************************************************
 * Flash Attention Forward - 详细展开学习版
 * 
 * 本文件将 run_mha_fwd_hdim64 函数完全展开，方便 CUDA 学习者理解
 * Flash Attention 的内部实现细节。
 * 
 * 配置: SM89 (Ada Lovelace), hdim=64, bf16, causal=true
 * 
 * ============================================================================
 * Flash Attention 算法核心思想:
 * ============================================================================
 * 
 * 传统 Attention 计算流程:
 *   1. S = Q @ K^T           (seqlen_q x seqlen_k, 需要 O(N^2) 存储)
 *   2. P = softmax(S)        (需要对整行做 softmax)
 *   3. O = P @ V             (seqlen_q x head_dim)
 * 
 * 问题: 中间矩阵 S 和 P 的大小是 O(N^2)，对于长序列会导致显存爆炸
 * 
 * Flash Attention 解决方案:
 *   - 将 Q, K, V 分块处理 (Tiling)
 *   - 使用 Online Softmax 算法，边计算边更新 softmax
 *   - 利用 GPU 的内存层次结构: HBM -> SRAM (Shared Memory) -> Registers
 *   - 从不在 HBM 中存储完整的 S 或 P 矩阵
 * 
 * ============================================================================
 * Online Softmax 算法:
 * ============================================================================
 * 
 * 标准 softmax: softmax(x_i) = exp(x_i) / sum(exp(x_j))
 * 
 * 为了数值稳定性，实际计算: softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
 * 
 * Online Softmax 的关键洞察:
 * 假设我们已经处理了前面的块，有:
 *   - m_prev: 之前所有元素的最大值
 *   - l_prev: 之前所有 exp(x - m_prev) 的和
 *   - o_prev: 之前累积的输出 (未归一化)
 * 
 * 当处理新的块时:
 *   1. 计算新块的 m_new = max(x_new)
 *   2. 更新全局最大值: m = max(m_prev, m_new)
 *   3. 重新缩放之前的结果:
 *      - l_prev *= exp(m_prev - m)
 *      - o_prev *= exp(m_prev - m)
 *   4. 计算新块的贡献: p_new = exp(x_new - m), l_new = sum(p_new)
 *   5. 更新: l = l_prev + l_new, o = o_prev + p_new @ V_new
 *   6. 最终归一化: O = o / l
 * 
 * ============================================================================
 *****************************************************************************/

#pragma once

// ============================================================================
// 第一部分: 头文件和命名空间
// ============================================================================

// CuTe - NVIDIA 的张量抽象库，用于描述张量布局和操作
#include <cute/tensor.hpp>

// Cutlass - NVIDIA 的 CUDA 模板库，提供高效的 GEMM 实现
#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>

// C10 CUDA 工具 (来自 PyTorch)
#include <c10/cuda/CUDAException.h>

// Flash Attention 内部头文件
#include "namespace_config.h"   // 命名空间配置
#include "flash.h"              // Flash_fwd_params 参数结构
#include "kernel_traits.h"      // Kernel 特性定义
#include "block_info.h"         // 块信息辅助类
#include "utils.h"              // 工具函数
#include "softmax.h"            // Softmax 实现
#include "mask.h"               // Mask 实现

namespace FLASH_NAMESPACE {

using namespace cute;

// ============================================================================
// 第二部分: Kernel 配置常量
// ============================================================================

/**
 * Flash_fwd_kernel_traits 模板参数说明:
 * 
 * ============ 维度对应关系 ============
 * 
 *   Attention 计算: S = Q @ K^T,  O = softmax(S) @ V
 * 
 *   矩阵维度:
 *     Q: [seqlen_q, head_dim]  -> 对应 [M, K] 维度
 *     K: [seqlen_k, head_dim]  -> 对应 [N, K] 维度 (转置后 K^T 是 [K, N])
 *     S: [seqlen_q, seqlen_k]  -> 对应 [M, N] 维度
 *     V: [seqlen_k, head_dim]  -> 对应 [N, K] 维度
 *     O: [seqlen_q, head_dim]  -> 对应 [M, K] 维度
 * 
 *   所以:
 *     M 维度 = seqlen_q (Q 的序列长度)
 *     N 维度 = seqlen_k (K/V 的序列长度)
 *     K 维度 = head_dim (注意力头维度)
 * 
 * ============ 分块参数 ============
 * 
 * kHeadDim = 64      : 每个注意力头的维度 (K 维度)
 * kBlockM  = 128     : 每个线程块处理的 Q 序列行数 (seqlen_q 的分块大小)
 * kBlockN  = 128     : 每次迭代处理的 K/V 序列行数 (seqlen_k 的分块大小)
 * kNWarps  = 4       : 每个线程块使用的 Warp 数量 (4 * 32 = 128 线程)
 * 
 * 为什么选择这些参数?
 * - kBlockM = 128: 每个 SM 可以并行处理多个 M 块 (Q 序列的不同部分)
 * - kBlockN = 128: K/V 的分块大小，需要平衡共享内存使用和计算密度
 * - kNWarps = 4: Tensor Core MMA 操作需要多个 Warp 协作
 */

// 定义 Kernel 特性类型别名
template<typename Element>
using KernelTraits = Flash_fwd_kernel_traits<
    64,     // kHeadDim = 64
    128,    // kBlockM = 128
    128,    // kBlockN = 128
    4,      // kNWarps = 4
    false,  // Is_Q_in_regs = false (Q 不预加载到寄存器)
    false,  // Share_Q_K_smem = false (Q 和 K 不共享 shared memory)
    Element // 数据类型: cutlass::bfloat16_t
>;

// ============================================================================
// 第三部分: 共享内存布局说明
// ============================================================================

/**
 * Flash Attention 的共享内存布局 (以 hdim=64, BlockM=128, BlockN=128 为例):
 * 
 * Shared Memory 总大小 = kSmemSize
 * 
 * 布局结构:
 * +------------------+
 * |       sQ         |  <- Q 的共享内存区域 [128 x 64] = 8KB (bf16)
 * +------------------+
 * |       sK         |  <- K 的共享内存区域 [128 x 64] = 8KB (bf16)
 * +------------------+
 * |       sV         |  <- V 的共享内存区域 [128 x 64] = 8KB (bf16)
 * +------------------+
 * 
 * 总计约 24KB 共享内存 (SM80+ 支持最大 164KB)
 * 
 * Swizzle 技术:
 * - 为了避免 bank conflict，使用 Swizzle 模式重排共享内存访问
 * - Bank conflict: 当同一 warp 中的多个线程访问同一 bank 的不同地址时发生
 * - Swizzle 通过 XOR 操作打乱地址映射，使访问更均匀分布
 */

// ============================================================================
// 第四部分: compute_attn_1rowblock - 核心计算函数 (完全展开版)
// ============================================================================

/**
 * 处理一个 M 块的完整 Attention 计算
 * 
 * @tparam Kernel_traits  Kernel 配置特性
 * @tparam Is_causal      是否使用因果遮罩
 * @param params          Flash Attention 参数
 * @param bidb            当前批次索引
 * @param bidh            当前注意力头索引  
 * @param m_block         当前处理的 M 块索引
 */
template<typename Kernel_traits, bool Is_causal, typename Params>
__device__ __forceinline__ void compute_attn_1rowblock_detailed(
    const Params &params,
    const int bidb,      // batch index
    const int bidh,      // head index
    const int m_block    // M block index: Q 的 **sequence 维度** 分块索引
                         // 处理 Q 序列的第 [m_block*kBlockM, (m_block+1)*kBlockM) 行
) {
    // ========================================================================
    // 4.1 类型定义和常量提取
    // ========================================================================
    
    // 元素类型 (bf16)
    using Element = typename Kernel_traits::Element;
    // 累加器类型 (fp32) - 用于精确计算
    using ElementAccum = typename Kernel_traits::ElementAccum;
    // 索引类型
    using index_t = typename Kernel_traits::index_t;
    
    // 块大小常量
    constexpr int kBlockM = Kernel_traits::kBlockM;     // 128: Q 的行块大小
    constexpr int kBlockN = Kernel_traits::kBlockN;     // 128: K/V 的行块大小  
    constexpr int kHeadDim = Kernel_traits::kHeadDim;   // 64: 头维度
    constexpr int kNWarps = Kernel_traits::kNWarps;     // 4: Warp 数量
    
    // 共享内存指针 (由 kernel launch 时分配)
    extern __shared__ char smem_[];
    
    // 当前线程在 block 内的索引
    const int tidx = threadIdx.x;  // 0 ~ 127 (128 线程)
    
    // ========================================================================
    // 4.2 块信息和边界检查
    // ========================================================================
    
    // BlockInfo 辅助类，处理变长序列的偏移计算
    // Is_even_MN = true 时假设序列长度是块大小的整数倍
    const BlockInfo</*Varlen=*/false> binfo(params, bidb);
    
    // 提前退出: 如果当前 M 块超出实际 Q 序列长度
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;
    
    // ========================================================================
    // 4.3 计算 K/V 的块范围 [n_block_min, n_block_max)
    // ========================================================================
    
    // 确定需要遍历的 K/V 块范围
    // 对于 Causal attention，只需要处理 j <= i 的位置
    int n_block_min = 0;
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    
    if constexpr (Is_causal) {
        // Causal mask: 对于 Q 的位置 i，只关注 K 的位置 j <= i
        // 当前 M 块处理 Q 的行: [m_block * kBlockM, (m_block + 1) * kBlockM)
        // 最大的 Q 行索引是 (m_block + 1) * kBlockM - 1
        // 加上 seqlen_k - seqlen_q 的偏移 (处理 Q/K 长度不同的情况)
        n_block_max = std::min(
            n_block_max,
            cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k 
                          - binfo.actual_seqlen_q, kBlockN)
        );
    }
    
    // ========================================================================
    // 4.4 创建 Global Memory 张量视图
    // ========================================================================
    
    /**
     * CuTe Tensor 基础:
     * 
     * make_tensor(pointer, shape, stride) 创建张量视图:
     * - pointer: 数据指针
     * - shape: 张量形状 (例如 [seqlen_q, num_heads, head_dim])
     * - stride: 每个维度的步长
     * 
     * local_tile: 从大张量中提取局部 tile
     */
    
    // Q 张量: [seqlen_q, num_heads, head_dim]
    // 计算当前 batch 和 head 的 Q 偏移
    Tensor mQ = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
                      + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
        make_shape(binfo.actual_seqlen_q, params.h, params.d),
        make_stride(params.q_row_stride, params.q_head_stride, _1{})
    );
    // 提取当前 head 的 Q，并按 M 块分块
    // gQ: [kBlockM, kHeadDim]
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                          make_coord(m_block, 0));
    
    // K 张量: [seqlen_k, num_heads_k, head_dim]
    Tensor mK = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr)
                      + binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)),
        make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
        make_stride(params.k_row_stride, params.k_head_stride, _1{})
    );
    // gK: [kBlockN, kHeadDim, num_n_blocks] - 所有 N 块的视图
    // h_h_k_ratio: 用于 GQA (Grouped Query Attention)，多个 Q head 共享一个 K head
    Tensor gK = local_tile(mK(_, bidh / params.h_h_k_ratio, _), 
                          Shape<Int<kBlockN>, Int<kHeadDim>>{},
                          make_coord(_, 0));
    
    // V 张量: [seqlen_k, num_heads_k, head_dim]
    Tensor mV = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element*>(params.v_ptr)
                      + binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)),
        make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
        make_stride(params.v_row_stride, params.v_head_stride, _1{})
    );
    // gV: [kBlockN, kHeadDim, num_n_blocks]
    Tensor gV = local_tile(mV(_, bidh / params.h_h_k_ratio, _), 
                          Shape<Int<kBlockN>, Int<kHeadDim>>{},
                          make_coord(_, 0));
    
    // ========================================================================
    // 4.5 创建 Shared Memory 张量
    // ========================================================================
    
    /**
     * 共享内存布局使用 Swizzle 避免 bank conflict:
     * 
     * SmemLayoutQ/KV 使用 Swizzle<3, 3, 3>:
     * - 第一个 3: log2(swizzle bits) = 3，即 8 位参与 swizzle
     * - 后两个 3: 控制 swizzle 的 shift 和 base
     * 
     * Bank conflict 问题:
     * - Shared Memory 分为 32 个 bank，每个 bank 4 字节宽
     * - 同一 warp 的线程访问同一 bank 的不同地址会导致串行化
     * - Swizzle 通过 XOR 操作使访问模式交错，避免冲突
     */
    
    // sQ: Q 的共享内存区域 [kBlockM, kHeadDim] = [128, 64]
    Tensor sQ = make_tensor(
        make_smem_ptr(reinterpret_cast<Element*>(smem_)),
        typename Kernel_traits::SmemLayoutQ{}
    );
    
    // sK: K 的共享内存区域，紧跟在 sQ 之后
    // 如果 Share_Q_K_smem = true，则 sK 与 sQ 共享同一区域
    Tensor sK = make_tensor(
        sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)),
        typename Kernel_traits::SmemLayoutKV{}
    );
    
    // sV: V 的共享内存区域，紧跟在 sK 之后
    Tensor sV = make_tensor(sK.data() + size(sK), 
                            typename Kernel_traits::SmemLayoutKV{});
    
    // sVt: V 的转置视图 (用于 P @ V 计算)
    // 矩阵乘法 P @ V 中，V 需要按列访问，转置后变成按行访问
    Tensor sVt = make_tensor(sV.data(), 
                             typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(
        sV.data().get(), 
        typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{}
    );
    
    // ========================================================================
    // 4.6 创建 Global Memory 到 Shared Memory 的拷贝器
    // ========================================================================
    
    /**
     * GmemTiledCopyQKV 说明:
     * 
     * 使用 cp.async 指令实现异步拷贝:
     * - SM80 引入的新指令，允许 Global -> Shared 的异步传输
     * - CPU/GPU 计算可以与内存传输重叠
     * - 每个线程一次加载 128 位 (8 个 bf16 元素)
     * 
     * 线程布局: GmemLayoutAtom
     * - 例如 [16, 8]: 16 行 x 8 列的线程组织
     * - 每行 8 个线程，每个线程加载 8 个元素
     * - 总共覆盖 [16, 64] 的数据块
     */
    
    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
    
    // 分区: 将全局张量按线程分配
    // tQgQ: 当前线程负责加载的 Q 数据 (从 Global Memory)
    // tQsQ: 当前线程负责存储的 Q 位置 (到 Shared Memory)
    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    
    // 类似地为 K 和 V 创建分区
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // [KCPY, KCPY_N, KCPY_K, num_n_blocks]
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);
    
    // ========================================================================
    // 4.7 创建 TiledMMA (Tensor Core Matrix Multiply-Accumulate)
    // ========================================================================
    
    /**
     * TiledMMA 说明:
     * 
     * Tensor Core 是 NVIDIA GPU 上的专用矩阵乘法单元:
     * - SM80 (Ampere) 使用 MMA m16n8k16 指令
     * - 每条指令计算 16x8x16 的矩阵乘法片段
     * 
     * MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>:
     * - 输入: BF16
     * - 累加器: FP32  
     * - 布局: A 转置，B 正常 (TN)
     * 
     * TiledMMA 将多个 MMA atoms 组合:
     * - 4 个 warp (128 线程) 协作
     * - 计算 [64, 16] x [16, 64] -> [64, 64] 的片段
     */
    
    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    
    // 为 MMA 创建寄存器 fragment
    // tSrQ: Q 的寄存器片段 (用于 S = Q @ K^T)
    Tensor tSrQ = thr_mma.partition_fragment_A(sQ);   // [MMA, MMA_M, MMA_K]
    // tSrK: K 的寄存器片段
    Tensor tSrK = thr_mma.partition_fragment_B(sK);   // [MMA, MMA_N, MMA_K]
    // tOrVt: V^T 的寄存器片段 (用于 O = P @ V)
    Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle);
    
    // 输出累加器: 存储最终结果
    // acc_o: [MMA, MMA_M, MMA_K] -> 形状对应 [kBlockM, kHeadDim]
    Tensor acc_o = partition_fragment_C(tiled_mma, 
                                       Shape<Int<kBlockM>, Int<kHeadDim>>{});
    
    // ========================================================================
    // 4.8 创建 Shared Memory 到 Register 的拷贝器
    // ========================================================================
    
    /**
     * Shared Memory 到寄存器的高效拷贝:
     * 
     * 使用 LDSM (Load Matrix from Shared Memory) 指令:
     * - SM75_U32x4_LDSM_N: 从共享内存加载 4x32 位数据
     * - 一条指令加载矩阵片段，优化 Tensor Core 输入
     * 
     * 转置版本 SM75_U16x8_LDSM_T:
     * - 用于 V 的转置读取
     */
    
    auto smem_tiled_copy_Q = make_tiled_copy_A(
        typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);
    
    auto smem_tiled_copy_K = make_tiled_copy_B(
        typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);
    
    auto smem_tiled_copy_V = make_tiled_copy_B(
        typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);
    
    // ========================================================================
    // 4.9 创建边界检查谓词
    // ========================================================================
    
    /**
     * 边界处理:
     * 
     * 当序列长度不是块大小的整数倍时，需要处理边界情况:
     * - Is_even_MN: 如果 seqlen_q % kBlockM == 0 且 seqlen_k % kBlockN == 0
     * - Is_even_K: 如果 head_dim == kHeadDim (64)
     * 
     * 使用 identity tensor 和谓词来跟踪哪些位置是有效的
     */
    
    // 创建身份张量用于索引计算
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));
    
    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);
    
    // K 维度的谓词 (假设 Is_even_K = true，head_dim = 64)
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));
    
    #pragma unroll
    for (int k = 0; k < size(tQpQ); ++k) {
        tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d;
    }
    #pragma unroll
    for (int k = 0; k < size(tKVpKV); ++k) {
        tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d;
    }
    
    // ========================================================================
    // 4.10 Prologue: 加载 Q 到 Shared Memory
    // ========================================================================
    
    /**
     * 数据加载流程:
     * 
     * 1. Q 只加载一次 (对于当前 M 块)
     * 2. K, V 每次迭代加载一个 N 块
     * 3. 使用 cp.async 实现异步加载
     * 
     * cp_async_fence(): 标记 async group 的边界
     * cp_async_wait<N>(): 等待直到最多 N 个 group 未完成
     */
    
    // 加载 Q 从 Global Memory 到 Shared Memory
    // copy<Is_even_MN, Is_even_K>: 模板参数控制是否需要边界检查
    FLASH_NAMESPACE::copy</*Is_even_MN=*/true, /*Is_even_K=*/true>(
        gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
        binfo.actual_seqlen_q - m_block * kBlockM
    );
    
    // 如果 Q 需要预加载到寄存器
    if constexpr (Kernel_traits::Is_Q_in_regs) {
        cute::cp_async_fence();
    }
    
    // 开始从最后一个 N 块向前迭代 (反向迭代可以节省一个寄存器)
    int n_block = n_block_max - 1;
    
    // 加载第一个 K 块
    FLASH_NAMESPACE::copy</*Is_even_MN=*/true, /*Is_even_K=*/true>(
        gmem_tiled_copy_QKV, tKgK(_, _, _, n_block), tKsK, tKVcKV, tKVpKV,
        binfo.actual_seqlen_k - n_block * kBlockN
    );
    cute::cp_async_fence();
    
    // 清空输出累加器
    clear(acc_o);
    
    // ========================================================================
    // 4.11 初始化 Softmax 状态
    // ========================================================================
    
    /**
     * Online Softmax 状态变量:
     * 
     * row_max: 每行的当前最大值 (用于数值稳定性)
     * row_sum: 每行的 exp(x - max) 之和 (用于归一化)
     * 
     * Softmax 类封装了 online softmax 的计算逻辑
     */
    
    // 2 * size<1>(acc_o) = 2 * MMA_M = 每个线程处理的行数
    FLASH_NAMESPACE::Softmax<2 * size<1>(acc_o)> softmax;
    
    // 创建 Mask 对象用于 causal masking
    FLASH_NAMESPACE::Mask<Is_causal, /*Is_local=*/false, /*Has_alibi=*/false> mask(
        binfo.actual_seqlen_k, 
        binfo.actual_seqlen_q,
        /*window_size_left=*/-1, 
        /*window_size_right=*/0,  // Causal: 只能看到当前位置及之前
        /*alibi_slope=*/0.f
    );
    
    // ========================================================================
    // 4.12 主循环: 遍历所有 K/V 块
    // ========================================================================
    
    /**
     * 主循环结构:
     * 
     * 分为两部分:
     * 1. 需要 masking 的迭代 (最后几个块，因为 causal mask 可能切断)
     * 2. 不需要 masking 的迭代 (完整块)
     * 
     * 每次迭代:
     * 1. 计算 S = Q @ K^T (部分)
     * 2. 应用 mask 和 softmax
     * 3. 计算 O += P @ V (累积)
     * 4. 预取下一个 K 块
     */
    
    // 需要 masking 的步数
    // Causal mask 可能在最后 ceil_div(kBlockM, kBlockN) 个块中生效
    constexpr int n_masking_steps = Is_causal ? 
        cute::ceil_div(kBlockM, kBlockN) + 1 : 1;
    
    // ========== 阶段 1: 需要 masking 的迭代 ==========
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        
        // 4.12.1 创建 score 累加器
        // acc_s 存储 S = Q @ K^T 的结果
        Tensor acc_s = partition_fragment_C(tiled_mma, 
                                           Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(acc_s);
        
        // 等待 K 加载完成
        FLASH_NAMESPACE::cp_async_wait<0>();
        __syncthreads();
        
        // 4.12.2 加载 V (下一次迭代需要)
        if (masking_step > 0) {
            FLASH_NAMESPACE::copy</*Is_even_MN=*/true, /*Is_even_K=*/true>(
                gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV
            );
        } else {
            // 第一次迭代需要处理边界
            FLASH_NAMESPACE::copy</*Is_even_MN=*/true, /*Is_even_K=*/true, 
                                 /*Clear_OOB_MN=*/true>(
                gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV,
                binfo.actual_seqlen_k - n_block * kBlockN
            );
        }
        cute::cp_async_fence();
        
        // 4.12.3 计算 S = Q @ K^T
        /**
         * gemm 函数执行分块矩阵乘法:
         * 
         * for k in 0..kHeadDim/MMA_K:
         *     从 sQ 加载 Q 片段到寄存器
         *     从 sK 加载 K 片段到寄存器  
         *     acc_s += Q_frag @ K_frag (使用 Tensor Core MMA 指令)
         * 
         * 使用软件流水线: 加载下一个 K 片段的同时计算当前片段
         */
        FLASH_NAMESPACE::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, 
            smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );
        
        // 4.12.4 应用 Causal Mask
        /**
         * Causal Mask 应用:
         * 
         * 对于位置 (i, j):
         * - 如果 j > i + (seqlen_k - seqlen_q)，则 mask 为 -inf
         * - 这确保位置 i 只能看到位置 j <= i 的信息
         */
        mask.template apply_mask<Is_causal, /*Is_even_MN=*/true>(
            acc_s, 
            n_block * kBlockN,                                    // col_idx_offset
            m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, // row_idx_offset
            kNWarps * 16                                          // warp_row_stride
        );
        
        // 等待 V 加载完成
        FLASH_NAMESPACE::cp_async_wait<0>();
        __syncthreads();
        
        // 4.12.5 预取下一个 K 块
        if (n_block > n_block_min) {
            FLASH_NAMESPACE::copy</*Is_even_MN=*/true, /*Is_even_K=*/true>(
                gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK, tKVcKV, tKVpKV
            );
            cute::cp_async_fence();
        }
        
        // 4.12.6 Online Softmax + 更新输出
        /**
         * softmax_rescale_o 执行:
         * 
         * Is_first = true (第一次迭代):
         *   1. row_max = reduce_max(acc_s)      // 找到每行最大值
         *   2. acc_s = exp(acc_s - row_max)     // 计算 exp
         *   3. row_sum = reduce_sum(acc_s)      // 计算 exp 的和
         * 
         * Is_first = false (后续迭代):
         *   1. row_max_prev = row_max
         *   2. row_max = max(row_max, reduce_max(acc_s))  // 更新最大值
         *   3. scale = exp(row_max_prev - row_max)
         *   4. row_sum *= scale                 // 重新缩放之前的和
         *   5. acc_o *= scale                   // 重新缩放之前的输出
         *   6. acc_s = exp(acc_s - row_max)
         *   7. row_sum += reduce_sum(acc_s)
         */
        if (masking_step == 0) {
            softmax.template softmax_rescale_o</*Is_first=*/true, /*Check_inf=*/Is_causal>(
                acc_s, acc_o, params.scale_softmax_log2
            );
        } else {
            softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal>(
                acc_s, acc_o, params.scale_softmax_log2
            );
        }
        
        // 4.12.7 将 acc_s (fp32) 转换为 P (bf16) 用于矩阵乘法
        Tensor rP = FLASH_NAMESPACE::convert_type<Element>(acc_s);
        
        // 4.12.8 计算 O += P @ V
        /**
         * 重排 P 的布局以匹配 MMA 指令的输入要求:
         * 
         * acc_s: (MMA=4, MMA_M, MMA_N) 
         *     -> ((4, 2), MMA_M, MMA_N / 2) for m16n8k16
         * 
         * gemm_rs: P 从寄存器读取，V 从共享内存读取
         */
        Tensor tOrP = make_tensor(rP.data(), 
            FLASH_NAMESPACE::convert_layout_acc_Aregs<typename Kernel_traits::TiledMma>(rP.layout())
        );
        FLASH_NAMESPACE::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, 
                                smem_tiled_copy_V, smem_thr_copy_V);
        
        // 检查是否需要提前退出
        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
    }
    
    // ========== 阶段 2: 不需要 masking 的迭代 ==========
    // 这些块不会被 causal mask 截断，可以跳过 mask 检查
    for (; n_block >= n_block_min; --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, 
                                           Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(acc_s);
        
        FLASH_NAMESPACE::cp_async_wait<0>();
        __syncthreads();
        
        // 加载 V
        FLASH_NAMESPACE::copy</*Is_even_MN=*/true, /*Is_even_K=*/true>(
            gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV
        );
        cute::cp_async_fence();
        
        // 计算 S = Q @ K^T
        FLASH_NAMESPACE::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, 
            smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );
        
        FLASH_NAMESPACE::cp_async_wait<0>();
        __syncthreads();
        
        // 预取下一个 K
        if (n_block > n_block_min) {
            FLASH_NAMESPACE::copy</*Is_even_MN=*/true, /*Is_even_K=*/true>(
                gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK, tKVcKV, tKVpKV
            );
            cute::cp_async_fence();
        }
        
        // 不需要 causal mask，但仍需要处理序列边界
        mask.template apply_mask</*Causal_mask=*/false>(
            acc_s, 
            n_block * kBlockN,
            m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4,
            kNWarps * 16
        );
        
        // Softmax + 更新输出
        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/false>(
            acc_s, acc_o, params.scale_softmax_log2
        );
        
        Tensor rP = FLASH_NAMESPACE::convert_type<Element>(acc_s);
        Tensor tOrP = make_tensor(rP.data(), 
            FLASH_NAMESPACE::convert_layout_acc_Aregs<typename Kernel_traits::TiledMma>(rP.layout())
        );
        FLASH_NAMESPACE::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, 
                                smem_tiled_copy_V, smem_thr_copy_V);
    }
    
    // ========================================================================
    // 4.13 Epilogue: 归一化并写回输出
    // ========================================================================
    
    /**
     * 最终归一化:
     * 
     * O_final = acc_o / row_sum
     * LSE = log(row_sum) + row_max * softmax_scale
     * 
     * LSE (Log-Sum-Exp) 用于:
     * 1. 反向传播时重建 softmax 概率
     * 2. 验证数值正确性
     */
    
    // 归一化输出并获取 LSE
    Tensor lse = softmax.template normalize_softmax_lse</*Is_dropout=*/false>(
        acc_o, params.scale_softmax, /*rp_dropout=*/1.0
    );
    
    // 将 acc_o 从 fp32 转换为 bf16
    Tensor rO = FLASH_NAMESPACE::convert_type<Element>(acc_o);
    
    // 创建共享内存中的输出区域
    Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});
    
    // 从寄存器拷贝到共享内存
    auto smem_tiled_copy_O = make_tiled_copy_C(
        typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);
    
    // 如果 Q 和 K 共享内存，需要同步
    if constexpr (Kernel_traits::Share_Q_K_smem) {
        __syncthreads();
    }
    
    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);
    
    // 创建输出的 Global Memory 张量
    Tensor mO = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                      + binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)),
        make_shape(binfo.actual_seqlen_q, params.h, params.d),
        make_stride(params.o_row_stride, params.o_head_stride, _1{})
    );
    Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                          make_coord(m_block, 0));
    
    // LSE 输出张量
    auto lse_offset = 0;  // 简化版本，假设 unpadded_lse = false
    auto gmem_ptr_lse = make_gmem_ptr(
        reinterpret_cast<ElementAccum*>(params.softmax_lse_ptr) + lse_offset);
    Tensor mLSE = make_tensor(gmem_ptr_lse, 
        make_shape(params.b, params.h, params.seqlen_q),
        make_stride(params.h * params.seqlen_q, params.seqlen_q, _1{}));
    Tensor gLSE = local_tile(mLSE(bidb, bidh, _), Shape<Int<kBlockM>>{}, 
                             make_coord(m_block));
    
    // 从共享内存拷贝到全局内存
    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
    
    __syncthreads();
    
    Tensor tOrO_final = make_tensor<Element>(shape(tOgO));
    cute::copy(gmem_tiled_copy_O, tOsO, tOrO_final);
    
    // 处理边界和写回
    Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
    
    #pragma unroll
    for (int k = 0; k < size(tOpO); ++k) {
        tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d;
    }
    
    // 写回输出
    FLASH_NAMESPACE::copy</*Is_even_MN=*/true, /*Is_even_K=*/true, 
                         /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_O, tOrO_final, tOgO, tOcO, tOpO,
        binfo.actual_seqlen_q - m_block * kBlockM
    );
    
    // 写回 LSE
    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
    Tensor taccOcO = thr_mma.partition_C(caccO);
    Tensor taccOcO_row = logical_divide(taccOcO, Shape<_2>{})(make_coord(0, _), _, 0);
    
    if (get<1>(taccOcO_row(0)) == 0) {
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO_row(mi));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM) {
                gLSE(row) = lse(mi);
            }
        }
    }
}

// ============================================================================
// 第五部分: compute_attn - Kernel 入口点
// ============================================================================

/**
 * 这是 GPU kernel 的入口函数
 * 每个 thread block 处理一个 (M_block, batch, head) 的组合
 */
template<typename Kernel_traits, bool Is_causal, typename Params>
__device__ __forceinline__ void compute_attn_detailed(const Params &params) {
    // blockIdx.x: M 块索引 (Q 的分块)
    const int m_block = blockIdx.x;
    // blockIdx.y: batch 索引
    const int bidb = blockIdx.y;
    // blockIdx.z: head 索引
    const int bidh = blockIdx.z;
    
    // 调用核心计算函数
    compute_attn_1rowblock_detailed<Kernel_traits, Is_causal>(params, bidb, bidh, m_block);
}

// ============================================================================
// 第六部分: flash_fwd_kernel - GPU Kernel 定义
// ============================================================================

/**
 * GPU Kernel 宏定义
 * 
 * KERNEL_PARAM_MODIFIER: 
 *   - SM80+: __grid_constant__，表示参数不会被修改，允许优化
 *   - 其他: 无修饰符
 */
template<typename Kernel_traits, bool Is_causal>
__global__ void flash_fwd_kernel_detailed(
    const Flash_fwd_params params  // 注意: 使用 __grid_constant__ 需要 const
) {
    compute_attn_detailed<Kernel_traits, Is_causal>(params);
}

// ============================================================================
// 第七部分: run_flash_fwd_detailed - Host 端启动函数
// ============================================================================

/**
 * 这是从 CPU 端调用的函数，负责:
 * 1. 计算 grid 和 block 维度
 * 2. 设置共享内存大小
 * 3. 启动 kernel
 */
template<typename Kernel_traits, bool Is_causal>
void run_flash_fwd_detailed(Flash_fwd_params &params, cudaStream_t stream) {
    
    // 共享内存大小
    constexpr size_t smem_size = Kernel_traits::kSmemSize;
    
    /**
     * Grid 维度计算:
     * 
     * grid.x = ceil(seqlen_q / kBlockM)  : M 方向的块数
     * grid.y = batch_size                 : batch 维度
     * grid.z = num_heads                  : head 维度
     * 
     * Block 维度:
     * kNThreads = kNWarps * 32 = 4 * 32 = 128 线程
     */
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.b, params.h);
    
    // 获取 kernel 函数指针
    auto kernel = &flash_fwd_kernel_detailed<Kernel_traits, Is_causal>;
    
    // 如果共享内存超过 48KB，需要动态分配
    if (smem_size >= 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel, 
            cudaFuncAttributeMaxDynamicSharedMemorySize, 
            smem_size
        ));
    }
    
    // 启动 kernel
    kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ============================================================================
// 第八部分: run_mha_fwd_hdim64_detailed - 顶层调度函数
// ============================================================================

/**
 * 这是最外层的调度函数
 * 
 * 对于 hdim=64:
 * - 使用 kBlockM=128, kBlockN=128
 * - 4 个 warps (128 线程)
 * 
 * DROPOUT_SWITCH: 根据是否有 dropout 选择不同的模板实例
 */
template<typename T, bool Is_causal>
void run_mha_fwd_hdim64_detailed(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 64;
    
    // 检查是否有 dropout (这里假设没有)
    constexpr bool Is_dropout = false;
    
    if constexpr (!Is_dropout) {
        // 不使用 dropout 时的配置
        // kBlockM=128, kBlockN=128 对于 seqlen=2k 最优
        run_flash_fwd_detailed<
            Flash_fwd_kernel_traits<Headdim, 128, 128, 4, false, false, T>,
            Is_causal
        >(params, stream);
    } else {
        // 使用 dropout 时的配置
        // kBlockN=64 以减少共享内存使用
        run_flash_fwd_detailed<
            Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, T>,
            Is_causal
        >(params, stream);
    }
}

// ============================================================================
// 第九部分: 显式模板实例化
// ============================================================================

/**
 * 为什么需要显式实例化?
 * 
 * 1. 减少编译时间: 不同 cu 文件可以并行编译
 * 2. 控制代码大小: 只生成需要的模板实例
 * 3. 隐藏实现细节: 头文件只需要声明，不需要完整定义
 */

// 显式实例化 run_mha_fwd_ 模板
// T = cutlass::bfloat16_t, Headdim = 64, Is_causal = true
template<>
void run_mha_fwd_<cutlass::bfloat16_t, 64, true>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim64_detailed<cutlass::bfloat16_t, true>(params, stream);
}

} // namespace FLASH_NAMESPACE

// ============================================================================
// 附录: 关键数据结构说明
// ============================================================================

/*
Flash_fwd_params 结构 (定义在 flash.h):
{
    // QKV 指针
    void *q_ptr, *k_ptr, *v_ptr;
    
    // 输出指针
    void *o_ptr;          // 主输出
    void *softmax_lse_ptr; // Log-Sum-Exp 输出
    
    // 维度
    int b;           // batch size
    int h, h_k;      // Q 和 K 的 head 数量
    int seqlen_q;    // Q 的序列长度
    int seqlen_k;    // K 的序列长度  
    int d;           // head 维度
    
    // 步长 (用于计算内存偏移)
    int64_t q_batch_stride, q_row_stride, q_head_stride;
    int64_t k_batch_stride, k_row_stride, k_head_stride;
    int64_t v_batch_stride, v_row_stride, v_head_stride;
    int64_t o_batch_stride, o_row_stride, o_head_stride;
    
    // 缩放因子
    float scale_softmax;      // 1/sqrt(d)
    float scale_softmax_log2; // log2(e) / sqrt(d), 用于快速 exp2 计算
    
    // 其他配置
    bool is_causal;    // 是否使用 causal mask
    float p_dropout;   // dropout 概率
}

Flash_fwd_kernel_traits 结构 (定义在 kernel_traits.h):
{
    // 数据类型
    using Element = cutlass::bfloat16_t;
    using ElementAccum = float;
    
    // 块大小
    static constexpr int kBlockM = 128;
    static constexpr int kBlockN = 128;
    static constexpr int kHeadDim = 64;
    
    // 线程配置
    static constexpr int kNWarps = 4;
    static constexpr int kNThreads = 128;
    
    // 共享内存布局
    using SmemLayoutQ = ...;
    using SmemLayoutKV = ...;
    
    // 拷贝操作
    using GmemTiledCopyQKV = ...;  // Global -> Shared
    using SmemCopyAtom = ...;      // Shared -> Register
    
    // MMA 操作
    using TiledMma = ...;          // Tensor Core MMA
}
*/

/*
CuTe Tensor 核心概念:

1. Tensor = Data + Layout
   - Data: 指向实际数据的指针
   - Layout: 描述如何将逻辑索引映射到物理地址

2. Layout = Shape + Stride
   - Shape: 张量的形状 (例如 [128, 64])
   - Stride: 每个维度的步长 (例如 [64, 1])

3. 访问公式:
   physical_addr = data_ptr + sum(logical_index[i] * stride[i])

4. 常用操作:
   - make_tensor: 创建张量视图
   - local_tile: 从大张量提取局部 tile
   - partition_S/D: 为 copy 操作分区源/目标
   - partition_fragment_A/B/C: 为 MMA 操作分区

5. Swizzle 机制:
   - 通过 XOR 操作重排索引
   - 目的: 避免 shared memory bank conflict
   - 公式: new_addr = old_addr ^ swizzle_pattern
*/

/*
Tensor Core MMA 指令说明:

SM80_16x8x16_F32BF16BF16F32_TN:
- 16x8x16: 矩阵片段大小 (M=16, N=8, K=16)
- F32: 累加器精度
- BF16: 输入精度
- TN: A 转置，B 正常

每条 MMA 指令:
- 1 个 warp (32 线程) 协作执行
- 计算 16x8x16 的矩阵乘法
- 输入: 2 个 bf16 矩阵片段
- 输出: 1 个 fp32 矩阵片段

TiledMMA 组合多个 MMA 指令:
- 4 个 warp 组成 TiledMMA
- 计算 64x64 或更大的矩阵片段
*/

