// Flash Attention Test
// Compares Flash Attention output with naive PyTorch implementation

#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <chrono>

#include <cutlass/numeric_types.h>

// Flash Attention headers
#include "namespace_config.h"
#include "flash.h"

// Declaration of the Flash Attention kernel function
namespace flash {
    template<typename T, int Headdim, bool Is_causal>
    void run_mha_fwd_(Flash_fwd_params &params, cudaStream_t stream);
}

// ============================================================================
// Naive Flash Attention Implementation (PyTorch-based)
// ============================================================================

// Naive attention: O = softmax(Q @ K^T / sqrt(d)) @ V with causal mask
torch::Tensor naive_attention_forward(
    const torch::Tensor& Q,  // [batch, seqlen_q, num_heads, head_dim]
    const torch::Tensor& K,  // [batch, seqlen_k, num_heads, head_dim]
    const torch::Tensor& V,  // [batch, seqlen_k, num_heads, head_dim]
    float softmax_scale,
    bool is_causal
) {
    // Transpose to [batch, num_heads, seqlen, head_dim] for easier computation
    auto q = Q.transpose(1, 2);  // [batch, num_heads, seqlen_q, head_dim]
    auto k = K.transpose(1, 2);  // [batch, num_heads, seqlen_k, head_dim]
    auto v = V.transpose(1, 2);  // [batch, num_heads, seqlen_k, head_dim]

    // Compute attention scores: S = Q @ K^T
    auto scores = torch::matmul(q, k.transpose(-2, -1)) * softmax_scale;
    // scores: [batch, num_heads, seqlen_q, seqlen_k]

    // Apply causal mask if needed
    if (is_causal) {
        int seqlen_q = Q.size(1);
        int seqlen_k = K.size(1);
        // Create causal mask: mask[i, j] = True if j > i (for same seqlen)
        // For different seqlens: mask[i, j] = True if j > i + (seqlen_k - seqlen_q)
        auto mask = torch::ones({seqlen_q, seqlen_k}, Q.options().dtype(torch::kBool));
        for (int i = 0; i < seqlen_q; i++) {
            for (int j = 0; j < seqlen_k; j++) {
                // Causal: can only attend to position j if j <= i + (seqlen_k - seqlen_q)
                if (j > i + (seqlen_k - seqlen_q)) {
                    mask[i][j] = true;
                } else {
                    mask[i][j] = false;
                }
            }
        }
        // Apply mask with -inf
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), -std::numeric_limits<float>::infinity());
    }

    // Softmax
    auto attn_weights = torch::softmax(scores.to(torch::kFloat32), -1).to(Q.dtype());

    // Compute output: O = attn_weights @ V
    auto out = torch::matmul(attn_weights, v);  // [batch, num_heads, seqlen_q, head_dim]

    // Transpose back to [batch, seqlen_q, num_heads, head_dim]
    return out.transpose(1, 2).contiguous();
}

// ============================================================================
// Flash Attention Wrapper
// ============================================================================

torch::Tensor flash_attention_forward(
    const torch::Tensor& Q,  // [batch, seqlen_q, num_heads, head_dim]
    const torch::Tensor& K,  // [batch, seqlen_k, num_heads_k, head_dim]
    const torch::Tensor& V,  // [batch, seqlen_k, num_heads_k, head_dim]
    float softmax_scale,
    bool is_causal
) {
    TORCH_CHECK(Q.is_cuda(), "Q must be on CUDA");
    TORCH_CHECK(K.is_cuda(), "K must be on CUDA");
    TORCH_CHECK(V.is_cuda(), "V must be on CUDA");
    TORCH_CHECK(Q.dtype() == torch::kBFloat16, "Q must be bf16");

    const int batch_size = Q.size(0);
    const int seqlen_q = Q.size(1);
    const int num_heads = Q.size(2);
    const int head_dim = Q.size(3);
    const int seqlen_k = K.size(1);
    const int num_heads_k = K.size(2);

    TORCH_CHECK(head_dim == 64, "This kernel only supports head_dim=64");

    // Create output tensor
    auto out = torch::empty_like(Q);
    
    // Create softmax LSE tensor
    auto softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, 
                                     Q.options().dtype(torch::kFloat32));

    // Round up to multiples
    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_rounded = round_multiple(head_dim, 32);
    const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

    // Set up Flash_fwd_params
    flash::Flash_fwd_params params = {};
    
    params.is_bf16 = true;
    
    // Set pointers
    params.q_ptr = Q.data_ptr();
    params.k_ptr = K.data_ptr();
    params.v_ptr = V.data_ptr();
    params.o_ptr = out.data_ptr();
    
    // Set strides
    params.q_batch_stride = Q.stride(0);
    params.k_batch_stride = K.stride(0);
    params.v_batch_stride = V.stride(0);
    params.o_batch_stride = out.stride(0);
    
    params.q_row_stride = Q.stride(1);
    params.k_row_stride = K.stride(1);
    params.v_row_stride = V.stride(1);
    params.o_row_stride = out.stride(1);
    
    params.q_head_stride = Q.stride(2);
    params.k_head_stride = K.stride(2);
    params.v_head_stride = V.stride(2);
    params.o_head_stride = out.stride(2);
    
    // Set dimensions
    params.b = batch_size;
    params.h = num_heads;
    params.h_k = num_heads_k;
    params.h_h_k_ratio = num_heads / num_heads_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d = head_dim;
    params.d_rounded = head_size_rounded;
    
    // Softmax parameters
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;
    params.softcap = 0.0f;
    
    // Softmax LSE
    params.softmax_lse_ptr = softmax_lse.data_ptr();
    
    // No dropout
    params.p_dropout = 1.0f;
    params.p_dropout_in_uint8_t = 255;
    params.rp_dropout = 1.0f;
    params.scale_softmax_rp_dropout = softmax_scale;
    
    // Causal parameters
    params.is_causal = is_causal;
    params.window_size_left = -1;
    params.window_size_right = is_causal ? 0 : -1;
    
    // No variable length
    params.cu_seqlens_q = nullptr;
    params.cu_seqlens_k = nullptr;
    params.seqused_k = nullptr;
    params.leftpad_k = nullptr;
    
    // No paged KV cache
    params.block_table = nullptr;
    params.page_block_size = 0;
    
    // No alibi
    params.alibi_slopes_ptr = nullptr;
    
    // No rotary
    params.rotary_dim = 0;
    
    // No new KV
    params.knew_ptr = nullptr;
    params.vnew_ptr = nullptr;
    params.seqlen_knew = 0;
    
    // No split KV
    params.num_splits = 1;
    params.oaccum_ptr = nullptr;
    params.softmax_lseaccum_ptr = nullptr;
    
    // No dropout return
    params.p_ptr = nullptr;
    params.rng_state = nullptr;
    
    // Other flags
    params.is_seqlens_k_cumulative = true;
    params.unpadded_lse = false;
    params.seqlenq_ngroups_swapped = false;
    
    // Get CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    
    // Call Flash Attention kernel
    flash::run_mha_fwd_<cutlass::bfloat16_t, 64, true>(params, stream);
    
    // Synchronize
    cudaDeviceSynchronize();
    
    return out;
}

// ============================================================================
// Test Function
// ============================================================================

void test_flash_attention() {
    std::cout << "=== Flash Attention Test (SM89, hdim64, bf16, causal) ===" << std::endl;
    
    // Test parameters
    const int batch_size = 2;
    const int seqlen = 256;
    const int num_heads = 8;
    const int head_dim = 64;
    const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const bool is_causal = true;
    
    std::cout << "Test config:" << std::endl;
    std::cout << "  batch_size = " << batch_size << std::endl;
    std::cout << "  seqlen = " << seqlen << std::endl;
    std::cout << "  num_heads = " << num_heads << std::endl;
    std::cout << "  head_dim = " << head_dim << std::endl;
    std::cout << "  softmax_scale = " << softmax_scale << std::endl;
    std::cout << "  is_causal = " << (is_causal ? "true" : "false") << std::endl;
    std::cout << std::endl;
    
    // Create random input tensors (bf16)
    torch::manual_seed(42);
    auto options = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA, 0);
    
    auto Q = torch::randn({batch_size, seqlen, num_heads, head_dim}, options);
    auto K = torch::randn({batch_size, seqlen, num_heads, head_dim}, options);
    auto V = torch::randn({batch_size, seqlen, num_heads, head_dim}, options);
    
    // Make contiguous with last dim stride = 1
    Q = Q.contiguous();
    K = K.contiguous();
    V = V.contiguous();
    
    std::cout << "Running naive attention..." << std::endl;
    auto start_naive = std::chrono::high_resolution_clock::now();
    auto out_naive = naive_attention_forward(Q, K, V, softmax_scale, is_causal);
    cudaDeviceSynchronize();
    auto end_naive = std::chrono::high_resolution_clock::now();
    auto duration_naive = std::chrono::duration_cast<std::chrono::microseconds>(end_naive - start_naive).count();
    std::cout << "Naive attention time: " << duration_naive << " us" << std::endl;
    
    std::cout << "Running Flash Attention..." << std::endl;
    auto start_flash = std::chrono::high_resolution_clock::now();
    auto out_flash = flash_attention_forward(Q, K, V, softmax_scale, is_causal);
    cudaDeviceSynchronize();
    auto end_flash = std::chrono::high_resolution_clock::now();
    auto duration_flash = std::chrono::duration_cast<std::chrono::microseconds>(end_flash - start_flash).count();
    std::cout << "Flash Attention time: " << duration_flash << " us" << std::endl;
    
    std::cout << std::endl;
    
    // Compare outputs
    std::cout << "Comparing outputs..." << std::endl;
    
    // Convert to float for comparison
    auto out_naive_f32 = out_naive.to(torch::kFloat32);
    auto out_flash_f32 = out_flash.to(torch::kFloat32);
    
    // Compute difference
    auto diff = (out_flash_f32 - out_naive_f32).abs();
    
    // Max absolute error
    float max_abs_error = diff.max().item<float>();
    std::cout << "Max absolute error: " << max_abs_error << std::endl;
    
    // Mean absolute error
    float mean_abs_error = diff.mean().item<float>();
    std::cout << "Mean absolute error: " << mean_abs_error << std::endl;
    
    // Relative error (avoid division by zero)
    auto ref_abs = out_naive_f32.abs() + 1e-6f;
    auto rel_error = diff / ref_abs;
    float max_rel_error = rel_error.max().item<float>();
    float mean_rel_error = rel_error.mean().item<float>();
    std::cout << "Max relative error: " << max_rel_error << std::endl;
    std::cout << "Mean relative error: " << mean_rel_error << std::endl;
    
    // Check if test passed (bf16 has ~3 decimal places of precision)
    // bf16 max error can be around 0.01-0.02, so we use 0.02 as threshold
    bool passed = max_abs_error < 0.02f && mean_abs_error < 0.001f;
    
    std::cout << std::endl;
    if (passed) {
        std::cout << "✓ Test PASSED!" << std::endl;
    } else {
        std::cout << "✗ Test FAILED!" << std::endl;
    }
    
    // Print sample values
    std::cout << std::endl;
    std::cout << "Sample values (first element of first batch/head):" << std::endl;
    std::cout << "  Naive: " << out_naive_f32[0][0][0][0].item<float>() << std::endl;
    std::cout << "  Flash: " << out_flash_f32[0][0][0][0].item<float>() << std::endl;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    // Check CUDA availability
    if (!torch::cuda::is_available()) {
        std::cerr << "CUDA is not available!" << std::endl;
        return 1;
    }
    
    std::cout << "CUDA device count: " << torch::cuda::device_count() << std::endl;
    
    try {
        test_flash_attention();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

