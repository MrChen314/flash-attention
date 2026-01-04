// Copyright (c) 2024, Tri Dao.
// SM89 (Ada Lovelace) version of Flash Attention Forward
// hdim=64, bf16, causal=true

// Cutlass and CuTe headers
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>

// Flash Attention headers
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

// Explicit template instantiation for SM89
// bf16, hdim=64, causal=true
// SM89 uses the same Tensor Core MMA instructions as SM80,
// so we can directly reuse the SM80 kernel implementation
template<>
void run_mha_fwd_<cutlass::bfloat16_t, 64, true>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim64<cutlass::bfloat16_t, true>(params, stream);
}

} // namespace FLASH_NAMESPACE

