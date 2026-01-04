#!/bin/bash
# Build script for Flash Attention test (SM89, hdim64, bf16, causal)

set -e
export PATH=/usr/local/cuda-12.8/bin:$PATH
# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
FLASH_ATTN_DIR="$SCRIPT_DIR/.."
CUTLASS_INCLUDE="$FLASH_ATTN_DIR/cutlass/include"
FLASH_SRC="$FLASH_ATTN_DIR/flash_attn/src"

echo "=== Flash Attention Build Script ==="
echo "Script dir: $SCRIPT_DIR"
echo "Cutlass include: $CUTLASS_INCLUDE"
echo "Flash src: $FLASH_SRC"

# Get PyTorch include and library paths
TORCH_INCLUDE=$(python3 -c "import torch; from torch.utils.cpp_extension import include_paths; print(' '.join(['-I' + p for p in include_paths()]))")
TORCH_LIB=$(python3 -c "import torch; from torch.utils.cpp_extension import library_paths; print(' '.join(['-L' + p for p in library_paths()]))")
TORCH_CXXFLAGS=$(python3 -c "import torch; print(' '.join(torch._C._PYBIND11_BUILD_ABI + ['']))" 2>/dev/null || echo "")

echo "PyTorch includes: $TORCH_INCLUDE"
echo "PyTorch libs: $TORCH_LIB"

# Check CUDA compute capability
echo ""
echo "Compiling for SM89 (Ada Lovelace)..."

# Compile the CUDA kernel
echo "Compiling flash_fwd_hdim64_bf16_causal_sm89.cu..."
nvcc -c \
    -arch=sm_89 \
    -std=c++17 \
    -O3 \
    --expt-relaxed-constexpr \
    --expt-extended-lambda \
    -Xcompiler -fPIC \
    -DNDEBUG \
    -I"$CUTLASS_INCLUDE" \
    -I"$FLASH_SRC" \
    $TORCH_INCLUDE \
    "$SCRIPT_DIR/flash_fwd_hdim64_bf16_causal_sm89.cu" \
    -o "$SCRIPT_DIR/flash_fwd_hdim64_bf16_causal_sm89.o"

# Compile the test program
echo "Compiling test_fa.cpp..."
nvcc -c \
    -arch=sm_89 \
    -std=c++17 \
    -O3 \
    --expt-relaxed-constexpr \
    --expt-extended-lambda \
    -Xcompiler -fPIC \
    -DNDEBUG \
    -I"$CUTLASS_INCLUDE" \
    -I"$FLASH_SRC" \
    $TORCH_INCLUDE \
    "$SCRIPT_DIR/test_fa.cpp" \
    -o "$SCRIPT_DIR/test_fa.o"

# Link
echo "Linking..."
nvcc \
    -arch=sm_89 \
    -std=c++17 \
    $TORCH_LIB \
    "$SCRIPT_DIR/flash_fwd_hdim64_bf16_causal_sm89.o" \
    "$SCRIPT_DIR/test_fa.o" \
    -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda \
    -o "$SCRIPT_DIR/test_fa"

echo ""
echo "=== Build successful! ==="
echo "Run with: $SCRIPT_DIR/test_fa"

