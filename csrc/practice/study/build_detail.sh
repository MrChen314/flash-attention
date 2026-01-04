#!/bin/bash
# Build script for Flash Attention Detailed Study Version
# (SM89, hdim64, bf16, causal)
#
# 这个脚本编译展开学习版本的 Flash Attention
# 与原版 build.sh 的区别:
#   - 编译 flash_fwd_hdim64_bf16_causal_sm89_detail.cu (详细展开版)
#   - 复用 practice 目录的 test_fa.cpp 作为测试程序

set -e
export PATH=/usr/local/cuda-12.8/bin:$PATH

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PRACTICE_DIR="$SCRIPT_DIR/.."
FLASH_ATTN_DIR="$SCRIPT_DIR/../.."
CUTLASS_INCLUDE="$FLASH_ATTN_DIR/cutlass/include"
FLASH_SRC="$FLASH_ATTN_DIR/flash_attn/src"

echo "=== Flash Attention Detailed Study Version Build Script ==="
echo "Script dir: $SCRIPT_DIR"
echo "Practice dir: $PRACTICE_DIR"
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
echo "This is the DETAILED STUDY version with comprehensive comments."

# Compile the CUDA kernel (详细展开版本)
echo ""
echo "Step 1: Compiling flash_fwd_hdim64_bf16_causal_sm89_detail.cu..."
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
    "$SCRIPT_DIR/flash_fwd_hdim64_bf16_causal_sm89_detail.cu" \
    -o "$SCRIPT_DIR/flash_fwd_hdim64_bf16_causal_sm89_detail.o"
echo "  -> Generated: flash_fwd_hdim64_bf16_causal_sm89_detail.o"

# Compile the test program (复用 practice 目录的 test_fa.cpp)
echo ""
echo "Step 2: Compiling test_fa.cpp (from practice directory)..."
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
    "$PRACTICE_DIR/test_fa.cpp" \
    -o "$SCRIPT_DIR/test_fa_detail.o"
echo "  -> Generated: test_fa_detail.o"

# Link
echo ""
echo "Step 3: Linking..."
nvcc \
    -arch=sm_89 \
    -std=c++17 \
    $TORCH_LIB \
    "$SCRIPT_DIR/flash_fwd_hdim64_bf16_causal_sm89_detail.o" \
    "$SCRIPT_DIR/test_fa_detail.o" \
    -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda \
    -o "$SCRIPT_DIR/test_fa_detail"
echo "  -> Generated: test_fa_detail"

echo ""
echo "=== Build successful! ==="
echo ""
echo "To run the test:"
echo "  cd $SCRIPT_DIR"
echo "  ./run_detail.sh"
echo ""
echo "Or directly:"
echo "  export LD_LIBRARY_PATH=/root/miniconda3/lib/python3.12/site-packages/torch/lib:\$LD_LIBRARY_PATH"
echo "  $SCRIPT_DIR/test_fa_detail"

