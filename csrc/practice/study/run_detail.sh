#!/bin/bash
# Run script for Flash Attention Detailed Study Version
#
# 这个脚本运行展开学习版本的 Flash Attention 测试
# 需要先运行 build_detail.sh 编译

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set PyTorch library path (根据你的 Python 环境调整)
export LD_LIBRARY_PATH=/root/miniconda3/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH

# Check if executable exists
if [ ! -f "$SCRIPT_DIR/test_fa_detail" ]; then
    echo "Error: test_fa_detail not found!"
    echo "Please run build_detail.sh first:"
    echo "  cd $SCRIPT_DIR && ./build_detail.sh"
    exit 1
fi

# Run the test
echo "=== Running Flash Attention Detailed Study Version ==="
echo ""
"$SCRIPT_DIR/test_fa_detail"

