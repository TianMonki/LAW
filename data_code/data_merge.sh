#!/bin/bash

# set -e: 如果任何命令失败，脚本将立即退出。
# set -x: 在执行之前打印每个命令，方便调试。
set -e
set -x

# LAW 主实验：把两个领域的 Stage II 结果按 1:1 token 混合为 1B token 训练集。
# Usage: bash data_code/data_merge.sh [SEED]
SEED=${1:-42}
PYTHON_SCRIPT="data_code/data_merge.py"

mkdir -p data/train
python $PYTHON_SCRIPT \
    --input_file1 "data/law_arxiv/stage2_selected_seed${SEED}.jsonl" \
    --input_file2 "data/law_code/stage2_selected_seed${SEED}.jsonl" \
    --output_file "data/train/law_main_seed${SEED}.jsonl" \
    --seed ${SEED}

echo "=================================================="
echo "所有任务已成功完成！"
echo "=================================================="
