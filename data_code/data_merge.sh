#!/bin/bash

# set -e: 如果任何命令失败，脚本将立即退出。
# set -x: 在执行之前打印每个命令，方便调试。
set -e
set -x

# 定义Python脚本的名称，方便修改
PYTHON_SCRIPT="code/data_merge.py"

# --- 任务1: 生成 0.5B token, 65k 长度, 含倒序 ---
echo "=================================================="
echo "开始任务 1: mixed_long_context (0.5B tokens, 65k len)"
echo "=================================================="
python $PYTHON_SCRIPT \
    --input_file1 "data/process_arxiv/random_64k.jsonl" \
    --input_file2 "data/process_code/random_64k.jsonl" \
    --output_file "data/train/random_64k.jsonl" \
