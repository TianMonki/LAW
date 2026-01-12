#!/bin/bash

# set -e: 如果任何命令失败，脚本将立即退出。
# set -x: 在执行之前打印每个命令，方便调试。
set -e
set -x

# 定义Python脚本的名称，方便修改
PYTHON_SCRIPT="code/data_main.py"

# --- 任务1 ---
echo "=================================================="
echo "开始任务 1"
echo "=================================================="
python $PYTHON_SCRIPT \
    --input_file "data/process_code/final_aggregated_output.jsonl" \
    --output_file "data/process_code/mixed_long_context.jsonl" \
    --tokenizer_name "llm/Llama-2-7b-hf" \
    --token_limit_per_type 250000000 \
    --sequence_length 65536 \
    --use_reverse
