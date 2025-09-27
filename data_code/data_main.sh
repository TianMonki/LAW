#!/bin/bash

# set -e: 如果任何命令失败，脚本将立即退出。
# set -x: 在执行之前打印每个命令，方便调试。
set -e
set -x

# 定义Python脚本的名称，方便修改
PYTHON_SCRIPT="code/data_main.py"

# --- 任务1: 生成 0.5B token, 65k 长度, 含倒序 ---
echo "=================================================="
echo "开始任务 1: mixed_long_context (0.5B tokens, 65k len, with reverse)"
echo "=================================================="
python $PYTHON_SCRIPT \
    --input_file "data/process_code/final_aggregated_output.jsonl" \
    --output_file "data/process_code/mixed_long_context.jsonl" \
    --tokenizer_name "llm/Llama-2-7b-hf" \
    --token_limit_per_type 250000000 \
    --sequence_length 65536 \
    --use_reverse


# --- 任务3: 1B token, 65k 长度, 不含倒序 ---
echo "=================================================="
echo "开始任务 3: data_wo_reverse (1B tokens, 65k len, without reverse)"
echo "=================================================="
python $PYTHON_SCRIPT \
    --input_file "data/process_code/final_aggregated_output.jsonl" \
    --output_file "data/process_code/data_wo_reverse.jsonl" \
    --tokenizer_name "llm/Llama-2-7b-hf" \
    --token_limit_per_type 500000000 \
    --sequence_length 65536

# --- 任务4: 2B token, 65k 长度, 含倒序 ---
echo "=================================================="
echo "开始任务 4: mixed_long_context_2B (2B tokens, 65k len, with reverse)"
echo "=================================================="
python $PYTHON_SCRIPT \
    --input_file "data/process_code/final_aggregated_output.jsonl" \
    --output_file "data/process_code/mixed_long_context_2B.jsonl" \
    --tokenizer_name "llm/Llama-2-7b-hf" \
    --token_limit_per_type 500000000 \
    --sequence_length 65536 \
    --use_reverse

# --- 任务5: 1B token, 32k 长度, 含倒序 ---
echo "=================================================="
echo "开始任务 5: mixed_long_context_32k (1B tokens, 32k len, with reverse)"
echo "=================================================="
python $PYTHON_SCRIPT \
    --input_file "data/process_code/final_aggregated_output.jsonl" \
    --output_file "data/process_code/mixed_long_context_32k.jsonl" \
    --tokenizer_name "llm/Llama-2-7b-hf" \
    --token_limit_per_type 250000000 \
    --sequence_length 32768 \
    --use_reverse

echo "=================================================="
echo "所有任务已成功完成！"
echo "=================================================="