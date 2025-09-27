#!/bin/bash

# set -e: 如果任何命令失败，脚本将立即退出。
# set -x: 在执行之前打印每个命令，方便调试。
set -e
set -x

# 定义Python脚本的名称，方便修改
PYTHON_SCRIPT="code/data_merge.py"

# --- 任务1: 生成 0.5B token, 65k 长度, 含倒序 ---
echo "=================================================="
echo "开始任务 1: mixed_long_context (0.5B tokens, 65k len, with reverse)"
echo "=================================================="
python $PYTHON_SCRIPT \
    --input_file1 "data/process_arxiv/mixed_long_context.jsonl" \
    --input_file2 "data/process_code/mixed_long_context.jsonl" \
    --output_file "data/train/mixed_long_context.jsonl" \


# # --- 任务2: 处理 sample_output.jsonl, 1B token, 65k 长度, 含倒序 ---
# echo "=================================================="
# echo "开始任务 2: data_wo_filter (1B tokens, 65k len, with reverse, from sample)"
# echo "=================================================="
# python $PYTHON_SCRIPT \
#     --input_file1 "data/process_code/data_wo_filter.jsonl" \
#     --input_file2 "data/process_arxiv/data_wo_filter.jsonl" \
#     --output_file "data/train/data_wo_filter.jsonl" \
    
# --- 任务3: 1B token, 65k 长度, 不含倒序 ---
echo "=================================================="
echo "开始任务 3: data_wo_reverse (1B tokens, 65k len, without reverse)"
echo "=================================================="
python $PYTHON_SCRIPT \
    --input_file1 "data/process_code/data_wo_reverse.jsonl" \
    --input_file2 "data/process_arxiv/data_wo_reverse.jsonl" \
    --output_file "data/train/data_wo_reverse.jsonl" \

# --- 任务4: 2B token, 65k 长度, 含倒序 ---
echo "=================================================="
echo "开始任务 4: mixed_long_context_2B (2B tokens, 65k len, with reverse)"
echo "=================================================="
python $PYTHON_SCRIPT \
    --input_file1 "data/process_code/mixed_long_context_2B.jsonl" \
    --input_file2 "data/process_arxiv/mixed_long_context_2B.jsonl" \
    --output_file "data/train/mixed_long_context_2B.jsonl" \


# --- 任务5: 1B token, 32k 长度, 含倒序 ---
echo "=================================================="
echo "开始任务 5: mixed_long_context_32k (1B tokens, 32k len, with reverse)"
echo "=================================================="
python $PYTHON_SCRIPT \
    --input_file1 "data/process_code/mixed_long_context_32k.jsonl" \
    --input_file2 "data/process_arxiv/mixed_long_context_32k.jsonl" \
    --output_file "data/train/mixed_long_context_32k.jsonl" \


echo "=================================================="
echo "所有任务已成功完成！"
echo "=================================================="