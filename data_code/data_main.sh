#!/bin/bash

# set -e: 如果任何命令失败，脚本将立即退出。
# set -x: 在执行之前打印每个命令，方便调试。
set -e
set -x

# LAW 主实验：对每个领域，用 Stage I 选出的 8K 文档做 order-preserving weaving，生成 64K 候选池 W。
# 每领域 Stage II 后保留 0.5B token（arXiv:code = 1:1，共 1B）。
# Usage: bash data_code/data_main.sh [SEED]
SEED=${1:-42}
PYTHON_SCRIPT="data_code/data_main.py"

for DOMAIN in arxiv code; do
    echo "=================================================="
    echo "构建候选池: ${DOMAIN}, seed=${SEED}"
    echo "=================================================="
    python $PYTHON_SCRIPT \
        --input_file "data/law_${DOMAIN}/stage1_selected.jsonl" \
        --output_file "data/law_${DOMAIN}/candidates_seed${SEED}.jsonl" \
        --domain ${DOMAIN} \
        --tokenizer_name "llm/Meta-Llama-3-8B" \
        --doc_length 8192 \
        --sequence_length 65536 \
        --token_budget 500000000 \
        --p_long 0.5 \
        --seed ${SEED}
done

echo "=================================================="
echo "所有任务已成功完成！"
echo "=================================================="
