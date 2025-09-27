#!/bin/bash

# ---------------------------------------------------------------------------
# 步骤 1: 定义文件路径
# ---------------------------------------------------------------------------

# 原始的、未经筛选的数据文件
ORIGINAL_DATA_FILE="data/process_arxiv/sample_output.jsonl"

# 假设您已经生成了三个不同的 inference 结果文件
# 在实际使用中，您需要有生成这些文件的步骤
INFERENCE_PATH_1="data/process_arxiv/output_inference.jsonl"
INFERENCE_PATH_2="data/process_arxiv/output_inference_2.jsonl"
INFERENCE_PATH_3="data/process_arxiv/output_inference_8.jsonl"

# 最终筛选后数据的输出路径
FINAL_OUTPUT_PATH="data/process_arxiv/final_aggregated_output.jsonl"

# 定义错误日志文件
ERROR_LOG="error.log"

# 清空已有的错误日志文件
> "$ERROR_LOG"

# ---------------------------------------------------------------------------
# 步骤 2: 运行秩聚合筛选脚本
# ---------------------------------------------------------------------------
echo "Running Rank Aggregation and Filtering process..."

# 调用 Python 脚本，并传递多个 inference 路径
# 注意: --inference_path 已被修改为 --inference_paths
# 并且我们现在传递了三个文件路径给它
if ! python LongAttn/src/2_filtering_by_rank.py \
    --inference_paths $INFERENCE_PATH_1 $INFERENCE_PATH_2 $INFERENCE_PATH_3 \
    --output_path $FINAL_OUTPUT_PATH \
    --file_path $ORIGINAL_DATA_FILE 2>>"$ERROR_LOG"; then
    echo "Rank Aggregation script failed. Check $ERROR_LOG for details."
    exit 1
fi

echo "Processing complete. Aggregated and filtered results saved to $FINAL_OUTPUT_PATH."