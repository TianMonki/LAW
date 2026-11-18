#!/bin/bash
# LAW Stage I: score intact 8K source documents with Q_s = T * U (k_s = 2048) and keep the top p_s.
# Usage: bash score/scripts/stage1_source.bash arxiv|code
set -e

DOMAIN=${1:?"Usage: $0 arxiv|code"}

file_path="data/law_${DOMAIN}/source_docs.jsonl"            # Source pool D
output_file="data/law_${DOMAIN}/stage1_scores.jsonl"        # {"data_id", "T", "U", "Q"}
output_path="data/law_${DOMAIN}/stage1_selected.jsonl"      # D* = Top_{p_s}(D; Q_s)
batch_size=48
margin=4                                                    # k_s = 8192 // 4
max_length=8192
p_s=0.5
model_path="llm/Llama-3.1-70B"

accelerate launch score/src/1_inference_dp.py "$file_path" "$output_file" "$batch_size" "$margin" \
    --max_length "$max_length" \
    --model_path "$model_path"

python score/src/2_filtering_by_Lds.py \
    --inference_path "$output_file" \
    --file_path "$file_path" \
    --output_path "$output_path" \
    --fraction "$p_s"

echo "Stage I for ${DOMAIN} complete. Results saved to $output_path."
