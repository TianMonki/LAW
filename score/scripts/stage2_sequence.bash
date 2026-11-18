#!/bin/bash
# LAW Stage II: recompute Q_l = T * U (k_l = 16384) over each complete 64K woven sequence
# and keep the top p_l for training.
# Usage: bash score/scripts/stage2_sequence.bash arxiv|code [SEED]
set -e

DOMAIN=${1:?"Usage: $0 arxiv|code [SEED]"}
SEED=${2:-42}

file_path="data/law_${DOMAIN}/candidates_seed${SEED}.jsonl"         # Candidate pool W from data_main.py
output_file="data/law_${DOMAIN}/stage2_scores_seed${SEED}.jsonl"    # {"data_id", "T", "U", "Q"}
output_path="data/law_${DOMAIN}/stage2_selected_seed${SEED}.jsonl"  # W* = Top_{p_l}(W; Q_l)
batch_size=1
margin=4                                                            # k_l = 65536 // 4
max_length=65536
p_l=0.5
model_path="llm/Llama-3.1-70B"

accelerate launch score/src/1_inference_dp.py "$file_path" "$output_file" "$batch_size" "$margin" \
    --max_length "$max_length" \
    --model_path "$model_path"

python score/src/2_filtering_by_Lds.py \
    --inference_path "$output_file" \
    --file_path "$file_path" \
    --output_path "$output_path" \
    --fraction "$p_l"

echo "Stage II for ${DOMAIN} (seed ${SEED}) complete. Results saved to $output_path."
