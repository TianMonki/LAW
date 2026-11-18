#!/bin/bash
# LAW main experiment end to end (run from this directory).
#   Stage I (per domain, once) -> weaving + Stage II + 1:1 mix + training (per seed)
# Evaluation: longbench/scripts/eval_law.sh and LCEG-main/ruler (run.sh + eval.sh).
set -e

SEEDS=(42 43 44)

for DOMAIN in arxiv code; do
    bash score/scripts/preprocess.sh ${DOMAIN}
    bash score/scripts/stage1_source.bash ${DOMAIN}
done

for SEED in "${SEEDS[@]}"; do
    bash data_code/data_main.sh ${SEED}
    for DOMAIN in arxiv code; do
        bash score/scripts/stage2_sequence.bash ${DOMAIN} ${SEED}
    done
    bash data_code/data_merge.sh ${SEED}
    bash continuous_finetuning/finetune_ntk_main.sh ${SEED}
done
