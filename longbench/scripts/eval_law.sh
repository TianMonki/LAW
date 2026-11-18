### LAW main experiment on LongBench v1: 16 English tasks, greedy decoding, middle truncation.
### Run from longbench/: bash scripts/eval_law.sh
datasets=("narrativeqa" "qasper" "multifieldqa_en" "hotpotqa" "2wikimqa" "musique" \
          "gov_report" "qmsum" "multi_news" "trec" "triviaqa" "samsum" \
          "passage_count" "passage_retrieval_en" "lcc" "repobench-p")

### three runs with different seeds
models=(
    "llama3-8b-law-ntk-64k-seed42" \
    "llama3-8b-law-ntk-64k-seed43" \
    "llama3-8b-law-ntk-64k-seed44"
    )

for MODEL_NAME in "${models[@]}";
do
for dataset in "${datasets[@]}";
do
echo "$MODEL_NAME $dataset"
torchrun --nproc_per_node=8 pred.py \
--model ${MODEL_NAME} \
--dataset_name ${dataset}
done
python eval_64k.py --model ${MODEL_NAME}
done

python summarize_law.py --models "${models[@]}"
