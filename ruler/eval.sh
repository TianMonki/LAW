
methods=("llama3-8b-law-ntk-64k-seed42" "llama3-8b-law-ntk-64k-seed43" "llama3-8b-law-ntk-64k-seed44")

# Full 13-task synthetic suite at 4K-64K; run after `bash run.sh <method> synthetic`.
# Each pred/summary.csv holds per-task scores; the paper averages the 13 tasks per length,
# then the five lengths, then the three seed runs.
for method in "${methods[@]}"; 
do
for length in 4096 8192 16384 32768 65536;
do
python eval/evaluate.py \
  --data_dir ruler/results/${method}/synthetic/${length}/pred \
  --benchmark synthetic
done
done
