# LAW: Long-Attention Weaving

Code for the main experiment of LAW, a two-stage framework that synthesizes long-context training data from short documents:

1. **Stage I**: score 8K source documents with $Q = TU$ (remote attention strength x normalized global entropy) and keep the top 50%.
2. **Synthesis**: build 64K candidates by order-preserving weaving of 8 selected documents.
3. **Stage II**: rescore every complete 64K candidate with $Q$ and keep the top 50%.

Scores use the first decoder layer of a frozen LLaMA-3.1 70B. arXiv and code are processed separately and then mixed 1:1 by tokens (1B tokens in total) to continually pre-train LLaMA-3 8B with NTK scaling at 64K context.

## Layout

```
score/src/              LongAttn.py (Q = TU scoring), 0_pre-process.py, 1_inference_dp.py, 2_filtering_by_Lds.py
score/scripts/          preprocess.sh, stage1_source.bash, stage2_sequence.bash
data_code/              data_main.py (weaving), data_merge.py (1:1 domain mix)
continuous_finetuning/  fine-tune.py, finetune_ntk_main.sh
models/llama_ntk_64k/   NTK model used for evaluation
longbench/              LongBench prediction, scoring, and summary (eval_law.sh)
ruler/                  RULER, 13 synthetic tasks at 4K-64K
run_law_main.sh         end-to-end data construction and training
```

## Requirements

`transformers==4.43.3`, `torch`, `accelerate`, `deepspeed`, `datasets`, `flash-attn`, and `jsonlines`. Put the models under `llm/` (`llm/Meta-Llama-3-8B`, `llm/Llama-3.1-70B`).

## Data

Put the raw files in `data/arxiv/` and `data/code/`. Each line is `{"text": "..."}`.

## Run

From the repository root:

```bash
bash run_law_main.sh
```

This runs preprocessing and Stage I once per domain. For each seed (42, 43, 44) it then runs weaving, Stage II, the 1:1 mix, and training. Checkpoints are written to `ckpts/law_main/`.

Evaluation (put the LongBench v1 `*.jsonl` files in `longbench/data/`, and download the RULER data with the scripts in `ruler/data/synthetic/json/`):

```bash
cd longbench && bash scripts/eval_law.sh
cd ruler && for s in 42 43 44; do bash run.sh llama3-8b-law-ntk-64k-seed$s synthetic; done && bash eval.sh
```

Reported numbers are means over the three seeds.
