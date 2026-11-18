# LAW: A Two-Stage Framework for Synthesizing Long-Context Training Data

This repository contains the main-experiment code of Long-Attention Weaving (LAW). LAW selects 8K source documents before synthesis, weaves them into 64K sequences, and selects the synthesized sequences afterward.

## Repository structure

```text
.
├── score/                  # Q = TU attention scoring and Stage I / Stage II selection
├── data_code/              # Order-preserving weaving and 1:1 arXiv/code mixing
├── continuous_finetuning/  # LLaMA-3 8B continual pre-training (NTK, 64K)
├── models/llama_ntk_64k/   # NTK model used for evaluation
├── longbench/              # LongBench evaluation
├── ruler/                  # RULER evaluation (4K-64K)
└── run_law_main.sh         # End-to-end data construction and training
```

Generated data are written to `data/` and checkpoints to `ckpts/`.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install transformers==4.43.3 torch accelerate deepspeed datasets flash-attn jsonlines
```

Place the models under `llm/` (`llm/Meta-Llama-3-8B` and `llm/Llama-3.1-70B`).

## Data

Raw corpora go in `data/arxiv/` and `data/code/`, one `{"text": "..."}` per line.

Experiment data

```text
data/
└── train/law_main_seed42.jsonl  https://drive.google.com/file/d/1i_mGJcNRj0kCr5vrU9M7RpG9QjJXIYZg/view?usp=drive_link
```

This is the 1B-token training set of the main experiment. Each line holds one 64K-token sequence in `input_ids` (LLaMA-3 tokenizer), with arXiv and code mixed 1:1.

## Environment

Computational platform: 8 NVIDIA A800 GPUs, Transformers 4.43.3, DeepSpeed ZeRO-3.

## Running the model

Build the data and train from the repository root (seeds 42, 43, 44):

```bash
bash run_law_main.sh
```

To train directly on the released data:

```bash
bash continuous_finetuning/finetune_ntk_main.sh 42
```

Evaluate on LongBench (put the LongBench v1 `*.jsonl` files in `longbench/data/`) and RULER:

```bash
cd longbench && bash scripts/eval_law.sh
cd ruler && for s in 42 43 44; do bash run.sh llama3-8b-law-ntk-64k-seed$s synthetic; done && bash eval.sh
```

Both selection stages use a frozen LLaMA-3.1 70B (first decoder layer) and retain 50% of the candidates. Reported numbers are means over three seeds.
