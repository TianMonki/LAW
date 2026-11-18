import json
import argparse

# LongBench table of the paper: category scores average their tasks, Avg. is the unweighted
# mean over all 16 tasks, and every number is the mean over the seed runs.
CATEGORIES = {
    "Single-Doc QA": ["narrativeqa", "qasper", "multifieldqa_en"],
    "Multi-Doc QA": ["hotpotqa", "2wikimqa", "musique"],
    "Summ.": ["gov_report", "qmsum", "multi_news"],
    "Few-shot": ["trec", "triviaqa", "samsum"],
    "Synthetic": ["passage_count", "passage_retrieval_en"],
    "Code": ["lcc", "repobench-p"],
}

def summarize(result):
    row = {name: sum(result[t] for t in tasks) / len(tasks) for name, tasks in CATEGORIES.items()}
    all_tasks = [t for tasks in CATEGORIES.values() for t in tasks]
    row["Avg."] = sum(result[t] for t in all_tasks) / len(all_tasks)
    return row

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=str, nargs='+', required=True, help='One model name per seed run')
    args = parser.parse_args()

    rows = []
    for model in args.models:
        with open(f"pred/{model}/result.json", "r", encoding="utf-8") as f:
            rows.append(summarize(json.load(f)))
        print(model, {k: round(v, 2) for k, v in rows[-1].items()})

    mean = {k: round(sum(r[k] for r in rows) / len(rows), 2) for k in rows[0]}
    print(f"Mean over {len(rows)} runs:", mean)
