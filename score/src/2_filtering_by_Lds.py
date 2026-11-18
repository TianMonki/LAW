import argparse
import jsonlines

# Top-p selection shared by both LAW stages (Eq. 1):
#   Stage I : D* = Top_{p_s}(D; Q_s)      Stage II: W* = Top_{p_l}(W; Q_l)
# Each stage ranks candidates within its own pool (one domain at a time).

def select_ids(inference_path, fraction):
    with jsonlines.open(inference_path) as reader:
        scores = [(obj['data_id'], obj['Q']) for obj in reader]

    keep = int(fraction * len(scores))
    selected = sorted(scores, key=lambda item: item[1], reverse=True)[:keep]
    return {data_id for data_id, _ in selected}, len(scores)

def main():
    parser = argparse.ArgumentParser(description="Select the top fraction of records by LAW score Q")
    parser.add_argument('--inference_path', type=str, required=True, help='Scores written by 1_inference_dp.py')
    parser.add_argument('--file_path', type=str, required=True, help='Records to filter (must carry data_id)')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the selected records')
    parser.add_argument('--fraction', type=float, default=0.5, help='Retained fraction p_s or p_l')

    args = parser.parse_args()

    selected, total = select_ids(args.inference_path, args.fraction)

    written = 0
    with jsonlines.open(args.file_path) as reader, jsonlines.open(args.output_path, 'w') as writer:
        for data in reader:
            if data['data_id'] in selected:
                writer.write(data)
                written += 1

    print(f"Scored records: {total}, selected: {len(selected)}, written: {written}")
    if written != len(selected):
        print("[Warning] Some selected data_id values were not found in --file_path.")

if __name__ == '__main__':
    main()
