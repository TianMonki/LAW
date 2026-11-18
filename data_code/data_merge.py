import json
import random
import argparse

def merge_and_shuffle_jsonl(file1_path, file2_path, output_path, seed=42):
    """
    Mixes the Stage II selections of two domains 1:1 by training tokens and shuffles the result.
    Every sequence has the same length L, so equal token counts mean equal sequence counts.

    Args:
        file1_path (str): Stage II selected sequences of the first domain.
        file2_path (str): Stage II selected sequences of the second domain.
        output_path (str): The path to the output merged file.
        seed (int): Controls the subsampling and the training-data order.
    """
    print(f"Loading data from {file1_path} and {file2_path}...")

    try:
        with open(file1_path, 'r', encoding='utf-8') as f1:
            data1 = f1.readlines()

        with open(file2_path, 'r', encoding='utf-8') as f2:
            data2 = f2.readlines()
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure both input files exist.")
        return

    rng = random.Random(seed)
    rng.shuffle(data1)
    rng.shuffle(data2)

    keep = min(len(data1), len(data2))
    if len(data1) != len(data2):
        print(f"Domain sizes differ ({len(data1)} vs {len(data2)}); keeping {keep} sequences from each for a 1:1 mix.")

    print("Merging and shuffling the selected data...")
    merged_data = data1[:keep] + data2[:keep]
    rng.shuffle(merged_data)
    print(f"Writing {len(merged_data)} sequences to {output_path}...")
    try:
        with open(output_path, 'w', encoding='utf-8') as f_out:
            # Using writelines for efficiency
            f_out.writelines(merged_data)
    except IOError as e:
        print(f"Error writing to file: {e}")
        return

    print("--- Merging and shuffling process completed successfully! ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="按 1:1 token 比例混合两个领域的数据。")
    parser.add_argument("--input_file1", type=str, required=True, help="输入的JSONL文件路径。")
    parser.add_argument("--input_file2", type=str, required=True, help="输入的JSONL文件路径。")
    parser.add_argument("--output_file", type=str, required=True, help="输出的JSONL文件路径。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子。")

    args = parser.parse_args()
    # It's good practice to confirm the files exist before running
    import os
    if not os.path.exists(args.input_file1) or not os.path.exists(args.input_file2):
        print("PLEASE UPDATE the 'input_file1' and 'input_file2' variables")

    else:
        merge_and_shuffle_jsonl(args.input_file1, args.input_file2, args.output_file, args.seed)
