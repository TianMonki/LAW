import json
import random
import argparse

def merge_and_shuffle_jsonl(file1_path, file2_path, output_path):
    """
    Loads two JSONL files, takes a random half of the data from each,
    merges them, shuffles the result, and writes to a new file.

    Args:
        file1_path (str): The path to the first simple_sequences.jsonl file.
        file2_path (str): The path to the second simple_sequences.jsonl file.
        output_path (str): The path to the output merged file.
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

    print("Selecting a random half from each file...")
    random.shuffle(data1)
    random.shuffle(data2)

    # half_data1 = data1[:len(data1) // 2]
    # half_data2 = data2[:len(data2) // 2]

    print("Merging and shuffling the selected data...")
    merged_data = data1 + data2
    random.shuffle(merged_data)
    print(f"Writing the merged and shuffled data to {output_path}...")
    try:
        with open(output_path, 'w', encoding='utf-8') as f_out:
            # Using writelines for efficiency
            f_out.writelines(merged_data)
    except IOError as e:
        print(f"Error writing to file: {e}")
        return

    print("--- Merging and shuffling process completed successfully! ---")

if __name__ == "__main__":
    # --- Configuration ---
    # Please replace these with the actual paths to your files
    parser = argparse.ArgumentParser(description="拼接数据。")
    parser.add_argument("--input_file1", type=str, required=True, help="输入的JSONL文件路径。")
    parser.add_argument("--input_file2", type=str, required=True, help="输入的JSONL文件路径。")
    parser.add_argument("--output_file", type=str, required=True, help="输出的JSONL文件路径。")

    args = parser.parse_args()
    # It's good practice to confirm the files exist before running
    import os
    if not os.path.exists(args.input_file1) or not os.path.exists(args.input_file2):
        print("PLEASE UPDATE the 'input_file1' and 'input_file2' variables")

    else:
        merge_and_shuffle_jsonl(args.input_file1, args.input_file2, args.output_file)