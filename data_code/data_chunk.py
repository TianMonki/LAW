import os
import json
import glob
import multiprocessing
from transformers import AutoTokenizer

def worker_task(worker_id, file_list, tokenizer_name, chunk_size, temp_output_template):
    """
    单个工作进程执行的任务。
    它将所有生成的chunks作为一个JSON列表写入单个临时文件。
    """
    print(f"[进程 {worker_id}] 已启动，将处理 {len(file_list)} 个文件...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as e:
        print(f"[进程 {worker_id}] 错误：加载分词器失败: {e}")
        return

    all_text = ""
    for filepath in file_list:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if "text" in data and data["text"]:
                            all_text += data["text"]
                    except json.JSONDecodeError:
                        pass
        except FileNotFoundError:
            print(f"[进程 {worker_id}] 警告：文件未找到 {filepath}")

    if not all_text:
        print(f"[进程 {worker_id}] 警告：未能从分配的文件中提取到任何文本内容。")
        return

    tokens = tokenizer.encode(all_text, add_special_tokens=False)
    
    if len(tokens) < chunk_size:
        print(f"[进程 {worker_id}] 警告：Token总数 ({len(tokens)}) 小于一个chunk的大小 ({chunk_size})。")
        return

    full_chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens) - (len(tokens) % chunk_size), chunk_size)]
    
    # --- 核心修改 1: 将所有chunks作为一个列表整体写入JSON文件 ---
    temp_output_file = temp_output_template.format(worker_id)
    with open(temp_output_file, 'w', encoding='utf-8') as f:
        json.dump(full_chunks, f)
            
    print(f"[进程 {worker_id}] 任务完成，成功生成了 {len(full_chunks)} 个chunks。")

def merge_temp_files(final_output_file, temp_file_list):
    """
    合并所有临时文件到一个最终文件中。
    现在它会读取每个文件中的JSON列表，然后逐个chunk写入新行。
    """
    print("\n所有进程已完成，开始合并临时文件...")
    print(f"预计将合并 {len(temp_file_list)} 个文件。")
    total_chunks = 0
    with open(final_output_file, 'w', encoding='utf-8') as f_out:
        for temp_file in temp_file_list:
            if not os.path.exists(temp_file):
                print(f"  警告：预期的临时文件 {temp_file} 未找到，跳过。")
                continue
            
            lines_in_this_file = 0
            try:
                with open(temp_file, 'r', encoding='utf-8') as f_in:
                    # --- 核心修改 2: 读取整个JSON文件并解析 ---
                    list_of_chunks = json.load(f_in)
                    
                    # --- 核心修改 3: 逐个chunk写入，确保每行一个 ---
                    for chunk in list_of_chunks:
                        f_out.write(json.dumps(chunk) + '\n')
                    
                    lines_in_this_file = len(list_of_chunks)
            except (json.JSONDecodeError, TypeError):
                 print(f"  警告：无法解析或处理文件 {temp_file}，跳过。")
                 continue


            print(f"  - 已合并文件 {temp_file}，包含 {lines_in_this_file} 个chunks。")
            total_chunks += lines_in_this_file
            os.remove(temp_file)
            
    print(f"合并完成！总共保存了 {total_chunks} 个token序列到 {final_output_file}")
    
    if total_chunks == 0:
        print("\n*** 最终警告：所有进程均未成功生成任何数据，因此输出文件为空。请检查上面各个进程的日志，确认问题所在。 ***")
    
    print("临时文件已清理。")

if __name__ == "__main__":
    # --- 确认设置 ---
    data_directory = "data/code/"
    output_file = "data/code/processed_tokens.jsonl" 
    tokenizer_name = "llm/Llama-2-7b-hf"
    chunk_size = 8192
    
    all_files = [os.path.join(data_directory, f) for f in sorted(os.listdir(data_directory)) if f.startswith("part-")]
    if not all_files:
        print(f"错误：在目录 '{data_directory}' 中没有找到任何 'part-' 开头的文件。")
    else:
        num_processes = multiprocessing.cpu_count()
        print(f"找到 {len(all_files)} 个文件，将使用 {num_processes} 个进程进行处理。")
        
        files_per_process = (len(all_files) + num_processes - 1) // num_processes
        file_chunks = [all_files[i:i + files_per_process] for i in range(0, len(all_files), files_per_process)]

        processes = []
        temp_file_template = "processed_tokens_temp_{}.jsonl"
        expected_temp_files = [temp_file_template.format(i) for i in range(len(file_chunks))]

        for i, chunk in enumerate(file_chunks):
            p = multiprocessing.Process(target=worker_task, args=(i, chunk, tokenizer_name, chunk_size, temp_file_template))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        merge_temp_files(output_file, expected_temp_files)