import json
import random
import os
import glob
import multiprocessing as mp
from functools import partial
from transformers import AutoTokenizer
import argparse

# --------------------------
# 1. 并行分词模块
# --------------------------
def batch_tokenize(texts, tokenizer, add_special_tokens=False):
    """批量分词单批文本"""
    try:
        outputs = tokenizer(
            texts,
            add_special_tokens=add_special_tokens,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False
        )
        return outputs["input_ids"]
    except Exception as e:
        print(f"批量分词出错: {e}")
        return [[] for _ in range(len(texts))]

def process_chunk(chunk, tokenizer_name):
    """处理单个数据分片：提取content并批量分词"""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    texts = []
    for line in chunk:
        try:
            data = json.loads(line)
            texts.append(data.get("content", ""))
        except json.JSONDecodeError:
            texts.append("")
    
    batch_size = 256
    all_tokens = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_tokens = batch_tokenize(batch_texts, tokenizer)
        all_tokens.extend(batch_tokens)
    
    return all_tokens

def load_chunks_parallel(input_file, tokenizer_name="llm/Llama-2-7b-hf", num_workers=None):
    """并行加载并分词数据"""
    cpu_cores = mp.cpu_count()
    effective_workers = num_workers if num_workers is not None else cpu_cores
    print(f"开始并行分词，使用 {effective_workers} 个进程...")
    
    chunks = []
    chunk_size = 2000
    with open(input_file, 'r', encoding='utf-8') as f:
        current_chunk = []
        for line in f:
            current_chunk.append(line.strip())
            if len(current_chunk) >= chunk_size:
                chunks.append(current_chunk)
                current_chunk = []
        if current_chunk:
            chunks.append(current_chunk)
    
    print(f"文件分割完成，共 {len(chunks)} 个分片")
    
    if not chunks:
        return []

    num_processes = min(len(chunks), effective_workers)
    
    with mp.Pool(num_processes) as pool:
        func = partial(process_chunk, tokenizer_name=tokenizer_name)
        results = pool.map(func, chunks)
    
    all_chunks = []
    for res in results:
        all_chunks.extend(res)
    
    print(f"并行分词完成，共处理 {len(all_chunks)} 条数据")
    return all_chunks

# --------------------------
# 2. 数据拼接模块（worker进程）
# --------------------------
# 修改：参数列表增加了 num_direct_concat
def worker_process(worker_id, num_interlaced, num_reverse, num_random_split, num_direct_concat, all_chunks, all_indices, seq_len):
    """生成交错拼接、倒序拼接、随机切分拼接以及直接拼接的序列"""
    task_info = []
    if num_interlaced > 0: task_info.append(f"交错{num_interlaced}条")
    if num_reverse > 0: task_info.append(f"倒序{num_reverse}条")
    if num_random_split > 0: task_info.append(f"随机切分{num_random_split}条")
    if num_direct_concat > 0: task_info.append(f"直接拼接{num_direct_concat}条") # 新增日志
    
    if not task_info:
        print(f"工作进程 {worker_id} (PID: {os.getpid()}) 无任务，即将退出。")
        return

    print(f"工作进程 {worker_id} (PID: {os.getpid()}) 启动，任务：{'，'.join(task_info)}")
    
    chunk_size_4k = 4096
    temp_file = f"temp_sequences_{worker_id}.jsonl"

    if seq_len == 65536:
        num = 8
    elif seq_len == 32768:
        num = 4
    else:
        num = 4
    
    with open(temp_file, 'w', encoding='utf-8') as f_out:
        # A. 交错拼接
        for _ in range(num_interlaced):
            selected_indices = random.sample(all_indices, k=num)
            selected_chunks = [all_chunks[i] for i in selected_indices]
            
            parts_1 = [chunk[:chunk_size_4k] for chunk in selected_chunks]
            parts_2 = [chunk[chunk_size_4k:] for chunk in selected_chunks]
            
            inter_sequence = [t for part in parts_1 for t in part]
            inter_sequence.extend([t for part in parts_2 for t in part])
            
            f_out.write(json.dumps({
                "input_ids": inter_sequence,
                "chunk_indices": selected_indices,
                "type": "interlaced"
            }) + '\n')

        # B. 倒序拼接
        for _ in range(num_reverse):
            selected_indices = random.sample(all_indices, k=num)
            selected_chunks = [all_chunks[i] for i in selected_indices]
            
            parts_1 = [chunk[:chunk_size_4k] for chunk in selected_chunks]
            parts_2 = [chunk[chunk_size_4k:] for chunk in selected_chunks]
            reversed_parts_2 = list(reversed(parts_2))
            
            rev_sequence = [t for part in parts_1 for t in part]
            rev_sequence.extend([t for part in reversed_parts_2 for t in part])
            
            f_out.write(json.dumps({
                "input_ids": rev_sequence,
                "chunk_indices": selected_indices,
                "type": "reverse"
            }) + '\n')

        # C. 均分后随机拼接
        for _ in range(num_random_split):
            selected_indices = random.sample(all_indices, k=num)
            selected_chunks = [all_chunks[i] for i in selected_indices]
            
            segments = []
            for chunk in selected_chunks:
                mid_point = len(chunk) // 2
                segments.append(chunk[:mid_point])
                segments.append(chunk[mid_point:])
            
            random.shuffle(segments)
            rand_sequence = [t for seg in segments for t in seg]
            
            f_out.write(json.dumps({
                "input_ids": rand_sequence,
                "chunk_indices": selected_indices,
                "type": "random_split"
            }) + '\n')

        # D. 直接随机拼接 (新增)
        for _ in range(num_direct_concat):
            selected_indices = random.sample(all_indices, k=num)
            selected_chunks = [all_chunks[i] for i in selected_indices]
            
            # 直接将所有chunk按顺序连在一起，不做切分
            direct_sequence = [t for chunk in selected_chunks for t in chunk]
            
            f_out.write(json.dumps({
                "input_ids": direct_sequence,
                "chunk_indices": selected_indices,
                "type": "direct_concat"
            }) + '\n')

    print(f"工作进程 {worker_id} 完成")

# --------------------------
# 3. 结果合并与打乱模块
# --------------------------
def merge_and_shuffle(file_pattern, final_output_file, seed=42):
    """合并临时文件并打乱顺序"""
    print(f"合并临时文件并打乱顺序...")
    temp_files = glob.glob(file_pattern)
    if not temp_files:
        print("警告：未找到任何临时文件进行合并。")
        return
        
    all_lines = []
    for temp_file in temp_files:
        try:
            with open(temp_file, 'r', encoding='utf-8') as f_temp:
                all_lines.extend(f_temp.readlines())
            os.remove(temp_file)
        except Exception as e:
            print(f"处理临时文件 {temp_file} 出错: {e}")
    
    random.seed(seed)
    random.shuffle(all_lines)
    
    with open(final_output_file, 'w', encoding='utf-8') as f_final:
        f_final.writelines(all_lines)
    
    print(f"合并完成，生成 {final_output_file}，共 {len(all_lines)} 条数据")

# --------------------------
# 主函数
# --------------------------
def main():
    parser = argparse.ArgumentParser(description="为长序列LLM训练并行处理和拼接数据。")
    parser.add_argument("--input_file", type=str, required=True, help="输入的JSONL文件路径。")
    parser.add_argument("--output_file", type=str, required=True, help="最终输出的JSONL文件路径。")
    parser.add_argument("--token_limit_per_type", type=int, required=True, help="每种拼接类型（交错/倒序/随机切分/直接拼接）的目标token数量。")
    parser.add_argument("--sequence_length", type=int, required=True, help="拼接后单条序列的目标token长度。")
    
    # 模式开关
    parser.add_argument("--use_interlaced", action='store_true', help="[开关] 是否执行交错拼接") 
    parser.add_argument("--use_reverse", action='store_true', help="[开关] 是否执行倒序拼接")
    parser.add_argument("--use_random_split", action='store_true', help="[开关] 是否执行均分后随机拼接")
    parser.add_argument("--use_direct_concat", action='store_true', help="[开关] 是否执行直接随机拼接（不切分、不倒序）") # 新增参数
    
    parser.add_argument("--tokenizer_name", type=str, default="llm/Llama-2-7b-hf", help="Hugging Face上的分词器名称。")
    parser.add_argument("--num_workers", type=int, default=None, help="用于处理的进程数。")
    
    args = parser.parse_args()

    # 步骤1：并行分词
    all_chunks = load_chunks_parallel(
        input_file=args.input_file,
        tokenizer_name=args.tokenizer_name,
        num_workers=args.num_workers
    )

    if not all_chunks or len(all_chunks) < 8:
        print("错误：有效数据不足8条，无法进行拼接。程序退出。")
        return

    # 步骤2：计算任务量并启动worker进程
    # 计算每种类型的条数
    count_per_type = (args.token_limit_per_type + args.sequence_length - 1) // args.sequence_length
    
    total_interlaced = count_per_type if args.use_interlaced else 0
    total_reverse = count_per_type if args.use_reverse else 0
    total_random_split = count_per_type if args.use_random_split else 0
    total_direct_concat = count_per_type if args.use_direct_concat else 0 # 新增计数

    # 打印任务概览
    task_summary = []
    if args.use_interlaced: task_summary.append(f"交错{total_interlaced}条")
    if args.use_reverse: task_summary.append(f"倒序{total_reverse}条")
    if args.use_random_split: task_summary.append(f"随机切分{total_random_split}条")
    if args.use_direct_concat: task_summary.append(f"直接拼接{total_direct_concat}条") # 新增打印
    
    if not task_summary:
        print("未选择任何拼接模式 (use_interlaced/use_reverse/use_random_split/use_direct_concat)，程序退出。")
        return

    print(f"总任务规划：{'，'.join(task_summary)} (每种类型约 {args.token_limit_per_type / 1e9:.2f}B tokens)")
    
    num_processes = args.num_workers or mp.cpu_count()
    total_tasks = total_interlaced + total_reverse + total_random_split + total_direct_concat
    num_processes = min(num_processes, os.cpu_count(), total_tasks)
    
    if num_processes == 0:
        print("计算出的进程数为0，程序退出。")
        return
        
    # 计算每个进程的负载
    per_proc_interlaced = (total_interlaced + num_processes - 1) // num_processes if total_interlaced > 0 else 0
    per_proc_reverse = (total_reverse + num_processes - 1) // num_processes if total_reverse > 0 else 0
    per_proc_random_split = (total_random_split + num_processes - 1) // num_processes if total_random_split > 0 else 0
    per_proc_direct_concat = (total_direct_concat + num_processes - 1) // num_processes if total_direct_concat > 0 else 0 # 新增负载
    
    print(f"使用 {num_processes} 个进程并行处理...")

    all_indices = list(range(len(all_chunks)))
    processes = []

    for i in range(num_processes):
        # 动态分配，防止超出总数
        start_inter = i * per_proc_interlaced
        curr_inter = min(per_proc_interlaced, max(0, total_interlaced - start_inter)) if total_interlaced > 0 else 0

        start_rev = i * per_proc_reverse
        curr_rev = min(per_proc_reverse, max(0, total_reverse - start_rev)) if total_reverse > 0 else 0
        
        start_rand = i * per_proc_random_split
        curr_rand = min(per_proc_random_split, max(0, total_random_split - start_rand)) if total_random_split > 0 else 0
        
        # 新增动态分配
        start_direct = i * per_proc_direct_concat
        curr_direct = min(per_proc_direct_concat, max(0, total_direct_concat - start_direct)) if total_direct_concat > 0 else 0
        
        if curr_inter > 0 or curr_rev > 0 or curr_rand > 0 or curr_direct > 0:
            p = mp.Process(
                target=worker_process,
                # 参数列表更新
                args=(i, curr_inter, curr_rev, curr_rand, curr_direct, all_chunks, all_indices, args.sequence_length)
            )
            processes.append(p)
            p.start()

    for p in processes:
        p.join()

    # 步骤3：合并并打乱结果
    merge_and_shuffle("temp_sequences_*.jsonl", args.output_file)
    print("--- 所有任务完成 ---")

if __name__ == "__main__":
    main()
