import json
import random
import os
import multiprocessing as mp
from functools import partial
from transformers import AutoTokenizer
import argparse

# LAW synthesis G: order-preserving weaving (Eq. 2 of the paper).
# Input : Stage I selected 8K source documents D* of one domain ("content" + "data_id").
# Output: candidate pool W of 64K sequences ("data_id" + "input_ids") for Stage II scoring.

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
    # 每个进程单独加载tokenizer（多进程安全）
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    texts = []
    for line in chunk:
        try:
            data = json.loads(line)
            texts.append(data.get("content", ""))
        except json.JSONDecodeError:
            texts.append("")  # 无效行用空字符串替代

    batch_size = 256
    all_tokens = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_tokens = batch_tokenize(batch_texts, tokenizer)
        all_tokens.extend(batch_tokens)

    return all_tokens

def load_chunks_parallel(input_file, tokenizer_name="llm/Meta-Llama-3-8B", num_workers=None):
    """并行加载并分词数据"""
    cpu_cores = mp.cpu_count()
    effective_workers = num_workers if num_workers is not None else cpu_cores
    print(f"开始并行分词，使用 {effective_workers} 个进程...")

    # 分割文件为多个分片
    chunks = []
    chunk_size = 2000  # 每个分片2000行
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

    # 多进程处理
    num_processes = min(len(chunks), effective_workers)

    with mp.Pool(num_processes) as pool:
        func = partial(process_chunk, tokenizer_name=tokenizer_name)
        results = pool.map(func, chunks)

    # 展平结果（保持与原始数据顺序一致）
    all_chunks = []
    for res in results:
        all_chunks.extend(res)

    print(f"并行分词完成，共处理 {len(all_chunks)} 条数据")
    return all_chunks

# --------------------------
# 2. 数据拼接模块（worker进程）
# --------------------------
def weave(selected_chunks, rng):
    """Order-preserving weaving: 每篇文档在 floor(n/2) 处二分，均匀采样“每个前半段先于其后半段”的排列。"""
    halves = [(chunk[:len(chunk) // 2], chunk[len(chunk) // 2:]) for chunk in selected_chunks]
    # 对多重集 {1,1,...,m,m} 均匀打乱，每个文档编号第一次出现放前半段，第二次出现放后半段。
    labels = [a for a in range(len(selected_chunks)) for _ in range(2)]
    rng.shuffle(labels)
    seen = set()
    sequence = []
    for a in labels:
        sequence.extend(halves[a][1] if a in seen else halves[a][0])
        seen.add(a)
    return sequence

def worker_process(worker_id, num_sequences, all_chunks, docs_per_sequence, domain, seed, temp_file):
    """生成候选长序列，供 Stage II 打分"""
    print(f"工作进程 {worker_id} (PID: {os.getpid()}) 启动，任务：{num_sequences}条")

    # 每个进程独立的随机数种子，避免 fork 后各进程生成相同序列
    rng = random.Random(seed * 100003 + worker_id)
    all_indices = range(len(all_chunks))

    with open(temp_file, 'w', encoding='utf-8') as f_out:
        for n in range(num_sequences):
            # 同一条序列内不放回采样
            selected_indices = rng.sample(all_indices, k=docs_per_sequence)
            selected_chunks = [all_chunks[i] for i in selected_indices]

            f_out.write(json.dumps({
                "data_id": f"{domain}_s{seed}_{worker_id:03d}_{n:08d}",
                "input_ids": weave(selected_chunks, rng),
                "chunk_indices": selected_indices,
            }) + '\n')

    print(f"工作进程 {worker_id} 完成")

# --------------------------
# 3. 结果合并与打乱模块
# --------------------------
def merge_and_shuffle(temp_files, final_output_file, seed=42):
    """合并临时文件并打乱顺序"""
    print(f"合并临时文件并打乱顺序...")
    all_lines = []

    # 读取所有数据
    for temp_file in temp_files:
        try:
            with open(temp_file, 'r', encoding='utf-8') as f_temp:
                all_lines.extend(f_temp.readlines())
            os.remove(temp_file)  # 删除临时文件
        except Exception as e:
            print(f"处理临时文件 {temp_file} 出错: {e}")

    # 打乱顺序
    random.Random(seed).shuffle(all_lines)

    # 写入最终文件
    with open(final_output_file, 'w', encoding='utf-8') as f_final:
        f_final.writelines(all_lines)

    print(f"合并完成，生成 {final_output_file}，共 {len(all_lines)} 条数据")

# --------------------------
# 主函数：串联整个流程
# --------------------------
def main():
    parser = argparse.ArgumentParser(description="LAW order-preserving weaving: build the Stage II candidate pool for one domain.")
    parser.add_argument("--input_file", type=str, required=True, help="Stage I 筛选后的短文档（content 字段）。")
    parser.add_argument("--output_file", type=str, required=True, help="Stage II 候选池 W 的输出路径。")
    parser.add_argument("--domain", type=str, required=True, help="arxiv / code，用于 data_id 前缀。")
    parser.add_argument("--tokenizer_name", type=str, default="llm/Meta-Llama-3-8B", help="训练模型的分词器。")
    parser.add_argument("--doc_length", type=int, default=8192, help="短文档统一长度 n，不足 n 的文档跳过。")
    parser.add_argument("--sequence_length", type=int, default=65536, help="目标长度 L = m * n。")
    parser.add_argument("--token_budget", type=int, default=500_000_000, help="该领域 Stage II 之后的训练 token 数（1B 按 1:1 混合，每领域 0.5B）。")
    parser.add_argument("--p_long", type=float, default=0.5, help="Stage II 保留比例 p_l，候选池规模 = 预算 / p_l。")
    parser.add_argument("--seed", type=int, default=42, help="控制文档采样与 weaving 排列的随机种子。")
    parser.add_argument("--num_workers", type=int, default=None, help="进程数（默认CPU核心数）。")

    args = parser.parse_args()

    # 步骤1：并行分词
    all_chunks = load_chunks_parallel(
        input_file=args.input_file,
        tokenizer_name=args.tokenizer_name,
        num_workers=args.num_workers
    )

    # 统一文档长度为 n，保证每条序列恰好为 L 个 token
    all_chunks = [chunk[:args.doc_length] for chunk in all_chunks if len(chunk) >= args.doc_length]
    docs_per_sequence = args.sequence_length // args.doc_length

    if len(all_chunks) < docs_per_sequence:
        print(f"错误：长度不少于{args.doc_length}的文档不足{docs_per_sequence}条，无法进行拼接")
        return

    # 步骤2：计算任务量并启动worker进程
    final_sequences = (args.token_budget + args.sequence_length - 1) // args.sequence_length
    # 保证 floor(p_l * |W|) >= final_sequences
    total_sequences = int(final_sequences / args.p_long) + 1

    num_processes = args.num_workers or mp.cpu_count()
    num_processes = min(num_processes, os.cpu_count(), total_sequences)
    per_proc = (total_sequences + num_processes - 1) // num_processes

    print(f"总任务：候选序列 {total_sequences} 条（约{total_sequences * args.sequence_length / 1e9:.2f}B token），"
          f"Stage II 后保留 {final_sequences} 条，每条 {docs_per_sequence} 篇文档，可用文档 {len(all_chunks)} 篇")
    print(f"使用 {num_processes} 个进程并行处理")

    processes = []
    temp_files = []

    for i in range(num_processes):
        curr = min(per_proc, total_sequences - i * per_proc)

        if curr > 0:
            temp_file = f"{args.output_file}.tmp_{i}"
            temp_files.append(temp_file)
            p = mp.Process(
                target=worker_process,
                args=(i, curr, all_chunks, docs_per_sequence, args.domain, args.seed, temp_file)
            )
            processes.append(p)
            p.start()

    # 等待所有worker完成
    for p in processes:
        p.join()

    # 步骤3：合并并打乱结果
    merge_and_shuffle(temp_files, args.output_file, seed=args.seed)
    print("--- 所有任务完成 ---")

if __name__ == "__main__":
    main()
