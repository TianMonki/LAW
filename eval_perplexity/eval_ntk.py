import os
import math
import torch
import argparse
import random
import numpy as np
from tqdm import tqdm
import transformers
from peft import PeftModel
import json
import sys
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.dirname(current_dir)
sys.path.append(models_dir)

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size during inference')
    parser.add_argument('--base_model', type=str, default="/data1/pretrained-models/llama-7b-hf")
    parser.add_argument('--cache_dir', type=str, default="./cache")
    parser.add_argument('--model_name', type=str, default="")
    parser.add_argument('--seq_len', type=int, default=2048, help='context length during evaluation')
    parser.add_argument('--context_size', type=int, default=-1, help='context size during fine-tuning')
    parser.add_argument('--peft_model', type=str, default=None, help='')
    parser.add_argument('--flash_attn', type=bool, default=False, help='')
    parser.add_argument('--data_path', type=str, default="./test.bin", help='')
    parser.add_argument('--base_factor', type=int, default=8, help='base factor to scale rope base')
    parser.add_argument('--output_dir', type=str, default='')
    args = parser.parse_args()
    return args

def get_as_batch(data, seq_length, batch_size, device='cpu', sliding_window=256, rank=0, world_size=1):
    all_ix = list(range(0, len(data) - seq_length, sliding_window))
    all_ix.pop()
    # 数据分片：每个进程只获取自己的一部分数据
    all_ix = all_ix[rank::world_size]

    for idx in range(0, len(all_ix), batch_size):
        ix = all_ix[idx:idx+batch_size]
        assert all([idx + seq_length + 1 <= len(data) for idx in ix])
        x = torch.stack([torch.from_numpy((data[i:i+seq_length]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+seq_length]).astype(np.int64)) for i in ix])
        if device != 'cpu':
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        yield x, y

def iceildiv(x, y):
    return (x + y - 1) // y

def evaluate(model, data, batch_size, device, seq_length, sliding_window=256, use_cache=False, rank=0, world_size=1):
    stats = {}
    model.eval()

    loss_list_val, acc_list = [], []
    loss_step_list_val = []

    doc_start_idx = np.flatnonzero(data['val'] == data['val'][0])
    if rank == 0:
        print('total tokens:', len(data['val']))
        print('selected 10 documents total tokens:', len(data['val'][:doc_start_idx[10]]))
    data['val'] = data['val'][:doc_start_idx[10]]

    with torch.no_grad():
        if rank == 0:
            print(f"Using seq length {seq_length}")
        
        num_local_indices = len(list(range(0, len(data['val']) - seq_length, sliding_window))) // world_size
        total_local_batches = iceildiv(num_local_indices, batch_size)
        
        progress_bar = tqdm(
            enumerate(get_as_batch(
                data['val'], seq_length, batch_size, device=device,
                sliding_window=sliding_window, rank=rank, world_size=world_size
            )),
            total=total_local_batches,
            disable=(rank != 0) # 只在主进程显示进度条
        )

        for idx, (x, y) in progress_bar:
            val_loss, acc, cnt = 0., 0., 0
            for part_idx, i in enumerate(range(0, x.shape[1], seq_length)):
                part_len = x[:, i:i + seq_length].shape[1]
                outputs = model(
                    input_ids=x[:, i:i + seq_length],
                    labels=x[:, i:i+seq_length].contiguous(),
                    use_cache=use_cache
                )
                val_loss = outputs.loss * part_len + val_loss
                acc = ((outputs.logits.argmax(-1) == y[:, i:i+seq_length]).float().sum()) + acc
                cnt += part_len
                while len(loss_step_list_val) <= part_idx:
                    loss_step_list_val.append([])
                loss_step_list_val[part_idx].append(outputs.loss.item())
            val_loss /= cnt
            acc /= cnt
            loss_list_val.append(val_loss.item())
            acc_list.append(acc.item())

    dist.barrier()
    # 汇总所有进程的结果
    gathered_losses = [None] * world_size
    dist.all_gather_object(gathered_losses, loss_list_val)
    gathered_accs = [None] * world_size
    dist.all_gather_object(gathered_accs, acc_list)
    gathered_step_losses = [None] * world_size
    dist.all_gather_object(gathered_step_losses, loss_step_list_val)

    if rank == 0:
        # 在主进程上计算最终指标
        flat_losses = [item for sublist in gathered_losses for item in sublist]
        flat_accs = [item for sublist in gathered_accs for item in sublist]
        num_chunks = len(gathered_step_losses[0]) if gathered_step_losses and gathered_step_losses[0] else 0
        merged_step_losses = [[] for _ in range(num_chunks)]
        for proc_list in gathered_step_losses:
            for chunk_idx, chunk_losses in enumerate(proc_list):
                merged_step_losses[chunk_idx].extend(chunk_losses)
        
        stats['val_acc'] = torch.as_tensor(flat_accs).mean().item()
        stats['val_loss'] = torch.as_tensor(flat_losses).mean().item()
        stats['val_perplexity'] = math.exp(stats['val_loss'])
        mean_loss_per_chunk = [torch.as_tensor(cl).mean().item() for cl in merged_step_losses]
        stats['val_perplexity_per_chunk'] = torch.exp(torch.as_tensor(mean_loss_per_chunk)).tolist()

    return stats

def main(args):
    # --- 分布式环境设置 ---
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    # 自动获取设备号，不再写死为 "cuda:0"
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(device)

    seed = 2
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    data = {'val': np.memmap(args.data_path, dtype=np.uint16, mode='r')}

    if rank == 0:
        print(f"Num validation tokens: {len(data['val'])}")
        print("data path", args.data_path)
        print("base model", args.base_model)
        print("peft model", args.peft_model)

    config = transformers.AutoConfig.from_pretrained(args.base_model, cache_dir=args.cache_dir)

    if args.model_name == "llama-2-7b-hf-ntk-frozen":
        scaling_factor = 2.0
        config.rope_scaling = {"type": "dynamic", "factor": scaling_factor}
        if rank == 0: print(config.rope_scaling)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.base_model, config=config, cache_dir=args.cache_dir,
            torch_dtype=torch.float16, attn_implementation="flash_attention_2"
        ).to(device)
    elif args.model_name == "llama-2-7b-hf-slimpajama-ntk-32k":
        if rank == 0: print(config.rope_scaling)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.base_model, config=config, cache_dir=args.cache_dir,
            torch_dtype=torch.float16, attn_implementation="flash_attention_2"
        ).to(device)
    elif args.model_name == "llama-2-7b-hf-slimpajama-ntk-64k" or args.model_name == "llama-2-7b-hf-slimpajama-ntk-64k-2B":
        if rank == 0: print(config.rope_scaling)
        from models.llama_ntk_64k import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(
            args.base_model, config=config, cache_dir=args.cache_dir,
            torch_dtype=torch.float16, use_flash_attention_2=True
        ).to(device)
    else:
        # 增加一个默认加载方式以提高兼容性
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.base_model, config=config, cache_dir=args.cache_dir,
            torch_dtype=torch.float16, attn_implementation="flash_attention_2"
        ).to(device)

    # 使用DDP包裹模型
    model = DDP(model, device_ids=[local_rank])

    stats = evaluate(model, data, args.batch_size, device, args.seq_len, sliding_window=256, rank=rank, world_size=world_size)

    if rank == 0:
        print(stats)
        with open(args.output_dir, 'w') as json_file:
            json.dump(stats, json_file, indent=4)

    dist.destroy_process_group()

if __name__ == "__main__":
    args = parse_config()
    main(args)