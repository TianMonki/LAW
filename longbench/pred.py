import os
from datasets import load_dataset
import torch
import json
import transformers
from tqdm import tqdm
import numpy as np
import random
import argparse
import sys
import torch.distributed as dist

current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.dirname(current_dir)
sys.path.append(models_dir)

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default=None)
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    return parser.parse_args(args)

def get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_name, rank=0):
    preds = []
    # Disable progress bar on non-main processes
    data_iterator = tqdm(data['test'], disable=(rank != 0))
    for json_obj in data_iterator:
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if "chatglm3" in model_name:
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)

        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]
        kwargs = {}
        kwargs['use_cache'] = True
        if model_name == "llama2-7b-hf-slimpajama-landmark" or model_name == "llama2-7b-hf-slimpajama-landmark-test4k":  
            kwargs['offload_cache_to_cpu'] = False
            kwargs['use_flash'] = False
            kwargs['cache_top_k'] = 5
        if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
                **kwargs,
            )[0]
        else:
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                **kwargs,
            )[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        preds.append({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]})
    return preds

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(path, model_name, device):
    if dist.get_rank() == 0:
        print('testing:', model_name)
        print('model path:', path)
    
    if model_name == "llama2-7b-hf" or model_name == "llama2-7b-hf-pi-64k" or model_name == "llama2-7b-hf-slimpajama-longlora-32k":
        config = transformers.AutoConfig.from_pretrained(path)
        if dist.get_rank() == 0:
            print('rope_scaling:', config.rope_scaling)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            path,
            config=config,
            torch_dtype=torch.float16,
            use_flash_attention_2=True,
        ).to(device)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            path,
            config=config,
        )
    elif model_name == "llama2-7b-hf-slimpajama-ntk-32k":
        config = transformers.AutoConfig.from_pretrained(path)
        if dist.get_rank() == 0:
            print('rope_scaling:', config.rope_scaling)
        from models.llama_ntk_32k import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(
            path,
            config=config,
            torch_dtype=torch.float16,
            use_flash_attention_2=True,
        ).to(device)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            path,
            config=config,
        )
    elif model_name == "llama2-7b-hf-slimpajama-ntk-64k" or model_name == "llama2-7b-hf-slimpajama-ntk-64k-2B":
        config = transformers.AutoConfig.from_pretrained(path)
        if dist.get_rank() == 0:
            print('rope_scaling:', config.rope_scaling)
        from models.llama_ntk_64k import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(
            path,
            config=config,
            torch_dtype=torch.float16,
            use_flash_attention_2=True,
        ).to(device)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            path,
            config=config,
        )
    elif model_name == "llama2-7b-hf-lminfinite":
        from models.llama_infinite import LlamaForCausalLM
        from models.llama_infinite.llama import convert_llama_model
        model = LlamaForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16,
        ).to(device)
        model = convert_llama_model(model, 4096, 10)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            path,
        )
    elif model_name == "llama2-7b-hf-ntk-frozen":
        config = transformers.AutoConfig.from_pretrained(path)
        scaling_factor = 2.0
        if dist.get_rank() == 0:
            print(config.rope_scaling)
        config.rope_scaling = {"type": "dynamic", "factor": scaling_factor}
        model = transformers.AutoModelForCausalLM.from_pretrained(
            path,
            config=config,
            torch_dtype=torch.float16,
            use_flash_attention_2=True,
        ).to(device)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            path,
        )
    elif model_name == "llama2-7b-hf-slimpajama-yarn-32k" or model_name == "llama2-7b-hf-yarn-64k":
        from models.llama_yarn.modeling_llama_yarn import LlamaForCausalLM
        from models.llama_yarn.configuration_llama import LlamaConfig
        config = LlamaConfig.from_pretrained(path)
        if dist.get_rank() == 0:
            print(config.rope_scaling)
        model = LlamaForCausalLM.from_pretrained(
            path,
            config=config,
            torch_dtype=torch.float16,
            use_flash_attention_2=True,
        ).to(device)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            path,
        )
    elif model_name == "llama2-7b-hf-selfextend":
        from transformers import AutoModelForCausalLM
        from models.llama_selfextend import SelfExtend
        window_size, group_size, use_flash = 1024, 64, True
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, use_flash_attention_2=use_flash).to(device)
        if dist.get_rank() == 0:
            print(f'using group size {group_size} using window size {window_size}')
        SelfExtend.apply(model, group_size, window_size, enable_flash_attention=use_flash, flash_attention_impl="flash_attn")
        tokenizer = transformers.AutoTokenizer.from_pretrained(path)
    elif model_name == "llama2-7b-hf-slimpajama-clex-32k":
        if dist.get_rank() == 0:
            print('eval clex')
        from models.llama_clex import LlamaForCausalLM, CLEXLlamaConfig
        config = CLEXLlamaConfig.from_pretrained(path)
        config.log_scale, config.use_flashattn = True, True
        if dist.get_rank() == 0:
            print(config.rope_scaling, flush=True)
        model = LlamaForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            config=config,
            attn_implementation="flash_attention_2",
        ).to(device)
        tokenizer = transformers.AutoTokenizer.from_pretrained(path)
    elif model_name == "llama2-7b-hf-slimpajama-landmark":
        from models.llama_landmark.llama_mem import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            path, padding_side="right", use_fast=False
        )
        tokenizer.padding_side, tokenizer.pad_token = "left", tokenizer.eos_token 
        mem_id = tokenizer.convert_tokens_to_ids("<landmark>")
        model.set_mem_id(mem_id)
        model = model.to(device)
    else:
        # if dist.get_rank() == 0:
        #     print('ERROR! Model_name not exist!')
        # exit()
        config = transformers.AutoConfig.from_pretrained(path)
        if dist.get_rank() == 0:
            print('rope_scaling:', config.rope_scaling)
        from models.llama_ntk_64k import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(
            path,
            config=config,
            torch_dtype=torch.float16,
            use_flash_attention_2=True,
        ).to(device)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            path,
            config=config,
        )
        
    model = model.eval()
    return model, tokenizer

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()

    # --- Distributed Setup ---
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(device)

    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    
    model_name = args.model
    # Each process loads its own copy of the model onto its assigned GPU
    model, tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device)
    max_length = model2maxlen[model_name]
    if rank == 0:
        print('max_length:', max_length)

    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    
    # Create output directories on the main process to avoid race conditions
    if rank == 0:
        if not os.path.exists("pred/"):
            os.makedirs("pred/")
        if not os.path.exists("pred_e"):
            os.makedirs("pred_e")
    
    dataset_path = "data"
    dataset = args.dataset_name
    if rank == 0:
        print('testing:', dataset)

    # Determine paths and load data on all processes
    if args.e:
        data = load_dataset('json', data_files={'test': os.path.join(dataset_path, f'{dataset}_e.jsonl')})
        if rank == 0 and not os.path.exists(f"pred_e/{model_name}"):
            os.makedirs(f"pred_e/{model_name}")
        out_path = f"pred_e/{model_name}/{dataset}.jsonl"
    elif "trec" in dataset and dataset != "trec":
        data = load_dataset('json', data_files={'test': os.path.join(dataset_path, f'{dataset}.jsonl')})
        if rank == 0 and not os.path.exists(f"pred_trec/{model_name}"):
            os.makedirs(f"pred_trec/{model_name}")
        out_path = f"pred_trec/{model_name}/{dataset}.jsonl"
    else:
        data = load_dataset('json', data_files={'test': os.path.join(dataset_path, f'{dataset}.jsonl')})
        if rank == 0 and not os.path.exists(f"pred/{model_name}"):
            os.makedirs(f"pred/{model_name}")
        out_path = f"pred/{model_name}/{dataset}.jsonl"

    # --- Shard the dataset for parallel processing ---
    data_list = list(data['test'])
    sharded_data_list = data_list[rank::world_size]
    sharded_data = {'test': sharded_data_list}

    if "trec" in dataset:
        dataset = "trec"
    prompt_format = dataset2prompt[dataset]
    max_gen = dataset2maxlen[dataset]
    
    # Each process gets predictions for its own data shard
    local_preds = get_pred(model, tokenizer, sharded_data, max_length, max_gen, prompt_format, dataset, device, model_name, rank)
    
    # --- Gather results from all processes ---
    dist.barrier()
    all_preds_gathered = [None] * world_size
    dist.all_gather_object(all_preds_gathered, local_preds)

    # --- Re-order and write output on the main process ---
    if rank == 0:
        # Reconstruct the original order from the sharded results
        final_preds = [None] * len(data_list)
        for i in range(world_size):
            final_preds[i::world_size] = all_preds_gathered[i]

        with open(out_path, "w", encoding="utf-8") as f:
            for pred in final_preds:
                if pred is not None:
                    json.dump(pred, f, ensure_ascii=False)
                    f.write('\n')
    
    # --- Cleanup ---
    dist.destroy_process_group()