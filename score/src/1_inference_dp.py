import os
import time
import argparse
import torch
from transformers import AutoTokenizer, AutoConfig
from tqdm import tqdm
from LongAttn import CustomLlamaForCausalLM
import jsonlines
from accelerate import Accelerator

# LAW scoring Q = T * U for both stages, using the first decoder layer of the frozen scorer.
#   Stage I : intact 8K source documents ("content" text),        --max_length 8192  -> k_s = 2048
#   Stage II: woven 64K sequences ("input_ids" from data_main.py), --max_length 65536 -> k_l = 16384
# k = n // margin with margin = 4. The training model (LLaMA-3 8B) and the scorer (LLaMA-3.1 70B)
# share one tokenizer, so Stage II scores the stored input_ids directly.
# Each output line: {"data_id", "T", "U", "Q"}.


def inference(data_id, input_ids, attention_mask, model):
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    output = []
    for i in range(len(data_id)):
        t_score = outputs.proportions[0][i, 0].item()
        u_score = outputs.uniformities[0][i, 0].item()
        output.append({"data_id": data_id[i], "T": t_score, "U": u_score, "Q": t_score * u_score})
    return output

def process_batch(batch, llm_tokenizer, max_length):
    if 'content' in batch[0]:
        # add_special_tokens=True inserts a single BOS token.
        tokens = llm_tokenizer(
            [data['content'] for data in batch],
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
    else:
        tokens = llm_tokenizer.pad(
            [{"input_ids": data['input_ids'][:max_length]} for data in batch],
            padding='max_length',
            max_length=max_length,
            return_tensors='pt',
            return_attention_mask=True
        )
    return tokens["input_ids"], tokens['attention_mask']

def process_file(data_list, llm_tokenizer, model, batch_size, max_length, output_file, device):
    batch = []
    with jsonlines.open(output_file, mode='w') as writer:
        for idx, data in enumerate(tqdm(data_list)):
            batch.append(data)
            if len(batch) >= batch_size or idx == len(data_list) - 1:
                input_ids, attention_mask = process_batch(batch, llm_tokenizer, max_length)
                data_id = [item['data_id'] for item in batch]
                inference_res = inference(data_id, input_ids.to(device), attention_mask.to(device), model)
                writer.write_all(inference_res)
                batch = []

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LAW attention scoring (Stage I and Stage II).')

    parser.add_argument('input_file_path', type=str, help='The path to the input file')
    parser.add_argument('output_file_path', type=str, help='The path to the output file')
    parser.add_argument('batch_size', type=int, help='The batch size for processing')
    parser.add_argument('margin', type=int, help='Distance threshold k = n // margin (4 in the paper)')
    parser.add_argument('--max_length', type=int, default=8192, help='8192 for Stage I, 65536 for Stage II')
    parser.add_argument('--model_path', type=str, default="llm/Llama-3.1-70B")
    parser.add_argument('--rope_theta', type=float, default=None, help='Override rope_theta; default keeps the checkpoint value')
    parser.add_argument('--row_block', type=int, default=4096, help='Query rows per attention block')

    args = parser.parse_args()

    accelerator = Accelerator()

    config_kwargs = {
        "cache_dir": None,
        "revision": 'main',
        "use_auth_token": None,
    }
    if args.rope_theta is not None:
        config_kwargs["rope_theta"] = args.rope_theta

    config = AutoConfig.from_pretrained(args.model_path, **config_kwargs)
    config.num_hidden_layers = 1
    config.law_row_block = args.row_block
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    llama_model = CustomLlamaForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        margin=args.margin,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map={"": accelerator.process_index}
    )

    llama_model.eval()

    with jsonlines.open(args.input_file_path) as reader:
        data_list = [obj for obj in reader]

    start_time = time.time()

    # Each process writes its own part file; the main process merges them.
    part_file = f"{args.output_file_path}.part_{accelerator.process_index}"
    with accelerator.split_between_processes(data_list) as data_parts:
        print(f"Processing {len(data_parts)} samples, process index: {accelerator.process_index}")
        process_file(data_parts, tokenizer, llama_model, args.batch_size,
                     args.max_length, part_file, accelerator.device)

    accelerator.wait_for_everyone()
    print(f"process {accelerator.process_index} 计算时间: {time.time() - start_time} 秒")

    if accelerator.is_main_process:
        with jsonlines.open(args.output_file_path, mode='w') as writer:
            for i in range(accelerator.num_processes):
                part = f"{args.output_file_path}.part_{i}"
                if os.path.exists(part):
                    with jsonlines.open(part) as reader:
                        writer.write_all(reader)
                    os.remove(part)
        print(f"Merged results saved to {args.output_file_path}")
