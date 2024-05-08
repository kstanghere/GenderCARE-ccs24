
import torch
import sys
import argparse
import pandas as pd
import deepspeed
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.generation.utils import GenerationConfig


def calculate_perplexity(text, tokenizer, model, device):
    input = tokenizer.encode(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(input, labels=input)
        loss = outputs.loss
    return torch.exp(loss).item()

def get_zero_ds_config(offload, stage=2):

    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "offload_param": {
            "device": device
        },
        "offload_optimizer": {
            "device": device
        },
        "stage3_param_persistence_threshold": 1e4,
        "stage3_max_live_parameters": 3e7,
        "stage3_prefetch_bucket_size": 3e7,
        "memory_efficient_linear": False
    }
    return {"zero": zero_opt_dict,}

def get_model(config, model_path, tokenizer):

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        from_tf=bool(".ckpt" in model_path),
        config=config,
        ignore_mismatched_sizes=True,
        trust_remote_code=True
    )
    model.resize_token_embeddings(len(tokenizer))
    # prepare the tokenizer and model config
    tokenizer.pad_token = tokenizer.eos_token
    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained model")
    parser.add_argument("--output_file", type=str, required=True, help="Output file for the results")
    parser.add_argument("--data_path", type=str, required=True, help="Input file for the predict")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument('--offload', action='store_true', help='Enable ZeRO Offload techniques.')
    parser.add_argument('--zero_stage', type=int, default=0, help='ZeRO optimization stage for Actor model (and clones).')
    parser.add_argument("--model_type",
                        type=str,
                        default="llama2",
                        choices=["alpaca", "open_llama", "llama2", "vicuna", "orca", "falcon", "baichuan_base", "baichuan_chat", "stablebeluga", "mistral", "platypus2", "stableBeluga"],
                        required=True)
    args = parser.parse_args()

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed()
    
    output_file = args.output_file
    data_path = args.data_path
    ds_config = get_zero_ds_config(offload=args.offload, stage=args.zero_stage)

    if args.model_type == "baichuan_base":
        print("baichuan_base_model loading")
        tokenizer_baseline = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        print("baichuan_base_model loaded")
        model_baseline = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)

    elif args.model_type == "baichuan_chat":
        print("baichuan_chat_model_loading")
        tokenizer_baseline = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, trust_remote_code=True)
        print("baichuan_chat_model_loading")
        model_baseline = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
        model_baseline.generation_config = GenerationConfig.from_pretrained(args.model_path)

    else:
        config_baseline = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
        
        print("model config loaded")
        if args.model_type == "alpaca":
            tokenizer_baseline = AutoTokenizer.from_pretrained(args.model_path, fast_tokenizer=True, unk_token="<unk>",
                                                    bos_token="<s>",
                                                    eos_token="</s>")
        else:
            tokenizer_baseline = AutoTokenizer.from_pretrained(args.model_path,fast_tokenizer=True)
        print("model tokenizer loaded")
        model_baseline= get_model(config_baseline, args.model_path, tokenizer_baseline)

    ds_model = deepspeed.init_inference(
        model=model_baseline,     
        mp_size=1,        # GPU numbers
        dtype=torch.float16, 
        replace_method="auto", 
        replace_with_kernel_inject=True, 
        config=ds_config, 
        )
    
    print(f"model loaded at:{ds_model.module.device}\n")

    data = pd.read_csv(data_path)
    results = []
    perplexity1_sum = 0
    perplexity2_sum = 0
    perplexity_diff_sum = 0
    total_rows = 0
    temp = 0

    for index, row in data.iterrows():
        text1 = row["sent_x"]
        text2 = row["sent_y"]

        perplexity1 = calculate_perplexity(text1, tokenizer_baseline, ds_model, device)
        perplexity2 = calculate_perplexity(text2, tokenizer_baseline, ds_model, device)
        total_rows += 1

        perplexity_diff = perplexity1 - perplexity2

            
        perplexity1_sum += perplexity1
        perplexity2_sum += perplexity2
        perplexity_diff_sum += perplexity_diff
        results.append({"Row": index, "sent_x": perplexity1, "sent_y": perplexity2, "perplexity_diff": perplexity_diff})
        print(f"Row {index}: sent_x: {perplexity1}; sent_y: {perplexity2}; perplexity_diff: {perplexity_diff}")
    
    result_df = pd.DataFrame(results)
    average_perplexity1 = perplexity1_sum / total_rows
    average_perplexity2 = perplexity2_sum / total_rows
    average_perplexity_diff = perplexity_diff_sum / total_rows

    print(f"average_perplexity1: {average_perplexity1}; average_perplexity2: {average_perplexity2}; average_perplexity_diff: {average_perplexity_diff};")
    result_df.loc["Averages"] = ["Averages", average_perplexity1, average_perplexity2, average_perplexity_diff]

    average_perplexity1_standardization = average_perplexity1 / (average_perplexity1+average_perplexity2)
    average_perplexity2_standardization = average_perplexity2 / (average_perplexity1+average_perplexity2)
    delta_average_perplexity1_perplexity2 = average_perplexity1_standardization - average_perplexity2_standardization

    result_df.loc["Average probability standardization"] = ["Average probability standardization", average_perplexity1_standardization, average_perplexity2_standardization, delta_average_perplexity1_perplexity2]
    print(f"average_sentmore_perplexity1_standardization: {average_perplexity1_standardization}; average_sentless_perplexity2_standardization: {average_perplexity2_standardization}; average_stereo-anti-stero: {delta_average_perplexity1_perplexity2};")

    result_df.to_csv(output_file, index=False)


if __name__ == "__main__":
    main()