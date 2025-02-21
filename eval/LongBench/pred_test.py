import os
from datasets import load_dataset
import torch
import json
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)
from tqdm import tqdm
import numpy as np
import random
import argparse
from utils import add_lbench_args, initialize_engine, process_requests


# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "llama-2" in model_name.lower() or "llama2" in model_name.lower():
        prompt = f"[INST]{prompt}[/INST]"
    return prompt


def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    elif "llama-3" in model_name.lower():
        response = (
            response.split(".assistant")[0]
            .split("\n\nQuestion")[0]
            .split("</s>")[0]
            .strip()
        )
    elif "Llama-2-7B-32K-Instruct" in model_name:
        response = (
            response.split("(Document")[0]
            .split("\n\nQuestion")[0]
            .split("\n\nAnswer")[0]
            .split("(Passage")[0]
            .strip()
        )
    return response


def get_pred(
    lserve_engine,
    tokenizer,
    eos_token_ids,
    data,
    max_length,
    max_gen,
    prompt_format,
    dataset,
    base_model_name,
    # decoding_simulation_length, # FIXME (Shang): What is this for? Seems like useless. Not sure if it is for duo-attention
):
    preds = []
    pbar = tqdm(data)
    prompts = []
    for idx, json_obj in enumerate(pbar):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(
            prompt, truncation=False, return_tensors="pt"
        ).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = tokenizer.decode(
                tokenized_prompt[:half], skip_special_tokens=True
            ) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if idx == 61:
            print(f"Prompt: {prompt}")
            # Save the prompt to txt for debugging
            with open("prompt.txt", "w") as f:
                f.write(prompt)
            exit()
        if dataset not in [
            "trec",
            "triviaqa",
            "samsum",
            "lsht",
            "lcc",
            "repobench-p",
        ]:  # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, base_model_name)

        prompts.append(prompt)
        # break   # Only add one prompt for DEBUGGINg

    # NOTE (Shang): Fix by adding eos_token_ids
    outputs = process_requests(lserve_engine, prompts, eos_token_ids, max_gen)

    for idx, json_obj in enumerate(pbar):
        pred = outputs[idx]
        # pred = pred.replace("<|eot_id|>", "") # FIXME (Shang): What is this for? Why do we need it?
        pred = post_process(pred, base_model_name)
        preds.append(
            {
                "pred": pred,
                "answers": json_obj["answers"],
                "all_classes": json_obj["all_classes"],
                "length": json_obj["length"],
            }
        )
    return preds


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False
    )
    generation_config = GenerationConfig.from_pretrained(model_path)
    lserve_engine, lserve_args = initialize_engine(args.model_path)
    
    eos_token_ids = generation_config.eos_token_id
    if not isinstance(eos_token_ids, list):
        eos_token_ids = [eos_token_ids]

    return lserve_engine, tokenizer, eos_token_ids


if __name__ == "__main__":
    seed_everything(42)
    parser = argparse.ArgumentParser()
    add_lbench_args(parser)
    args, _ = parser.parse_known_args()
    # model2path = json.load(open("eval/LongBench/config/model2path.json", "r"))
    model2maxlen = json.load(open("./config/model2maxlen.json", "r"))
    base_model_name = args.base_model
    quant_model_name = args.quant_model
    # define your model
    lserve_engine, tokenizer, eos_token_ids = load_model_and_tokenizer(args.model_path)

    max_length = model2maxlen[base_model_name]
    if args.e:
        datasets = [
            "qasper",
            "multifieldqa_en",
            "hotpotqa",
            "2wikimqa",
            "gov_report",
            "multi_news",
            "trec",
            "triviaqa",
            "samsum",
            "passage_count",
            "passage_retrieval_en",
            "lcc",
            "repobench-p",
        ]
    else:
        print(f"args.task: {args.task}")
        tasks = args.task.split("+")
        print(f"tasks: {tasks}")
        datasets = tasks
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("./config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("./config/dataset2maxlen.json", "r"))
    # predict on each dataset
    if not os.path.exists("./pred"):
        os.makedirs("./pred")
    if not os.path.exists("./pred_e"):
        os.makedirs("./pred_e")
    for dataset in datasets:
        data = load_dataset("THUDM/LongBench", dataset, split="test")
        if not os.path.exists(f"./pred/{quant_model_name}"):
            os.makedirs(f"./pred/{quant_model_name}")
        # if args.method == "duo_attn":
        #     out_path = f"eval/LongBench/pred/{model_name}/{dataset}-duo_attn-pattern-{args.attn_load_dir.split('/')[-1]}-sp-{args.sparsity}.jsonl"
        # elif args.method == "h2o":
        #     out_path = f"eval/LongBench/pred/{model_name}/{dataset}-h2o-hr-{args.heavy_ratio}-rr-{args.recent_ratio}.jsonl"
        # elif args.method == "streaming":
        #     out_path = f"eval/LongBench/pred/{model_name}/{dataset}-streaming-br-{args.budget_ratio}.jsonl"
        # elif args.method == "tova":
        #     out_path = f"eval/LongBench/pred/{model_name}/{dataset}-tova-br-{args.budget_ratio}.jsonl"
        # if args.method == "lServeWContext":
        #     pass
        # else:
        out_path = f"./pred/{quant_model_name}/{dataset}-{args.model_name_suffix}.jsonl"
        if os.path.exists(out_path):
            print(f"{out_path} already exists, skipping...")
            continue
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        
        
        preds = get_pred(
            lserve_engine,
            tokenizer,
            eos_token_ids,
            data,
            max_length,
            max_gen,
            prompt_format,
            dataset,
            base_model_name,
        )
        with open(out_path, "w", encoding="utf-8") as f:
            for pred in preds:
                json.dump(pred, f, ensure_ascii=False)
                f.write("\n")