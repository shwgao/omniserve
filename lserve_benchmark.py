# File authors: Haotian Tang, Shang Yang, Yujun Lin, Song Han
# @article{lin2024qserve,
#   title={QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving},
#   author={Lin*, Yujun and Tang*, Haotian and Yang*, Shang and Zhang, Zhekai and Xiao, Guangxuan and Gan, Chuang and Han, Song},
#   year={2024}
# }
# @article{yang2025lserve,
#   title={LServe: Efficient Long-sequence LLM Serving with Unified Sparse Attention},
#   author={Yang*, Shang and Guo*, Junxian and Tang, Haotian and Hu, Qinghao and Xiao, Guangxuan and Tang, Jiaming and Lin, Yujun and Liu, Zhijian and Lu, Yao and Han, Song},
#   year={2025}
# }

import argparse
import time
import gc
import torch

import omniserve.utils.constants
from omniserve import EngineArgs, LLMEngine, SamplingParams
from omniserve.config import ProfilingConfig

max_seq_len = omniserve.utils.constants.max_seq_len

import os

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="1"


def process_requests(
    engine: LLMEngine, batch_size: int, prompt_len: int, generation_len: int
):
    """Continuously process a list of prompts and handle the outputs."""
    request_id = 0
    profiling_config = ProfilingConfig(
        prompt_len=prompt_len, generation_len=generation_len
    )
    for b in range(batch_size):
        engine.add_request(
            str(b),
            prompt=None,
            profiling_config=profiling_config,
            sampling_params=SamplingParams(top_p=0.95, top_k=1, temperature=0.0),
        )

    if engine.ifb_mode == False:
        # We need to pre-caulcate the block table size for initialization
        block_size = engine.cache_config.block_size
        tot_length = prompt_len + generation_len
        init_num_blocks = (tot_length + block_size - 1) // block_size
        engine.update_init_num_blocks(init_num_blocks)

    # seq_group_metadata_list, scheduler_outputs = engine.step()
    iter = 1

    time_lis = []
    num_tokens = 0
    torch.cuda.synchronize()
    st = time.time()

    while engine.has_unfinished_requests():
        ### Schedule iteration 1 (context stage)
        requests_outputs = engine.step()
        num_tokens += len(requests_outputs)
        # torch.cuda.synchronize()
        if len(requests_outputs) == 0:
            break

        iter += 1
        if engine.profiling_mode and iter == generation_len + 1:
            break
    torch.cuda.synchronize()
    ed = time.time()
    time_lis.append(ed - st)
    return time_lis, num_tokens



def process_requests_split_stage(
    engine: LLMEngine, batch_size: int, prompt_len: int, generation_len: int
):
    """Continuously process a list of prompts and handle the outputs."""
    """Benchmark context & decoding speed seperately"""
    request_id = 0
    profiling_config = ProfilingConfig(
        prompt_len=prompt_len, generation_len=generation_len
    )
    for b in range(batch_size):
        engine.add_request(
            str(b),
            prompt=None,
            profiling_config=profiling_config,
            sampling_params=SamplingParams(top_p=0.95, top_k=1, temperature=0.0),
        )

    if engine.ifb_mode == False:
        # We need to pre-caulcate the block table size for initialization
        block_size = engine.cache_config.block_size
        tot_length = prompt_len + generation_len
        init_num_blocks = (tot_length + block_size - 1) // block_size
        engine.update_init_num_blocks(init_num_blocks)

    # seq_group_metadata_list, scheduler_outputs = engine.step()
    iter = 1

    time_lis = []   # time_lis[0] is the context latency, other's are decoding latency
    ctx_tokens = 0
    dec_tokens = 0

    while engine.has_unfinished_requests():
        ### Schedule iteration 1 (context stage)
        torch.cuda.synchronize()
        st = time.time()

        requests_outputs = engine.step()

        torch.cuda.synchronize()
        ed = time.time()
        time_lis.append(ed - st)
        ctx_tokens += len(requests_outputs)
        iter += 1
        break

    torch.cuda.synchronize()
    st = time.time()
    while engine.has_unfinished_requests():
        ### Schedule iteration 2-n (decoding stage)

        requests_outputs = engine.step()

        dec_tokens += len(requests_outputs)
        # torch.cuda.synchronize()
        if len(requests_outputs) == 0:
            break

        iter += 1
        if engine.profiling_mode and iter >= generation_len + 1:
            break

    torch.cuda.synchronize()
    ed = time.time()
    time_lis.append(ed - st)

    return time_lis, ctx_tokens, dec_tokens


def initialize_engine(args: argparse.Namespace) -> LLMEngine:
    """Initialize the LLMEngine from the command line arguments."""
    engine_args = EngineArgs.from_cli_args(args)
    return LLMEngine.from_engine_args(engine_args)


def main(args: argparse.Namespace):
    """Main function that sets up and runs the prompt processing."""

    gpu_capabilites = torch.cuda.get_device_properties(0)
    # print("GPU Name:", gpu_capabilites.name)
    str = gpu_capabilites.name
    if "A100" in str:
        device_name = "A100"
    elif "A6000" in str:
        device_name = "A6000"
    elif "4090" in str:
        device_name = "RTX4090"
    else:
        print("Unsupported GPU")
    if "PCIe" in str:
        device_name += "_PCIe"
    else:
        device_name += "_SXM"
    
    print("Device Name:", device_name)


    batch_size = int(os.environ.get("GLOBAL_BATCH_SIZE"))
    prompt_len = int(os.environ.get("GLOBAL_PROMPT_LEN"))
    generation_len = int(os.environ.get("GLOBAL_GENERATE_LEN"))
    rounds = 5
    exact_model_name = args.model.split("/")[-1]
    result_file_path = f"./results/profile_results/{exact_model_name}_results_bts{batch_size}_plen{prompt_len}_glen{generation_len}_persition{args.precision}_sparsity{args.static_sparsity}_sparse_context_mode{args.sparse_context_mode}_device{device_name}.csv"

    os.makedirs(os.path.dirname(result_file_path), exist_ok=True)
    
    with open(result_file_path, "a") as file:
        print("=" * 50, file=file)
        print(
            f"{args.model}: Batch={batch_size}, Input={prompt_len}, Output={generation_len}",
            file=file,
        )

    with torch.no_grad():
        for rnd in range(rounds):
            if rnd < rounds - 1:
                print("[Warmup Round %d]" % rnd)
            engine = initialize_engine(args)
            engine.profiling_mode = True
            # warm up
            # time_lis, num_tokens = process_requests(
            #     engine,
            #     batch_size=batch_size,
            #     prompt_len=prompt_len,
            #     generation_len=generation_len,
            # )
            time_lis, ctx_tokens, dec_tokens = process_requests_split_stage(
                engine,
                batch_size=batch_size,
                prompt_len=prompt_len,
                generation_len=generation_len,
            )
            del engine
            torch.cuda.empty_cache()
            gc.collect()
            print(f"Round {rnd} Time List:", time_lis, "(second)")
            throughput = (ctx_tokens + dec_tokens) / sum(time_lis)
            print(f"Round {rnd} Throughput:", throughput, "tokens / second.")
            print(f"Round {rnd} ctx_lentency:", time_lis[0], "second.")
            # print(f"Round {rnd} dec_lentency:", time_lis[1], "second.")
            print(f"Round {rnd} dec_lentency:", time_lis[1] / dec_tokens, "second / token.")
            with open(result_file_path, "a") as file:
                print(
                    f"Round {rnd} Throughput:",
                    throughput,
                    "tokens / second.",
                    file=file,
                )

    with open(result_file_path, "a") as file:
        print("=" * 50, file=file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demo on using the LLMEngine class directly"
    )
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)
