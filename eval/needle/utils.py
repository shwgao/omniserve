import glob

def load_context(fpath="eval/needle/PaulGrahamEssays/*.txt", ctx_len=100000):
    context = ""
    for file in glob.glob(fpath):
        with open(file, 'r') as f: 
            context += f.read()
    LLAMA_CHAR_TO_TOKEN_RATIO = 3.66
    context = context[: int(ctx_len * LLAMA_CHAR_TO_TOKEN_RATIO)]
    return context

def insert_needle(context, needle, depth):
    context = context.split(".")
    c_len = len(context)
    needle_place = int(depth * c_len)
    context = ".".join(context[:needle_place]) + "." + needle + ".".join(context[needle_place:])
    return context

def add_NIH_args(parser):
    parser.add_argument("-s", "--s_len", metavar="N", type=int, help="a number")
    parser.add_argument("-e", "--e_len", metavar="N", type=int, help="a number")
    parser.add_argument("--model_path", type=str, default=None, help="path to model")
    parser.add_argument("--model_name", type=str, default=None, help="name of model")
    parser.add_argument(
        "--model_name_suffix", type=str, default=None, help="name of model"
    )
    parser.add_argument(
        "--model_provider", type=str, default="LLaMA", help="which model to use"
    )
    parser.add_argument("--api_key", type=str, default="", help="OpenAI API Key")
    parser.add_argument(
        "--attn_load_dir", type=str, default=None, help="attention pattern directory"
    )
    parser.add_argument("--start_size", type=int, default=None)
    parser.add_argument("--recent_size", type=int, default=None)
    parser.add_argument("--context_lengths_num_intervals", type=int, default=40)
    parser.add_argument("--sparsify_threshold", type=float, default=0.5)
    parser.add_argument("--sparsity", type=float, default=0.5)


# for lServe
import argparse
from typing import List, Tuple
import random

import datasets

import omniserve.utils.constants
from omniserve import EngineArgs, LLMEngine, SamplingParams
from omniserve.conversation import get_conv_template_name, get_conv_template

def _initialize_engine(args: argparse.Namespace) -> LLMEngine:
    """Initialize the LLMEngine from the command line arguments."""
    engine_args = EngineArgs.from_cli_args(args)
    return LLMEngine.from_engine_args(engine_args)

def initialize_engine(model) -> LLMEngine:
    """Initialize the LLMEngine from the command line arguments."""
    parser = argparse.ArgumentParser(description="Demo on using the LLMEngine class directly")
    parser = EngineArgs.add_cli_args(parser)
    args, _ = parser.parse_known_args()
    args.model = model
    return _initialize_engine(args), args

max_seq_len = omniserve.utils.constants.max_seq_len
BG_BLUE = "\033[44m"
BG_GREEN = "\033[42m"
BG_PINK = "\033[45m"
RESET = "\033[0m"


from omniserve.conversation import get_conv_template_name, get_conv_template

def create_test_prompt(model, raw_prompt):
    conv_t = get_conv_template_name(model)
    print(f"[in create test prompt] {model}")
    print(f"[in create test prompt] {conv_t}")
    conv = get_conv_template(conv_t)
    conv.append_message(conv.roles[0], raw_prompt)
    conv.append_message(conv.roles[1], "")
    return conv.get_prompt()


def process_requests(engine: LLMEngine, test_prompt: str):
    """Continuously process a list of prompts and handle the outputs."""
    request_id = 0
    sampling_params = SamplingParams(
        temperature=0.0, top_p=1.0, stop_token_ids=[128001, 128009], max_tokens=128
    )
    test_prompts = [(test_prompt, sampling_params)]
    while test_prompts or engine.has_unfinished_requests():
        if test_prompts:
            prompt, sampling_params = test_prompts.pop(0)
            succeeded = engine.add_request(str(request_id), prompt, sampling_params)
            if succeeded:
                request_id += 1
        num_test_prompts = request_id

        if not test_prompts:
            break

    if engine.ifb_mode == False:
        # We need to pre-caulcate the block table size for initialization
        block_size = engine.cache_config.block_size
        max_context_length = 128
        max_gen_length = 384
        tot_length = (
            max_context_length + max_gen_length
        )  # Set the upper bound for (prompt + gen) length
        init_num_blocks = (tot_length + block_size - 1) // block_size
        engine.update_init_num_blocks(init_num_blocks)

    # seq_group_metadata_list, scheduler_outputs = engine.step()
    iter = 1
    finished = 0
    while engine.has_unfinished_requests():
        ### Schedule iteration 1 (context stage)
        requests_outputs = engine.step()
        if len(requests_outputs) == 0:
            break
        # print(
        #     BG_BLUE
        #     + "*" * 5
        #     + "Iteration %d (remaining req.s = %d)"
        #     % (iter, len(requests_outputs) + len(engine.scheduler.waiting))
        #     + "*" * 5
        #     + RESET
        # )
        for request_output in requests_outputs:
            if request_output["finished"]:
                finished += 1
                # print(
                #     f"{BG_GREEN}[Conversation {request_output['id']} output]{RESET} {request_output['text']}"
                # )
        iter += 1
        if engine.ifb_mode == False:
            if iter == max_gen_length:  # Early exit
                # for request_output in requests_outputs:
                    # print(
                    #     f"{BG_GREEN}[Conversation {request_output['id']} output]{RESET} {request_output['tokens']}"
                    # )
                break
    assert num_test_prompts == finished
    
    for request_output in requests_outputs:
        return request_output['text']
    
    print(f"{BG_PINK}{finished} requests are finished.{RESET}")

