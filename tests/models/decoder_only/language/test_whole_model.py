import torch
from collections import Counter as cCounter
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed
import logging
import os, time
import numpy as np
import math
# import sccl
import sys
import argparse
from transformers import AutoTokenizer
import json
from datetime import datetime

from vllm.platforms.sophtpu import get_soph_config_manager
from vllm.logger import init_logger
from vllm.engine.llm_engine import LLMEngine
from vllm.usage.usage_lib import UsageContext
from typing import (Sequence, Union, cast)
from vllm.engine.arg_utils import (EngineArgs)
from vllm.sampling_params import ( SamplingParams)
from vllm.inputs import PromptType
from vllm.pooling_params import PoolingParams
from vllm.utils import Counter

logger = init_logger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    default="qwen2-7b",
    # required=True,
    choices=["llama2-7b", "llama3.1-8b", "llama3.1-70b", "qwen2.5-32b", "qwen2.5-14b", "qwen2.5-7b","qwen2-7b", "qwen2-72b", "qwen2-57b-a14b", "qwen3-32b", "qwen3-4b", "qwen3-8b", "qwen3-235b-a22b",
             "qwq", "deepseek_v2", "deepseek_v3"],
    help="Model name to test (e.g., llama2-7b, llama3.1-8b, llama3.1-70b, qwen2.5-32b, qwen2.5-14b, qwen2-7b, qwen2-57b-a14b, qwen3-32b, qwq, qwen3-8b, deepseek_v2, deepseek_v3)",
)
parser.add_argument(
    "--quantize",
    type=str,
    default=None,
    choices=["gptq", "awq"],
    help="Quantization method (e.g., gptq, awq)",
)
parser.add_argument("--batch", type=int, default=1, help="Batch size for testing")
parser.add_argument("--path", type=str, default="/data/", help="Path to model dir")
parser.add_argument("--mode", type=str, default="chat", choices=["chat", "generation"], help="Run with chat mode or generation mode")
parser.add_argument("--useV1", type=bool, default=True, help="Use vLLM V1. Set False to use V0.")
parser.add_argument("--tp_size", type=int, default=1, help="Tensor Parallel size. Set to 1 to disable tensor parallel.")
parser.add_argument(
    "--save-json",
    type=str,
    help="Save outputs to specified file in json format.",
)
args = parser.parse_args()

RANK = int(os.environ.get("OMPI_COMM_WORLD_RANK", "0"))
WORLD_SIZE = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1"))
SIMULATE_RANK_NUM = int(os.environ.get("SIMULATE_RANK_NUM", "1"))
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "6000"

log_file = f"{os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))}/run_local_{RANK}.log"
if os.path.exists(log_file):
    os.remove(log_file)
# logger.add(log_file, level=soph_config.SLOG_LEVEL)
soph_config_manager = get_soph_config_manager()
DECODE_TOKEN_LEN = soph_config_manager.get('DECODE_TOKEN_LEN')
CONTEXT_LEN = soph_config_manager.get('CONTEXT_LEN')

def default_llm_engine(model, quantize, path, use_v1, tp_size, batches):
    model_configs = {
        "llama2-7b": {
            "gptq": {"model_id": "Llama-2-7b-Chat-GPTQ", "dtype": "float16"},
            "default": {"model_id": "llama-2-7b-chat-hf", "dtype": "float16"},
        },
        "llama3.1-8b": {
            "gptq": {"model_id": "Llama-3.1-8B-Instruct-GPTQ-INT4", "dtype": "float16"},
            "default": {"model_id": "Llama-3.1-8B-Instruct", "dtype": "float16"},
        },
        "llama3.1-70b": {
            "gptq": {"model_id": "Meta-Llama-3.1-70B-Instruct-GPTQ-INT4", "dtype": "float16"},
            "default": {"model_id": "Llama-3.1-70B-Instruct", "dtype": "float16"},
        },
        "qwen2.5-32b": {
            "gptq": {"model_id": "Qwen2.5-32B-Instruct-GPTQ-Int4", "dtype": "float16"},
            "default": {"model_id": "Qwen2.5-32B-Instruct", "dtype": "bfloat16"}
        },
        "qwen2.5-14b": {
            "gptq": {"model_id": "Qwen2.5-14B-Instruct-GPTQ-Int4", "dtype": "float16"},
            "default": {"model_id": "Qwen2.5-14B-Instruct", "dtype": "bfloat16"}
        },
        "qwen2-7b": {
            "gptq": {"model_id": "Qwen2-7B-Instruct-GPTQ-Int4","dtype": "float16"},
            "default": {"model_id": "Qwen2-7B-Instruct", "dtype": "bfloat16"}
        },
        "qwen2-72b": {
            "gptq": {"model_id": "Qwen2-72B-Instruct-GPTQ-Int4","dtype": "float16"},
            "default": {"model_id": "Qwen2-72B-Instruct", "dtype": "bfloat16"}
        },        
        "qwen2-57b-a14b": {
            "gptq": {"model_id": "Qwen2-57B-A14B-Instruct-GPTQ-Int4","dtype": "float16"},
            "default": {"model_id": "Qwen2-57B-A14B-Instruct", "dtype": "bfloat16"}
        },
        "qwen3-4b": {
            "gptq": {"model_id": "Qwen3-4B-GPTQ-Int4","dtype": "bfloat16"},
            "default": {"model_id": "Qwen3-4B", "dtype": "bfloat16"}
        },
        "qwen3-8b": {
            "gptq": {"model_id": "Qwen3-8B","dtype": "float16"},
            "default": {"model_id": "Qwen3-8B", "dtype": "bfloat16"}
        },
        "qwen3-235b-a22b": {
            "gptq": {"model_id": "Qwen3-235B-A22B","dtype": "float16"},
            "default": {"model_id": "Qwen3-235B-A22B", "dtype": "bfloat16"}
        },
        "qwen3-32b": {
            "gptq": {"model_id": "Qwen3-32B-GPTQ-Int4","dtype": "bfloat16"},
            "default": {"model_id": "Qwen3-32B", "dtype": "bfloat16"}
        },
        "qwq": {
            "awq": {"model_id": "QwQ-32B-AWQ","dtype": "float16"},
            "default": {"model_id": "QwQ-32B", "dtype": "bfloat16"}
        },
        "deepseek_v2": {
            "gptq": {"model_id": "DeepSeek-V2-Lite-chat", "dtype": "bfloat16"},
            "default": {"model_id": "DeepSeek-V2-Lite", "dtype": "bfloat16"},
        },
        "deepseek_v3": {
            "gptq": {"model_id": "DeepSeek-V3", "dtype": "bfloat16"},
            "default": {"model_id": "DeepSeek-V3", "dtype": "bfloat16"},
        },
    }

    model_info = model_configs.get(model, {}).get(quantize, model_configs.get(model, {}).get("default"))
    if model_info is None:
        raise ValueError(f"Unsupported model or quantization: {model}, {quantize}")
    model_id = os.path.join(path, model_info["model_id"])

    prompts, num_prompts_total_tokens = defatult_multi_batch_prompt(model_id, batches)
    soph_config_manager.config['CURRENT_BATCH_SIZE'] = batches
    max_model_len = DECODE_TOKEN_LEN + math.ceil(num_prompts_total_tokens / batches) # max_len per (prompot+decodes)

    dtype = model_info["dtype"]

    engine_args = EngineArgs(
        model=model_id,
        dtype=dtype,
        max_model_len=max_model_len,
        enforce_eager=True,
        trust_remote_code=True,
        tensor_parallel_size=tp_size,
        distributed_executor_backend=None if tp_size == 1 else 'mp'
    )
    if use_v1:
        os.environ["VLLM_USE_V1"] = "1"
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        from vllm.v1.engine.llm_engine import LLMEngine as V1LLMEngine
        engine_class = V1LLMEngine
    else:
        engine_class = LLMEngine

    llm_engine = engine_class.from_engine_args(
        engine_args, usage_context=UsageContext.LLM_CLASS)
    return llm_engine, prompts

def get_default_sampling_params(llm_engine) -> SamplingParams:
    diff_sampling_param = (
        llm_engine.model_config.get_diff_sampling_param())
    if diff_sampling_param:
        return SamplingParams.from_optional(**diff_sampling_param)
    return SamplingParams()

def validate_and_add_requests(
    llm_engine,
    prompts: Union[PromptType, Sequence[PromptType]],
    params: Union[SamplingParams, Sequence[SamplingParams], PoolingParams,
                    Sequence[PoolingParams]],
    lora_request = None,
    prompt_adapter_request = None,
    priority = None,
) -> None:

    if isinstance(prompts, (str, dict)):
        # Convert a single prompt to a list.
        prompts = [prompts]
    request_counter = Counter()
    # Add requests to the engine.
    input_text = {}
    for i, prompt in enumerate(prompts):
        request_id = str(next(request_counter))
        llm_engine.add_request(
            request_id,
            prompt,
            params[i] if isinstance(params, Sequence) else params,
            lora_request=lora_request[i] if isinstance(
                lora_request, Sequence) else lora_request,
            prompt_adapter_request=prompt_adapter_request,
            priority=priority[i] if priority else 0,
        )
        input_text[request_id] = prompt
    return input_text

def qwen_chat_wrapper(question):
    return f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"

def llama_chat_wrapper(question):
    return f"<s>[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n{question}[/INST]"

def deepseek_chat_wrapper(question):
    res = f"<｜begin▁of▁sentence｜><｜User｜>{question}<｜Assistant｜>"
    return res

# group model by model structure
llama_models = ["llama2-7b", "llama3.1-8b", "llama3.1-70b"]
qwen_models = ["qwen2.5-32b", "qwen2.5-14b", "qwen2-7b","qwen2-72b", "qwen2-57b-a14b", "qwen3-32b", "qwen3-4b", "qwq", "qwen3-8b", "qwen3-235b-a22b"]
deepseek_models = ["deepseek_v2", "deepseek_v3"]
# decide the chat wrapper by model type
CHAT_WRAPPER = {model: llama_chat_wrapper for model in llama_models}
CHAT_WRAPPER.update({model: qwen_chat_wrapper for model in qwen_models})
CHAT_WRAPPER.update({model: deepseek_chat_wrapper for model in deepseek_models})

llama_wrapper_token = 27
qwen_wrapper_token = 19
deepseek_wrapper_token = 3
CHAT_WRAPPER_TOKEN_NUM = {model: llama_wrapper_token for model in llama_models}
CHAT_WRAPPER_TOKEN_NUM.update({model: qwen_wrapper_token for model in qwen_models})
CHAT_WRAPPER_TOKEN_NUM.update({model: deepseek_wrapper_token for model in deepseek_models})

def defatult_multi_batch_prompt(model_id, num_request, mode="chat"):
    if os.path.exists('questions.txt'):
        with open('questions.txt', 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f.readlines() if line.strip() and not line.startswith('#')]
        if len(questions) < num_request:
            logging.error(f'Size mismatch: The number of questions is {len(questions)}, which is smaller than {num_request}.')
        question_length = [int(line.strip()) for line in open("questions_length.txt")]
    else:
        questions = [
            "What is Deep Learning?",
            #"What is TPU?",
            # "Consider a scenario where an autonomous vehicle must make an ethical decision between two unavoidable outcomes: saving the passengers inside the vehicle or protecting pedestrians. Discuss how a large language model (LLM) could be integrated into the vehicle's decision-making system to assist with ethical considerations. What are the potential benefits and limitations of relying on an LLM for such critical decisions?",
        ] * num_request
        question_length = [5] * num_request

    if mode == "chat":
        questions = [CHAT_WRAPPER[args.model](question + ". " * max(0, CONTEXT_LEN - qlen - CHAT_WRAPPER_TOKEN_NUM[args.model])) for qlen, question in zip(question_length, questions)]
    elif mode == "generation":
        questions = [". " * max(0, CONTEXT_LEN - 5) + question for question in questions]
    PRINT_LEN = 128
    if CONTEXT_LEN > PRINT_LEN:
        questions_p = [question[:PRINT_LEN] + " ..." for question in questions]
        logger.info(f"Questions: {questions_p}")
    else:
        logger.info(f"Questions: {questions}")

    prompts = cast(Union[PromptType, Sequence[PromptType]],questions)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer:
        total_tokens = 0
        for prompt in questions:
            tokens = tokenizer.encode(prompt, add_special_tokens=False)
            total_tokens += len(tokens)

    return prompts, total_tokens

def test_whole_model(
    batches=1, model_id="llama", model_path="/data", quantize=None, mode="chat", use_v1=True, tp_size=2
):
    logger.info(f"Model id: {model_id}, Quantize: {quantize}, Batch: {batches}")
    enable_profile = soph_config_manager.get('ENABLE_PROFILE')
    book_keeping = soph_config_manager.get('PROFILE_BOOK_KEEPING')
    profile_starting_token = soph_config_manager.get('PROFILE_STARTING_TOKEN')
    max_record_num = int(1e6)
    llm_engine, prompts = default_llm_engine(model_id, quantize, model_path, use_v1, tp_size, batches)
    import torch_tpu

    # 初始化返回值变量 - Initialize return value variables
    FTL_ms = 0.0
    TPS = 0.0
    quality_results = []

    for it in range(2):
        if it > 0:
            time.sleep(2)
            torch_tpu.tpu.optimer_utils.OpTimer_reset()

        generated_tokens_len = cCounter()
        # sampling_params & prompts
        sampling_params = SamplingParams(max_tokens=DECODE_TOKEN_LEN, temperature=0, ignore_eos=True)
        if sampling_params is None:
            sampling_params = get_default_sampling_params(llm_engine)

        # Add requests to llm_engine Obj
        input_text = validate_and_add_requests(
            llm_engine=llm_engine,
            prompts=prompts,
            params=sampling_params,)

        previous_texts = {}
        generated_text = {}
        time_list = []
        step_counter = 0
        while llm_engine.has_unfinished_requests():
            if it > 0 and enable_profile and step_counter >= profile_starting_token:
                torch.ops.my_ops.enable_profile(max_record_num, book_keeping)
            os.environ["TOKEN_IDX"] = str(step_counter)

            generate_start = time.time_ns()
            generations = llm_engine.step()
            generate_end = time.time_ns()
            time_list.append(generate_end - generate_start)

            new_texts_log = []
            for generation in generations:
                request_id = generation.request_id
                current_text = generation.outputs[0].text
                prev_text = previous_texts.get(request_id, "")
                new_text = current_text[len(prev_text):]

                if request_id not in generated_text:
                    generated_text[request_id] = new_text
                else:
                    generated_text[request_id] += new_text

                # 统计生成的token数量 - Count generated tokens
                generated_tokens_len[int(request_id)] += 1

                previous_texts[request_id] = current_text
                new_texts_log.append(new_text)

            if new_texts_log:
                logger.info(f'Token {step_counter} {[text for text in new_texts_log]}')
            step_counter += 1

        # Generation finished，abort all request and reset KVcache
        if llm_engine.has_unfinished_requests():
            reqs_to_abort= [g.request_id for g in generations]
            llm_engine.abort_request(reqs_to_abort)
        llm_engine.reset_prefix_cache()

        if enable_profile and it > 0:
            torch_tpu.tpu.optimer_utils.OpTimer_dump()
            torch.ops.my_ops.disable_profile()

        TPS = batches / np.mean(time_list[1:]) * 1000**3
        FTL_ms = time_list[0] / 1000**2
        TTFT = time_list[0] / 1000**3
        TPOT = np.mean(time_list[1:]) / 1000**3
        Throughput = batches / TPOT * 1000**3
        for key in generated_text.keys():
            cleaned_text = generated_text[key].rstrip('\n')
            logger.info(f"rank: {RANK}, Batch {key}: {cleaned_text}\n")
        logger.warning(f"FTL: {FTL_ms:.1f}ms, TPS: {TPS:.1f}")
        logger.warning(f'TTFT: {time_list[0] /1000**3:.3f}s, TPOT: {np.mean(time_list[1:]) /1000**3:.3f}s, Throughput: {batches / np.mean(time_list[1:]) *1000**3:.1f}, TPS: {1/ np.mean(time_list[1:]) *1000**3:.1f}')
        logger.info(f"-----------------------------")

    # 质量检查处理 - Quality check processing
    quality_results = []  # 保持为空列表 - Keep as empty list

    del llm_engine
    input_tokens_len = {k: len(v) for k, v in input_text.items()}
    # 返回与test_model.py对齐的结果 - Return results aligned with test_model.py
    return input_text, generated_text, input_tokens_len, dict(generated_tokens_len), FTL_ms, TPS, quality_results

def collect_meta(args):
    from torch_tpu.utils.collect_env import collect_cpu_performance,pretty_version_info
    from torch_tpu.tpu.versions import versions
    import traceback
    
    try:
        torch_version_info = pretty_version_info()
    except Exception as e:
        logger.error(f"Failed to collect torch version info: {e}")
        torch_version_info = traceback.format_exc()

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cpu": collect_cpu_performance(),
        "tpu": versions(),
        "torch": torch_version_info,
        "sys_env": dict(os.environ),
        "vllm": {
            "v1": args.useV1,
            "exec_file": os.path.abspath(__file__),
        }
    }

if __name__ == "__main__":
    if args.save_json == 'auto':
        abs_dir = os.path.join('/record/vllm_results', datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        os.makedirs(abs_dir, exist_ok=True)
        assert os.access(abs_dir, os.W_OK), f"Permission denied: {abs_dir}"

    meta = collect_meta(args)
    (
        input_text,
        generated_text,
        real_input_len,
        real_output_len,
        FTL_ms,
        TPS,
        quality_results,
    ) = test_whole_model(
        batches=args.batch,
        model_id=args.model,
        quantize=args.quantize,
        model_path=args.path,
        mode=args.mode,
        use_v1 = args.useV1,
        tp_size = args.tp_size,
    )
    
    # 保存JSON结果 - Save JSON results
    if args.save_json:
        # 确保目录存在 - Ensure directory exists
        if args.save_json == 'auto':
            file_name = f"{args.model_id}_{args.dtype}_{args.batch}_{args.input_length}_{args.max_new_tokens}_{args.mode}_{RANK}.json"
            abs_dir = os.path.join('/record/vllm_results', datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
            os.makedirs(abs_dir, exist_ok=True)
            args.save_json = os.path.join(abs_dir, file_name)
        else:
            os.makedirs(os.path.dirname(os.path.abspath(args.save_json)), exist_ok=True)
        
        # 构建保存的数据结构 - Build data structure to save
        json_data = {
            "schema_version": "v2",
            "benchmark": {
                "model": args.model,
                "quantize": args.quantize,
                "batch_size": args.batch,
                "max_new_tokens": DECODE_TOKEN_LEN,
                "context_len": CONTEXT_LEN,
                "mode": args.mode,
                "tp_size": args.tp_size,
            },
            "meta": meta,
            "performance": {
                "FTL_ms": FTL_ms,
                "TPS": TPS,
            },
            "quality": quality_results,
            "data": {
                "text_input": input_text,
                "text_generated": generated_text,
                "real_input_len": real_input_len,
                "real_output_len": real_output_len,
            }
        }
        
        # 保存到文件，添加rank后缀 - Save to file with rank suffix
        output_file = f"{args.save_json}_{RANK}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                json_data,
                f,
                ensure_ascii=False,
                indent=4,
            )
        logger.info(f"Results saved to {output_file}")
