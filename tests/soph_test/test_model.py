import torch
import torch.nn as nn
import torch.nn.functional as F
import os, time
import numpy as np
import math
from datetime import datetime
import sys
import argparse
from transformers import AutoTokenizer
from transformers.configuration_utils import PretrainedConfig

from vllm.engine import llm_engine
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
from vllm.sophtpu_utils import LLMQualityChecker

logger = init_logger(__name__)

RANK = int(os.environ.get("LOCAL_RANK", "0"))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', '1'))
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "6000"

soph_config_manager = get_soph_config_manager()
MAX_IMG_TOKEN = soph_config_manager.get('MAX_IMG_TOKEN')
ENABLE_PROFILE = soph_config_manager.get('ENABLE_PROFILE')
BOOK_KEEPING = soph_config_manager.get('PROFILE_BOOK_KEEPING')

log_file = f"{os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))}/run_local_{RANK}.log"
if os.path.exists(log_file):
    os.remove(log_file)
# logger.add(log_file, level=soph_config_manager.get('SLOG_LEVEL'))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="The path of the model to load.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default='auto',
        choices=["float16", "bfloat16"],
        help="The dtype to be forced upon the model.",
    )
    parser.add_argument("--batch", type=int, default=16, help="Batch size for testing")
    parser.add_argument(
        "--input-length",
        type=int,
        default=128,
        help="If specified, input tokens will be padded to this length",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["chat", "generate"],
        default="generate",
        help="Inference mode, chat or generate, defaults to chat.",
    )

    parser.add_argument(
        "--save-results",
        action='store_true',
        help="save performance results in csv file."
    )

    parser.add_argument(
        "--quality-check",
        action='store_true',
        help="Quality check of Generated text. API_KEY is required for this mode.",
    )

    parser.add_argument(
        "--save-json",
        type=str,
        help="Save outputs to specified file in json format.",
    )

    parser.add_argument(
        "--useV1", 
        type=bool, 
        default=True, 
        help="Use vLLM V1. Set False to use V0.")
    
    parser.add_argument(
        "--tp_size", 
        type=int, 
        required=True,
        help="Tensor Parallel size. Set to 1 to disable tensor parallel.")

    args = parser.parse_args()

    if args.mode == "chat" and "--input-length" in sys.argv:
        logger.warning("input tokens will only be truncated but not be padded to input-length when --mode is chat.")
    if args.quality_check and not os.getenv('API_KEY'):
        parser.error("API_KEY must be provide for --quality-check.")

    return args

def apply_chat_template(question):
    messages = [
        {"role": "system", "content": "You are a helpful assistant.",},
        {"role": "user", "content": f"{question}"},
    ]
    return messages

def default_llm_engine(model_id, dtype, use_v1, max_new_tokens, input_length, tp_size):
    engine_args = EngineArgs(
        model=model_id,
        dtype=dtype,
        max_model_len=max_new_tokens + input_length,
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
    return llm_engine

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
    chat_mode = True,
    input_length = 128,
    lora_request = None,
    prompt_adapter_request = None,
    priority = None,
) -> None:
    if isinstance(prompts, (str, dict)):
        # Convert a single prompt to a list.
        prompts = [prompts]

    tokenizer = llm_engine.tokenizer.tokenizer

    if llm_engine.tokenizer.tokenizer.chat_template is None:
        # Define a simple chat template for models like LLaMA.
        llama_template = """{% for message in messages %}{% if message['role'] == 'user' %}<s>[INST] {{ message['content'] }} [/INST]{% elif message['role'] == 'assistant' %}{{ message['content'] }}</s>{% endif %}{% endfor %}"""
        tokenizer.chat_template = llama_template
    if chat_mode:
        # Apply chat template if in chat mode.
        prompts = [tokenizer.apply_chat_template(apply_chat_template(text), tokenize=False) for text in prompts]

    # Truncate prompts
    tokenizer.truncation_side = "left" if chat_mode else "right"
    truncated_prompts = []
    for prompt in prompts:
        tokens = tokenizer.encode(prompt, max_length=input_length, truncation=True)
        truncated_text = tokenizer.decode(tokens, skip_special_tokens=True)
        truncated_prompts.append(truncated_text)

    request_counter = Counter()
    # Add requests to the engine.
    input_text = {}
    for i, prompt in enumerate(truncated_prompts):
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

def gen_test_case(batch_size, chat: False):
    questions = []
    if chat:
        questions_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'questions.txt')
        if os.path.exists(questions_file):
            with open(questions_file, 'r', encoding='utf-8') as f:
                questions = [line.strip() for line in f.readlines() if line.strip() and not line.startswith('#')]
            if len(questions) < batch_size:
                logger.error(f'Size mismatch: The number of questions is {len(questions)}, which is smaller than {batch_size}.')
            questions = questions[:batch_size]
        else:
            questions = [
                "What is Deep Learning?",
            ] * batch_size
    else:
        prompt_pth = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset', "wikipedia/llm_introduction.txt")
        with open(prompt_pth, 'r', encoding='utf-8') as f:
            questions = [f.read()] * batch_size

    return questions

def gen_vlm_test_batch(
    model, batch_size, input_length, max_new_tokens, chat_mode
):
    pass

def test_whole_model(
    model_id, dtype, batch, input_length, max_new_tokens, chat_mode:bool = False, quality_check:bool = False,
    use_v1:bool = True, tp_size:int=1
):  
    prof = None
    profile_starting_token = soph_config_manager.get("PROFILE_STARTING_TOKEN")
    soph_config_manager.config['CURRENT_BATCH_SIZE'] = batch

    llm_engine = default_llm_engine(
        model_id=model_id, dtype=dtype, use_v1=use_v1, 
        max_new_tokens=max_new_tokens, input_length=input_length, tp_size=tp_size
    )
    prompts = gen_test_case(batch, chat_mode)

    for repeat_i in range(2):
        logger.warning(f'================================ Run iter: {repeat_i + 1} ================================')
        
        # sampling_params & prompts
        sampling_params = SamplingParams(max_tokens=max_new_tokens, temperature=0, ignore_eos=True)
        if sampling_params is None:
            sampling_params = get_default_sampling_params(llm_engine)

        # Add requests to llm_engine Obj
        input_text = validate_and_add_requests(
            llm_engine=llm_engine,
            prompts=prompts,
            params=sampling_params,
            chat_mode=chat_mode,
            input_length=input_length)

        input_tokens_len = [len(b) for b in input_text.values()]

        previous_texts = {}
        generated_text = {}
        time_list = []
        generated_tokens_len = [0 for _ in range(batch)]
        step_counter = 0
        while llm_engine.has_unfinished_requests():
            if repeat_i > 0 and ENABLE_PROFILE and step_counter >= profile_starting_token:
                torch.ops.my_ops.enable_profile(int(1e6), BOOK_KEEPING)
            os.environ["TOKEN_IDX"] = str(step_counter)
            
            generate_start = time.time_ns()
            generations = llm_engine.step()
            generate_end = time.time_ns()
            time_list.append(generate_end - generate_start)

            new_texts_log = []
            for generation in generations:
                request_id = int(generation.request_id)
                current_text = generation.outputs[0].text
                prev_text = previous_texts.get(request_id, "")
                new_text = current_text[len(prev_text):]

                if request_id not in generated_text:
                    generated_text[request_id] = new_text
                else:
                    generated_text[request_id] += new_text
                generated_tokens_len[request_id] += 1

                previous_texts[request_id] = current_text
                new_texts_log.append(new_text)

            if new_texts_log:
                logger.info(f'Token {step_counter} {[text for text in new_texts_log]}')
            step_counter += 1
            
        if llm_engine.has_unfinished_requests():
            reqs_to_abort= [g.request_id for g in generations]
            llm_engine.abort_request(reqs_to_abort)
        llm_engine.reset_prefix_cache()

        logger.info(f'Real input text: {input_text}')
        for key in generated_text.keys():
            cleaned_text = generated_text[key].rstrip('\n')
            logger.info(f"rank: {RANK}, Batch {key}: {cleaned_text}\n")

    FTL_ms = time_list[0] / 1000**2
    TPS = batch / np.mean(time_list[1:]) * 1000**3

    logger.warning(
        f"rank: {RANK}, Real input length: {input_tokens_len}, real output length: {generated_tokens_len}"
    )
    logger.warning(
        f"rank: {RANK}, FTL: {FTL_ms:.1f}ms, TPS: {TPS:.1f}"
    )
    if repeat_i > 0 and ENABLE_PROFILE:
        torch.ops.my_ops.disable_profile()
    if prof is not None:
        prof.stop()
        prof.export_chrome_trace("profile.trace.json")
    logger.info(f"-----------------------------")

    quality_results = []
    if RANK == 0:
        print(f'========================== Test Summary ==========================')
        print(f'FTL: {FTL_ms:.1f} ms')
        print(f'TPS: {TPS:.1f}')
        if quality_check:
            quality_checker = LLMQualityChecker()
            print(f'------------ Quality Check -------------')
            for i, (prompt, generated) in enumerate(zip(input_text.values(), generated_text.values())):
                if prompt not in list(input_text.values())[:i] or 1:
                    res = quality_checker.check_quality(prompt=prompt, generated=generated)
                    quality_results.append(res)
                    print(f'### Quality Check for Batch {i}: \n{res}')
        print(f'==================================================================')

    return input_text, generated_text, input_tokens_len, generated_tokens_len, FTL_ms, TPS, quality_results

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
    args = parse_args()

    # auto 字段处理 - Auto field handling
    if args.save_json == 'auto':
        abs_dir = os.path.join('/record/vllm_results', datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        os.makedirs(abs_dir, exist_ok=True)
        assert os.access(abs_dir, os.W_OK), f"Permission denied: {abs_dir}"

    meta = collect_meta(args)  # 收集环境信息 - Collect environment info
    logger.warning(f"ARGS: {args}")
    (
        input_text,
        generated_text,
        real_input_len,
        real_output_len,
        FTL_ms,
        TPS,
        quality_results,
    ) = test_whole_model(
        model_id=args.model_id,
        dtype=args.dtype,
        batch=args.batch,
        input_length=args.input_length,
        max_new_tokens=args.max_new_tokens,
        chat_mode=args.mode == "chat",
        quality_check=args.quality_check,
        use_v1=args.useV1,
        tp_size=args.tp_size
    )
    # 保存JSON结果 - Save JSON results
    if args.save_json:
        import json
        # 确保目录存在 - Ensure directory exists
        if args.save_json == 'auto':
            file_name = f"{args.model_id}_{args.dtype}_{args.batch}_{args.input_length}_{args.max_new_tokens}_{args.mode}_{RANK}.json"
            abs_dir = os.path.join('/record/vllm_results', datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
            os.makedirs(abs_dir, exist_ok=True)
            args.save_json = os.path.join(abs_dir, file_name)
        else:
            os.makedirs(os.path.dirname(os.path.abspath(args.save_json)), exist_ok=True)

        json_data = {
            "schema_version": "v2",
            "benchmark": {
                "model": args.model_id,
                # "quantize": args.quantize,
                "batch_size": args.batch,
                "max_new_tokens": args.max_new_tokens,
                "context_len": args.input_length,
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
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(
                json_data,
                f,
                ensure_ascii=False,
                indent=4,
            )
        logger.info(f"Results saved to {args.save_json}")

    if args.save_results:
        import pandas as pd
        df = pd.DataFrame(
            [
                {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "model_id": args.model_id,
                    "dtype": args.dtype,
                    "batch_size": args.batch,
                    "input_length": args.input_length,
                    "max_new_tokens": args.max_new_tokens,
                    "mode": args.mode,
                    "text_input": input_text,
                    "text_generated": generated_text,
                    "real_input_len": real_input_len,
                    "real_output_len": real_output_len,
                    "TP": WORLD_SIZE,
                    "FTL_ms": FTL_ms,
                    "TPS": TPS,
                    "quality": quality_results,
                }
            ]
        )
        output_file = f"results_{args.tp_size}.csv"
        df.to_csv(
            output_file, mode="a", header=not os.path.exists(output_file), index=False
        )
        logger.info(f"Results saved to {output_file}")

# python test_model.py --model-id /data/llama-2-7b-chat-hf/ --max-new-tokens=20 --mode chat --tp_size 2
# python test_model.py --model-id /data/llava-v1.6-vicuna-7b/ --max-new-tokens 20  --mode generate --tp_size 2
# CHIP_MAP=0,1 torchrun --nproc_per_node 2 --nnodes 1 test_model.py --model-id /data/llava-v1.6-vicuna-7b/ --max-new-tokens 20 --mode generate --tp_size 2

"""
帮助信息:
此脚本用于测试SOPH-vLLM内LLM推理性能，可使用以下参数指定模型路径、数据类型、批大小、及推理模式。

参数说明:
--model-id: 必需参数，指定要加载的模型的路径。
--dtype: 可选参数，指定强制加载的模型的数据类型，可选值为 float16 和 bfloat16, 默认按模型本身配置数据类型加载。
--batch: 可选参数，默认构造单测批次为1。
--input-length: 可选参数，若指定，输入token将被填充为该长度。
--max-new-tokens: 可选参数，指定要生成的最大token数，默认值为 128。
--mode: 可选参数，指定运行模式，可选值为 chat 和 generate，默认值为 generate。
--quality-check: 可选参数，模型生成文本质量检查，如果设置该参数则需要提供SophNet API_KEY。
--save-json: 可选参数，指定该参数后，测试结果将保存至 JSON 文件中。
--save-results: 可选参数，指定该参数后，测试结果将保存至 results_{RANK}.csv 文件中。
--useV1: 可选参数，指定是否使用 vLLM V1 版本，默认为 True。
--tp_size: 必须参数，指定张量并行大小。

"""
