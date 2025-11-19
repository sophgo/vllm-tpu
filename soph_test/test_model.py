import torch
import torch.nn as nn
import torch.nn.functional as F
import os, time, json
import numpy as np
import math
from datetime import datetime
import sys
import argparse
import random
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, asdict
from abc import abstractmethod

from transformers import AutoTokenizer
from transformers.configuration_utils import PretrainedConfig

from vllm.engine import llm_engine
from vllm.logger import init_logger
from vllm.engine.llm_engine import LLMEngine
from vllm.usage.usage_lib import UsageContext
from typing import Sequence, Union, cast
from vllm.engine.arg_utils import EngineArgs
from vllm.sampling_params import SamplingParams
from vllm.inputs import PromptType
from vllm.pooling_params import PoolingParams
from vllm.utils import Counter

# from vllm.sophtpu_utils import LLMQualityChecker
# from vllm.platforms.sophtpu import get_soph_config_manager

from vllm_sophon.hack.soph_utils import LLMQualityChecker
from vllm_sophon.platform import get_soph_config_manager

logger = init_logger(__name__)

RANK = int(os.environ.get("LOCAL_RANK", "0"))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "6000"

soph_config_manager = get_soph_config_manager()
MAX_IMG_TOKEN = soph_config_manager.get("MAX_IMG_TOKEN")
ENABLE_PROFILE = soph_config_manager.get("ENABLE_PROFILE")
BOOK_KEEPING = soph_config_manager.get("PROFILE_BOOK_KEEPING")
DECODE_TOKEN_LEN = soph_config_manager.get("DECODE_TOKEN_LEN")

log_file = f"{os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))}/run_local_{RANK}.log"
if os.path.exists(log_file):
    os.remove(log_file)
# logger.add(log_file, level=soph_config_manager.get('SLOG_LEVEL'))

CHAT_DATASET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "dataset", "questions.txt"
)
GENERATE_DATASET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "dataset",
    "AndersenFairyTales",
    "Andersen_Fairy_Tales.txt",
)


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
        default="auto",
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
        help="Inference mode, chat or generate, defaults to generate.",
    )

    parser.add_argument(
        "--save-results",
        action="store_true",
        help="save performance results in csv file.",
    )

    parser.add_argument(
        "--quality-check",
        action="store_true",
        help="Quality check of Generated text. API_KEY is required for this mode.",
    )

    parser.add_argument(
        "--save-json",
        type=str,
        help="Save outputs to specified file in json format.",
    )

    parser.add_argument(
        "--useV1", type=bool, default=True, help="Use vLLM V1. Set False to use V0."
    )

    parser.add_argument(
        "--tp_size",
        type=int,
        required=True,
        help="Tensor Parallel size. Set to 1 to disable tensor parallel.",
    )

    parser.add_argument(
        "--image-size",
        type=str,
        default=None,
        help="Resize input image to specified size (e.g., 224x224, 448x640). If not set, use original image size.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1000,
        help="random seed for dataset selection.",
    )

    args = parser.parse_args()

    if args.mode == "chat" and "--input-length" in sys.argv:
        logger.warning(
            "input tokens will only be truncated but not be padded to input-length when --mode is chat."
        )
    if args.quality_check and not os.getenv("API_KEY"):
        parser.error("API_KEY must be provide for --quality-check.")

    return args


def collect_meta():
    """
    收集环境信息，包括CPU性能、TPU版本、torch版本和系统环境
    Collect environment information including CPU performance, TPU versions, torch version and system environment
    """
    from uuid import uuid4
    import traceback
    import torch_tpu
    import subprocess
    from torch_tpu.utils.collect_env import (
        get_pretty_env_info,
        pretty_version_info,
        collect_cpu_performance,
    )
    from torch_tpu.tpu.versions import versions

    try:
        torch_version_info = pretty_version_info()
    except Exception as e:
        logger.error(f"Failed to collect torch version info: {e}")
        torch_version_info = traceback.format_exc()

    try:
        # get SN number
        driver = subprocess.check_output(
            "/opt/tpuv7/tpuv7-current/bin/test_tpuv7_manager 0 1", shell=True
        ).decode("utf-8")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to collect driver version info: {e}")
        driver = traceback.format_exc() + str(e.stdout) + str(e.stderr)

    return Meta(
        uuid=str(uuid4()),
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        cpu=collect_cpu_performance(),
        tpu=versions(),
        torch=torch_version_info,
        sys_env=dict(os.environ),
        driver=driver,
        vllm={
            "v1": os.getenv("VLLM_USE_V1", 0),
            "exec_file": os.path.abspath(__file__),
        },
    )


class ReqMode(Enum):
    GENERATE = 0
    CHAT = 1


@dataclass
class Meta:
    uuid: str
    timestamp: str
    cpu: str
    tpu: str
    torch: str
    sys_env: dict
    vllm: dict
    driver: str


@dataclass
class CaseResult:
    @dataclass
    class Benchmark:
        model_id: str
        dtype: str
        batch_size: int
        input_length: int
        max_new_tokens: int
        mode: str
        world_size: int

    @dataclass
    class Performance:
        FTL_ms: float
        TPS: float

    @dataclass
    class Quality:
        score: float = -1.0

    @dataclass
    class Texts:
        text_input: str
        text_generated: str
        real_input_len: int
        real_output_len: int
        text_generated_ids: list

    benchmark: Benchmark
    performance: Performance
    quality: Quality
    inference_data: Texts
    start_time: str
    end_time: str

    def save_csv(self, path):
        import pandas as pd

        df = pd.DataFrame(
            [
                {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "model_id": self.benchmark.model_id,
                    "dtype": self.benchmark.dtype,
                    "batch_size": self.benchmark.batch_size,
                    "input_length": self.benchmark.input_length,
                    "max_new_tokens": self.benchmark.max_new_tokens,
                    "mode": self.benchmark.mode,
                    "text_input": self.inference_data.text_input,
                    "text_generated": self.inference_data.text_generated,
                    "real_input_len": self.inference_data.real_input_len,
                    "real_output_len": self.inference_data.real_output_len,
                    "TP": self.benchmark.world_size,
                    "FTL_ms": self.performance.FTL_ms,
                    "TPS": self.performance.TPS,
                    "quality": self.quality.score,
                }
            ]
        )
        df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
        logger.info(f"Results saved to {path}")

    def save_json(self, file_name, detail=False):
        if detail:
            json_data = {
                "schema_version": "v2",
                "benchmark": asdict(self.benchmark),
                "meta": asdict(collect_meta()),
                "performance": asdict(self.performance),
                "quality": asdict(self.quality),
                "data": asdict(self.inference_data),
            }
        else:
            json_data = {
                "FTL_ms": self.performance,
                "TPS": self.performance.TPS,
                "text_input": self.inference_data.text_input,
                "text_generated": self.inference_data.text_generated,
                "real_input_len": self.inference_data.real_input_len,
                "real_output_len": self.inference_data.real_output_len,
                "text_generated_ids": self.inference_data.text_generated_ids,
                "quality": self.quality.score,
                "meta": asdict(collect_meta()),
            }
        Path(file_name).parent.mkdir(parents=True, exist_ok=True)
        assert os.access(
            Path(file_name).parent, os.W_OK
        ), f"Permission denied: {Path(file_name).parent}"
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=4, ensure_ascii=False)
        logger.info(f"Results saved to {file_name}")


class RequestGenerator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @abstractmethod
    def __gen_test_questions(self, batch_size, chat=False):
        pass

    @abstractmethod
    def _truncate_prompts(self):
        pass


class RequestLM(RequestGenerator):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)

    def _truncate_prompts(self, prompts, truncated_length):
        truncated_prompts = []
        truncated_lengths = []
        for prompt in prompts:
            tokens = self.tokenizer.encode(
                prompt, max_length=truncated_length, truncation=True
            )
            truncated_lengths.append(len(tokens))
            truncated_text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            truncated_prompts.append(truncated_text)
        return truncated_prompts, truncated_lengths


class RequestLMGenerate(RequestLM):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.tokenizer.truncation_side = "right"

    def __gen_test_questions(self, batch_size, input_length):
        questions = ["" for _ in range(batch_size)]
        from dataset.AndersenFairyTales.AndersenFairyTales import DatasetAndersen

        dataset = DatasetAndersen(GENERATE_DATASET_PATH)
        contents = dataset.get_contents()
        for b_idx in range(batch_size):
            chas_select = random.choices(contents, k=len(contents))
            for cha in chas_select:
                questions[b_idx] += dataset.get_chapter(cha)
                if (
                    self.tokenizer(
                        questions[b_idx],
                        padding=False,
                        truncation=False,
                        return_tensors="pt",
                    ).input_ids.shape[1]
                    >= input_length
                ):
                    break
        return questions

    def gen_test_prompts(self, batch_size, input_length):
        test_text = self.__gen_test_questions(batch_size, input_length)
        truncated_text, truncated_lengths = self._truncate_prompts(
            test_text, input_length
        )
        return truncated_text, truncated_lengths


class RequestLMChat(RequestLM):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.tokenizer.truncation_side = "left"

    def __gen_test_questions(self, batch_size):
        questions = []
        with open(CHAT_DATASET_PATH, "r", encoding="utf-8") as f:
            dataset_questions = [
                line.strip()
                for line in f.readlines()
                if line.strip() and not line.startswith("#")
            ]
            questions = (
                dataset_questions * (batch_size // len(dataset_questions))
                + dataset_questions[: batch_size % len(dataset_questions)]
            )
        return questions

    def gen_test_prompts(self, batch_size, input_length):
        test_text = self.__gen_test_questions(batch_size)
        test_text = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for text in test_text
        ]
        truncated_text, truncated_length = self._truncate_prompts(
            test_text, input_length
        )
        return truncated_text, truncated_length


class RequestVLMChat(RequestGenerator):
    def __init__(self, tokenizer, model_arch):
        super().__init__(tokenizer)
        self.tokenizer.truncation_side = "left"
        self.model_arch = model_arch

    def __gen_test_questions(self, batch_size, image_size):
        def prepare_image(img_pth):
            if not os.path.exists(img_pth):
                try:
                    import requests as rq

                    os.makedirs(os.path.dirname(img_pth), exist_ok=True)
                    response = rq.get(
                        "https://raw.githubusercontent.com/huggingface/text-generation-inference/main/integration-tests/images/chicken_on_money.png"
                    )
                    response.raise_for_status()
                    with open(img_pth, "wb") as f:
                        f.write(response.content)
                    logger.info(
                        f"Input image does not exist, automatically downloading test image and saved to {img_pth}"
                    )
                except Exception as e:
                    logger.error(f"Failed to download image: {e}")

        def get_pic(img_pth, image_size):
            from PIL import Image

            image = Image.open(img_pth)
            if image_size:
                try:
                    width, height = map(int, image_size.split("x"))
                    image = image.resize((width, height), Image.Resampling.LANCZOS)
                    logger.info(f"Image resized to {width}x{height}")
                except ValueError:
                    logger.warning(
                        f"Invalid image size format: {image_size}. Using original image size."
                    )
            return image

        image_path = "dataset/images/chicken_on_money.png"
        prepare_image(image_path)
        question = "What is shown in this image?"
        prompt_header = "A chat between a curious human and an artificial intelligence assistant. \
                        The assistant gives helpful, detailed, and polite answers to the human's questions. USER: ![]("
        image = get_pic(image_path, image_size)
        if self.model_arch == [
            "Qwen2_5_VLForConditionalGeneration"
        ] or self.model_arch == ["Qwen2VLForConditionalGeneration"]:
            question = "<|image_pad|>What is shown in this image?ASSISTANT:"
        else:
            question = "<image>What is shown in this image?ASSISTANT:"
        input_str = f"{prompt_header}{question}"
        prompts = [
            {
                "prompt": input_str,
                "multi_modal_data": {"image": image},
            }
            for _ in range(batch_size)
        ]

        prompts = cast(Union[PromptType, Sequence[PromptType]], prompts)
        return prompts

    def gen_test_prompts(self, batch_size, image_size=None):
        prompts = self.__gen_test_questions(batch_size, image_size)
        prompt_token_len = [
            len(self.tokenizer.encode(prompt["prompt"])) + MAX_IMG_TOKEN
            for prompt in prompts
        ]
        return prompts, prompt_token_len


class TestModelRunner:
    def __init__(
        self,
        model_id,
        dtype,
        tp,
        max_model_len,
        max_num_seqs,
        max_num_batched_tokens,
        use_v1=True,
    ):
        self.model_id = model_id
        self.dtype = dtype
        self.tp = tp
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        config_dict, _ = PretrainedConfig.get_config_dict(model_id)
        model_type = config_dict.get("model_type", None)
        self.is_multi_modal = model_type in {"llava_next", "qwen2_vl", "qwen2_5_vl"}
        self.use_v1 = use_v1

        self.engine = self.__init_engine()

    def __init_engine(self):
        max_model_leng = self.max_model_len + self.is_multi_modal * MAX_IMG_TOKEN

        tpu_graph_enabled = os.environ.get("PYTORCH_TPU_ALLOCATOR")
        engine_args = EngineArgs(
            model=self.model_id,
            dtype=self.dtype,
            max_model_len=max_model_leng,
            enforce_eager=False if tpu_graph_enabled else True,
            trust_remote_code=True,
            tensor_parallel_size=self.tp,
            distributed_executor_backend=None if self.tp == 1 else "mp",
            max_num_seqs=self.max_num_seqs,
            max_num_batched_tokens=max(self.max_num_batched_tokens, 8192),
        )
        if self.use_v1:
            os.environ["VLLM_USE_V1"] = "1"
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
            from vllm.v1.engine.llm_engine import LLMEngine as V1LLMEngine

            engine_class = V1LLMEngine
        else:
            engine_class = LLMEngine

        return engine_class.from_engine_args(
            engine_args, usage_context=UsageContext.LLM_CLASS
        )

    def run_test_case(
        self,
        batch_size,
        input_length,
        max_new_tokens,
        mode=ReqMode.GENERATE,
        image_size=None,
    ) -> CaseResult:
        prof = None
        profile_starting_token = soph_config_manager.get("PROFILE_STARTING_TOKEN")
        start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.error(
            f"Running test case: batch_size={batch_size}, input_length={input_length}, max_new_tokens={max_new_tokens}, mode={mode}"
        )
        if self.is_multi_modal:
            prompts, input_tokens_len = RequestVLMChat(
                self.engine.tokenizer,
                getattr(self.engine.model_config.hf_config, "architectures", []),
            ).gen_test_prompts(batch_size, image_size)
        elif mode == ReqMode.CHAT:
            prompts, input_tokens_len = RequestLMChat(
                self.engine.tokenizer
            ).gen_test_prompts(batch_size, input_length)
        else:
            prompts, input_tokens_len = RequestLMGenerate(
                self.engine.tokenizer
            ).gen_test_prompts(batch_size, input_length)

        tpu_graph_enabled = os.environ.get("PYTORCH_TPU_ALLOCATOR")
        repeat_iter = 1 if tpu_graph_enabled else 2
        for repeat_i in range(repeat_iter):
            if repeat_iter == 2:
                logger.warning(
                    f'================ Run iter: {repeat_i if repeat_i else "Warmup"} ================'
                )
            # sampling_params & prompts
            sampling_params = SamplingParams(
                max_tokens=max_new_tokens, temperature=0, ignore_eos=True
            )
            lora_request = None
            #prompt_adapter_request = None
            priority = None

            request_counter = Counter()
            input_text = {}
            for i, prompt in enumerate(prompts):
                request_id = str(next(request_counter))
                self.engine.add_request(
                    request_id,
                    prompt,
                    (
                        sampling_params[i]
                        if isinstance(sampling_params, Sequence)
                        else sampling_params
                    ),
                    lora_request=(
                        lora_request[i]
                        if isinstance(lora_request, Sequence)
                        else lora_request
                    ),
                    #prompt_adapter_request=prompt_adapter_request,
                    priority=priority[i] if priority else 0,
                )
                input_text[request_id] = prompt

            previous_texts = {}
            generated_text = {}
            generated_ids = {}
            time_list = []
            generated_tokens_len = [0 for _ in range(batch_size)]
            step_counter = 0
            while self.engine.has_unfinished_requests():
                if (
                    repeat_i > 0
                    and ENABLE_PROFILE
                    and step_counter >= profile_starting_token
                ):
                    torch.ops.my_ops.enable_profile(int(1e6), BOOK_KEEPING)
                os.environ["TOKEN_IDX"] = str(step_counter)

                generate_start = time.time_ns()
                generations = self.engine.step()
                generate_end = time.time_ns()
                time_list.append(generate_end - generate_start)

                new_texts_log = []
                for generation in generations:
                    request_id = int(generation.request_id)
                    current_text = generation.outputs[0].text
                    current_ids = generation.outputs[0].token_ids
                    prev_text = previous_texts.get(request_id, "")
                    new_text = current_text[len(prev_text) :]
                    new_id = current_ids[len(prev_text) :]

                    if request_id not in generated_text:
                        generated_text[request_id] = new_text
                        generated_ids[request_id] = new_id
                    else:
                        generated_text[request_id] += new_text
                        generated_ids[request_id] += new_id
                    generated_tokens_len[request_id] += 1

                    previous_texts[request_id] = current_text
                    new_texts_log.append(new_text)

                if new_texts_log:
                    logger.info(
                        f"Token {step_counter} {[text for text in new_texts_log]}"
                    )
                step_counter += 1

            if self.engine.has_unfinished_requests():
                reqs_to_abort = [g.request_id for g in generations]
                self.engine.abort_request(reqs_to_abort)
            self.engine.reset_prefix_cache()

            logger.info(f"Real input text: {input_text}")
            for key in generated_text.keys():
                cleaned_text = generated_text[key].rstrip("\n")
                logger.info(f"rank: {RANK}, Batch {key}: {cleaned_text}\n")

        FTL_ms = time_list[0] / 1000**2
        TPS = batch_size / np.mean(time_list[1:]) * 1000**3

        logger.warning(
            f"rank: {RANK}, Real input length: {input_tokens_len}, real output length: {generated_tokens_len}"
        )
        logger.warning(f"rank: {RANK}, FTL: {FTL_ms:.1f}ms, TPS: {TPS:.1f}")
        if repeat_i > 0 and ENABLE_PROFILE:
            torch.ops.my_ops.disable_profile()
        if prof is not None:
            prof.stop()
            prof.export_chrome_trace("profile.trace.json")

        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        case_result = CaseResult(
            benchmark=CaseResult.Benchmark(
                model_id=self.model_id,
                dtype=str(self.dtype),
                batch_size=batch_size,
                input_length=input_length,
                max_new_tokens=max_new_tokens,
                mode=mode.name,
                world_size=self.tp,
            ),
            performance=CaseResult.Performance(
                FTL_ms=FTL_ms,
                TPS=TPS,
            ),
            quality=CaseResult.Quality(),
            inference_data=CaseResult.Texts(
                text_input=str(prompts),
                text_generated=dict(generated_text),
                real_input_len=input_tokens_len,
                real_output_len=generated_tokens_len,
                text_generated_ids=dict(generated_ids),
            ),
            start_time=start_time,
            end_time=end_time,
        )

        return case_result


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)

    runner = TestModelRunner(
        args.model_id,
        args.dtype,
        args.tp_size,
        args.input_length + args.max_new_tokens,
        args.batch,
        args.batch * args.input_length,
    )

    mode = ReqMode[args.mode.upper()]

    result = runner.run_test_case(
        args.batch, args.input_length, args.max_new_tokens, mode
    )

    if args.save_json:
        abs_dir = os.path.dirname(os.path.abspath(args.save_json))
        os.makedirs(abs_dir, exist_ok=True)

        json_data = {
            "schema_version": "v2",
            "benchmark": asdict(result.benchmark),
            "meta": asdict(collect_meta()),
            "performance": asdict(result.performance),
            "quality": asdict(result.quality),
            "data": {
                "text_input": result.inference_data.text_input,
                "text_generated": result.inference_data.text_generated,
                "real_input_len": result.inference_data.real_input_len,
                "real_output_len": result.inference_data.real_output_len,
                "text_generated_ids": result.inference_data.text_generated_ids,
            },
        }

        output_file = os.path.join(abs_dir, args.save_json)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)
        logger.info(f"Results saved to {output_file}")
    if args.save_results:
        result.save_csv(f"results_{args.tp_size}.csv")
