import os
from typing import List
#import pytest
from transformers import AutoTokenizer
# from tests.quantization.utils import is_quant_method_supported
from vllm import LLM, SamplingParams

from vllm.platforms import CpuArchEnum
import time

os.environ["TOKENIZERS_PARALLELISM"] = "true"

MAX_MODEL_LEN = 128

# MODELS = ["/workspace/data/data/Qwen2-7B"]
# MODELS = ["/datas/Qwen2-7B"]
# MODELS = ["/workspace/data/data/Qwen2.5-32B-Instruct"]
MODELS = ["/workspace/data/data/Qwen2-7B-Instruct-GPTQ-Int4"]


def load_example_prompts(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        # Read all lines from the file and strip extra spaces or newlines
        prompts = [line.strip() for line in f.readlines()]
    return prompts


def test_models(example_prompts, model_name) -> None:
    model = LLM(
        model=model_name,
        max_model_len=MAX_MODEL_LEN,
        #trust_remote_code=False,
        enforce_eager=True,
        quantization="gptq"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    formatted_prompts = [
        tokenizer.apply_chat_template([{
            "role": "user",
            "content": prompt
        }],
                                      tokenize=False,
                                      add_generation_prompt=True)
        for prompt in example_prompts
    ]
    params = SamplingParams(max_tokens=128, temperature=0)

    # for it in range(2):
    #     if it > 0:
    #         time.sleep(2)
    #     outputs = model.generate(formatted_prompts, params)
    # del model

    outputs = model.generate(formatted_prompts, params)
    del model

    # print(model_name, outputs)
    for i in range(len(example_prompts)):
        generated_str = outputs[i].outputs[0].text
        print(f"Test {i}:")
        print(f"example_prompts: {example_prompts[i]!r}")
        print(f"Generated: {generated_str!r}")
        print("-" * 50) 


if __name__ == "__main__":

    #file_path = "tests/prompts/example.txt"
    #example_prompts = load_example_prompts(file_path)
 
    example_prompts = [
        "What is Deep Learning?",
        "What is TPU?",
        "How does climate change impact species migration patterns and food web stability in Arctic ecosystems?",
    ]

    # Run the tests
    for model_name in MODELS:
        test_models(example_prompts, model_name)

    
