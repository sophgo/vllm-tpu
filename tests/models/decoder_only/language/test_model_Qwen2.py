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
MODELS = ["/datas/Qwen2-7B"]

EXPECTED_STRS_MAP = {
    "/workspace/data/data/Qwen2-7B": [
        "VLLM (Very Large Language Model Serving) is indeed a high-throughput and memory-efficient inference",
        'The development of artificial intelligence (AI) has been a journey marked by significant milestones, with major advancements ',
        'Artificial intelligence (AI) and human intelligence both process information, but they do so in fundamentally different',
        'A neural network is a computational model inspired by the structure and function of the human brain. It is',
        'In the vast, silent halls of the Cybernetic City, there lived a robot named Zephy',
        'The COVID-19 pandemic has had a profound and multifaceted impact on global economic structures and',
        'The Mona Lisa, painted by Leonardo da Vinci in the early 16th century, is one',
        'Sure, here are the translations of the phrase "The early bird catches the worm" into Japanese,'
    ]
}

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
    generations: List[str] = []
    for prompt in formatted_prompts:
        for it in range(2):
            if it > 0:
                time.sleep(2)
            outputs = model.generate(prompt, params)
            if it > 0:
                generations.append(outputs[0].outputs[0].text)
    del model

#    generations: List[str] = []
#    for prompt in formatted_prompts:
#        for it in range(2):
#            if it > 0:
#                import time
#                time.sleep(2)
#            outputs = model.generate(prompt, params)
#            if it > 0:
#                generations.append(outputs[0].outputs[0].text)
#    del model

    print(model_name, generations)
    # expected_strs = EXPECTED_STRS_MAP[model_name]
    for i in range(len(example_prompts)):
        generated_str = generations[i]
        # expected_str = expected_strs[i]
        # assert expected_str == generated_str, (
        #     f"Test{i}:\nExpected: {expected_str!r}\nvLLM: {generated_str!r}")
        print(f"Test {i}:")
        print(f"example_prompts: {example_prompts[i]!r}")
        print(f"Generated: {generated_str!r}")
        print("-" * 50) 


if __name__ == "__main__":

    #file_path = "tests/prompts/example.txt"
    #example_prompts = load_example_prompts(file_path)
 
    example_prompts = ['what is deep learning?']

    # Run the tests
    for model_name in MODELS:
        test_models(example_prompts, model_name)

    
