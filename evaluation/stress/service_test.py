from openai import OpenAI
import random

import argparse

parser = argparse.ArgumentParser(description="Process some integers.")

parser.add_argument('--loops', default=10, type=int,
                    help='loop num, default 10')
parser.add_argument('--port', default=8000, type=int,
                    help='port, default 8000')

args = parser.parse_args()

client = OpenAI(
        base_url=f"http://localhost:{args.port}/v1",
    api_key="-"
)

def post_request(msg):
    print(f'Question: \n{msg}')
    chat_completion = client.chat.completions.create(
        model="vllm",
        messages=[
            {"role": "system", "content": "You are a helpful assistant." },
            {"role": "user", "content": f"{msg}"}
        ],
        max_tokens=1024,
        stream=True
    )

    print(f'Answer:')
    for message in chat_completion:
        print(message.choices[0].delta.content, end="")
    print()


QUESTIONS = [
    "全球十大半导体公司",
    "介绍算能科技",
    "介绍比特大陆",
    "介绍RISC-V在中国的发展前景",
    "写一首关于中国科技的诗",
    "What is deep learning?",
    "简单介绍什么是TPU？",
    "周末去巴黎游玩，有什么推荐"
]

for loop_idx in range(args.loops):
    print(f'---------------- loop: {loop_idx} -----------')
    question = random.choice(QUESTIONS)
    post_request(question)