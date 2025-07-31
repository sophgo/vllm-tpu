from openai import OpenAI
import random

import argparse

parser = argparse.ArgumentParser(description="Process some integers.")

parser.add_argument('--loops', default=10, type=int,
                    help='loop num, default 10')
parser.add_argument('--port', default=8000, type=int,
                    help='port, default 8000')
# 添加新参数指定问题文件路径
parser.add_argument('--questions_file', default='long_questions.txt', type=str,
                    help='path to questions txt file, default questions.txt')

args = parser.parse_args()

# 读取问题文件
def load_questions(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        questions = [line.strip() for line in f if line.strip()]
    return questions

# 加载问题
QUESTIONS = load_questions(args.questions_file)

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


for loop_idx in range(args.loops):
    print(f'---------------- loop: {loop_idx} -----------')
    question = random.choice(QUESTIONS)
    post_request(question)