## 项目介绍

本项目基于 [vLLM v7.3.0](https://github.com/vllm-project/vllm/releases/tag/v0.7.3)，支持在 Sophon TPU SG2260 上运行 LLaMa, Qwen, DeepSeek 等主流大语言模型。

### Release 地址

vLLM项目代码、docker镜像、主要模型权重/数据集等资源在 FTP 服务器上存放路径：
`ftp://172.28.141.89/LLMs/vLLM`

### 模型支持列表

| 模型名称      | 权重类型    | 模型链接     |
|---------------|-------------|------------|
| Llama2-7B     | FP16        | [Llama2-7B](https://huggingface.co/meta-llama/Llama-2-7b-hf) |
| Llama2-7B     | w4a16       | [Llama-2-7B-Chat-GPTQ](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GPTQ) |
| Llama3.1-8B   | w4a16       | [Llama3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8b) |
| Llama3.1-70B  | FP16        | [Llama3.1-70B](https://huggingface.co/meta-llama/Llama-3.1-70b) |
| Llama3.1-70B  | w4a16       | [Llama-3.1-70B-Instruct-int4-auto-gptq](https://huggingface.co/sofya-ai/Meta-Llama-3.1-70B-Instruct-int4-auto-gptq) |
| Qwen2-7B      | BF16        | [Qwen2-7B](https://huggingface.co/Qwen/Qwen2-7B) |
| Qwen2-7B      | w4a16       | [Qwen2-7B](https://huggingface.co/Qwen/Qwen2-7B-Instruct-GPTQ-Int4) |
| Qwen2-57B-A14B| BF16        | [Qwen2-57B-A14B-Instruct](https://huggingface.co/Qwen/Qwen2-57B-A14B-Instruct) |
| Qwen2-72B     | w4a16       | [Qwen2-72B](https://huggingface.co/Qwen/Qwen2-72B) |
| Qwen2.5-14B   | BF16        | [Qwen2.5-14B](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct) |
| QwQ-32B       | BF16        | [QwQ-32B](https://huggingface.co/Qwen/QwQ-32B) |
| QwQ-32B       | w4a16       | [QwQ-32B-AWQ](https://huggingface.co/Qwen/QwQ-32B-AWQ) |
| LLaVa-Next 7B | BF16        | [llava-v1.6-vicuna-7b](https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b) |
| LLaVa-Next 13B| BF16        | [llava-v1.6-vicuna-13b-hf](https://huggingface.co/llava-hf/llava-v1.6-vicuna-13b-hf) |
| Qwen2.5-VL 7B | BF16        | [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) |
| DeepSeek-V3   | FP8         | [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3)
| DeepSeek-R1   | FP8         | [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1)

**注释**：
- `FP16` 和 `BF16` 均为16位浮点数格式，但具体编码方式不同。
- `w4a16`: 表示权重为4位量化，激活值为16位量化。

## 项目环境搭建

### 准备模型权重

从[Hugging Face](https://huggingface.co/models)、[ModelScope](https://modelscope.cn/models)、[gitee AI](https://ai.gitee.com/models) 等社区下载模型权重到本地。

> *注意*：对于量化模型，我们会在初次运行该模型时对模型权重进行重排处理，处理后的模型保存在`/data/.reorder_cache/<MODEL_NAME>`中。由于版本的升级，权重处理逻辑可能会发生改变，因此在更新vLLM版本时应该删除重排的权重文件目录以便于重新生成。


### Docker环境

从 FTP 服务器上`LLMs/vLLM/daily_build/latest_release` 目录下拉取docker镜像并加载：

```shell
bunzip2 -c docker-soph_vllm-<TAG>.tar.bz2 | docker load
```

（可选）如有需要可以手动编译docker镜像

```shell
docker build -t soph_vllm:0.7.3 -f Dockerfile.sophtpu .
```

### 驱动及Runtime环境

驱动和运行时软件包在 FTP 服务器`ftp://172.28.141.89/sg2260/tpuv7-runtime/`对应目录的`PCIe/x86_6/`下，下载和安装即可。

以  `1.1.3` 版本为例，下载的文件如下：
```shell
tpuv7-driver_1.1.3_amd64.deb
tpuv7-runtime_1.1.3_amd64.deb
tpuv7-runtime-dev_1.1.3_amd64.deb
```
对应安装命令：
```shell
dpkg -i tpuv7-driver_1.1.3_amd64.deb
dpkg -i tpuv7-runtime_1.1.3_amd64.deb
dpkg -i tpuv7-runtime-dev_1.1.3_amd64.deb
```
安装完成后，可执行`tpu-smi` 命令查看芯片状态。

关于 TPUV7-RUNTIME 详细说明参见 FTP 服务器`/sg2260/tpuv7-runtime/daily_build/latest_release/Doc/` 目录下《TPUV7-RUNTIME快速入门指南.pdf》。


### docker 启动模型推理服务

docker 启动模型推理服务，将`模型权重`、`运行时目录`和`device`映射到docker容器中

#### Cmodel 推理


```shell
docker run --privileged -td --restart always \
  --name <CONTAINER_NAME> \
  --shm-size 1g \
  -p 8080:80 \
  -v <DATA_PATH>:/data \
  -v /opt/tpuv7:/opt/tpuv7 \
  soph_vllm:0.7.3
```

说明：
1. Cmodel模式下无需映射device，但需要映射 Runtime 目录。
2. `CMODEL_FAST_EXEC=1` 环境变量可以指定是否用oneDNN加速Cmodel推理。

#### Device 推理

```shell
docker run --privileged -td --restart always \
  --name <CONTAINER_NAME> \
  --shm-size 1g \
  -p 8080:80 \
  -v /dev/:/dev/ \
  -v <DATA_PATH>:/data \
  -v /opt/tpuv7:/opt/tpuv7 \
  soph_vllm:0.7.3
```

说明：
1. Device推理时需要将device和runtime目录映射到docker容器内。
2. c2c topo默认顺序为`0,1,2,3,4,5,6,7`，若实际topo和默认topo不一致，需要在创建docker时添加CHIP_MAP环境变量，如`-e CHIP_MAP=x,x,x,x`

#### 启动docker容器
```shell
docker exec -it <CONTAINER_NAME> bash
```

#### 安装Torch-TPU whl包

从FTP服务器上`torch_tpu/release_build/latest_release`目录下拉取torch-tpu whl包并安装：

```shell
tar -xvf torch-tpu_*.tar.gz
pip install dist/torch_tpu-*_x86_64.whl --force-reinstall
```

#### 映射 `vllm` 代码（可选，如果使用本地代码则需要映射）
1. 使用非插件化功能
```shell
export PYTHONPATH=path_to_vLLM
```
2. 使用插件化功能
```shell
export PYTHONPATH=path_to_vllm:path_to_vllm_sophon
```
**注意**：
 - 由于 `vLLM` 同样存在 `vllm`目录，因此使用插件化功能时需要优先设置官网 `vllm` 路径，否则会加载 `vLLM/vllm` 目录。即不能设置 `export PYTHONPATH=path_to_vllm_sophon:path_to_vllm`。
 - 当前仅支持 `vllm v0.7.3` 版本，需要切换到该分支。
 - 使用 `docker` 环境中的 `PyTorch` 环境: `python3 use_existing_torch.py`。
 - `vllm` 需要 `torch-2.5.1` 及以上版本，由于当前 `torch-tpu` 仅支持 `torch-2.1.0`，对于插件化功能需要临时以下部分代码:
   - 注释 `vllm/__init__.py` 37行代码 `torch._inductor.config.compile_threads = 1`
   - 修改 `vllm/_custom_ops.py` 28行代码 `if TYPE_CHECKING:` 为 `if True:`，注册 `register_fake` 函数。
 - 上述事项只针对插件化功能，源码修改都是在官方 `vllm/vllm` 代码，而不是 `vLLM/vllm` 代码。


### 设置环境变量

设置分布式环境变量`MASTER_ADDR`和`MASTER_PORT`，并设置使用V1引擎启动推理服务。

```shell
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=29500
export VLLM_USE_V1=1
```

### 启动推理服务

使用`api_server`启动推理服务(以Qwen2-7B模型为例):

```shell
cd /workspace/vLLM
python3 -m vllm.entrypoints.openai.api_server \
    --model /data/Qwen2-7B-Instruct/ \
    --host localhost \
    --port 8000 \
    --served-model-name vllm \
    --enforce-eager \
    --tensor-parallel-size 1
```
其中各参数含义如下
--model：必需参数，指定要加载的模型的路径
--host：可选参数，服务器的主机地址，默认为localhost
--port：可选参数，服务器的端口号，默认为8000
--served-model-name：可选参数，自定义API返回的模型名称，默认为`--model`的传入参数
--enforce-eager：必需参数，禁用CUDA图
--tensor-parallel-size：可选参数，指定要推理的张量并行数量，默认为1

### 发送请求

#### 发送单个请求
按如下格式向前文中映射的端口发送POST请求：

```shell
curl localhost:8000/v1/chat/completions \
    -X POST \
    -d '{
  "model": "vllm",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "What is deep learning?"
    }
  ],
  "stream": false,
  "max_tokens": 128
}' \
    -H 'Content-Type: application/json'
```
其中，可以设置最大生成token数量 `max_new_tokens`

#### 模拟多用户发送请求

打开一个新的终端并进入同一个docker，运行以下测试脚本可以模拟4个用户同时发送请求：

```shell
cd /workspace/vLLM/evaluation/stress/
bash emulate_4_users.sh
```

结果会在终端输出，同时脚本所在目录将生成日志文件run_*.log。

## 单测/性能测试

我们提供简单的python单测脚本测试模型推理性能，该模式需要以交互模式进入docker容器后在容器内执行。
单测脚本将会给出两次`FTL`和`TPS`性能数据，第一次为`Warmup`，第二次为真实推理性能。


### 单测环境配置

1. 启动docker容器：

    ```shell
    docker run --privileged -itd --restart always \
              --name <CONTAINER_NAME> \
              --shm-size 1g \
              -p 8080:80 \
              -v $(pwd):/workspace \
              -v /dev/:/dev/ \
              -v <DATA_PATH>:/data \
              -v /opt/tpuv7:/opt/tpuv7 \
              --entrypoint /bin/bash \
              soph_vllm:0.7.3
    ```

2. 进入docker容器，初始化环境：

    ```shell
    docker exec -it <CONTAINER_NAME> bash
    ```

3. 安装Torch-TPU whl包

    从FTP服务器上`torch_tpu/release_build/latest_release`目录下拉取torch-tpu whl包并安装：

    ```shell
    tar -xvf torch-tpu_*.tar.gz
    pip install dist/torch_tpu-*_x86_64.whl --force-reinstall
    ```

### test_model.py 单测脚本使用说明

`test_model.py`脚本用于测试SOPH vLLM内LLM推理性能，相关参数如下：
  - --model-id: 必需参数，指定要加载的模型的路径。
  - --dtype: 可选参数，指定模型运行时的数据类型。
  - --quantize: 可选参数，指定强制加载的模型的数据类型，quantize为true时尽量量化模型推理。
  - --batch: 可选参数，默认构造单测批次为1。
  - --input-length: 可选参数，指定输入长度，默认128。
  - --max-new_tokens: 可选参数，指定生成的结果长度，默认1024。
  - --tp_size: 可选参数，指定要推理的张量并行数量，默认为1。
  - --mode: 可选参数，指定运行模式，可选值为 chat 和 generate，默认值为 generate。
  - --useV1: 可选参数，指定使用的vLLM Engine类型，默认使用vLLM V1 Engine。
  - --save-results: 可选参数，是否保存性能结果到csv文件中。
  - --quality-check: 可选参数，是否进行生成文本质量检测。
  - --save-json: 可选参数，是否以json格式保存输出到文件


#### 测试 LLaMA-7B 模型:

  ```shell
  python3 test_model.py --model-id /data/llama-2-7b-chat-hf
  ```

#### 生成模式下测试多模态 LLaVA-Next 模型:

  ```shell
  python3 test_model.py --model-id /data/llava-v1.6-vicuna-7b --max-new-tokens=20 --mode generate
  ```

#### 测试多芯并行推理结果:

  ```shell
  CHIP_MAP=0,1 test_model.py --model-id /data/llama-2-7b-chat-hf --batch 4 --tp_size 2
  ```

  CHIP_MAP=0,1：环境变量CHIP_MAP用于指定分布式推理需要使用的芯片编号。


### 测试Llama2-7b/Qwen2-7b模型推理性能

#### 测试Llama2-7b推理性能

```shell
python3 test_model.py --model-id /data/llama-2-7b-chat-hf --quantize gptq --batch 8 --input-length 4096 --max-new-tokens 128
```

#### Qwen2-7b推理性能

```shell
python3 test_model.py --model-id /data/Qwen2-7B-Instruct --quantize gptq --batch 8 --input-length 4096 --max-new-tokens 128
```

### 测试Llama2-70b/Qwen2-72b模型推理性能

#### 测试Llama2-70b推理性能

```shell
python3 test_model.py --model-id /data/Llama-2-70b-chat-hf --batch 16 --input-length 1024 --max-new-tokens 128 --tp_size 2
```

#### Qwen2-72b推理性能

```shell
python3 test_model.py --model /data/Qwen2-72B-Instruct --quantize gptq --input-length 512 --max-new-tokens 128
```

### 性能测试注意事项

1. 为测出最佳性能，需要关闭 Host CPU 变频。参考命令如下：

    ```shell
    echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
    cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
    ```
2. 如需运行权重较大的模型如`Llama2-70b/Qwen2-72b`，可以修改板卡配置，将芯片中分配给多媒体使用的一部分 DDR 内存分配给TPU. 修改方式如下：

    驱动安装成功后， 在`/lib/firmware/tpuv7/`目录下会有`sc11_config.ini`和`sc11_config_chip2.ini`两个文件，修改如下字段：

    ```shell
    share-mem-start=0x1e0000000
    share-mem-size=0x1e20000000
    ```
    重新加载驱动即可使配置生效。

## Prefix-Caching功能测试

### 功能介绍

Prefix-Caching的核心思想是缓存系统提示和历史对话中的 KV Cache，以便在后续请求中复用，从而减少首次 Token 的计算开销。在长系统提示以及多轮对话的场景下，当一个新的请求到来的时候，通过前缀匹配查找新的请求是否命中缓存，如果命中了缓存，会复用KV Cache来完成这次请求。因此Prefix-Caching能够实现跨请求的 KV Cache 复用，从而显著提升 prefill 的性能，并降低新一轮请求的首次 Token 计算时间（Time To First Token，TTFT）。

### 测试方法

1. 启动推理服务

首先进入docker并使用启动vLLM推理服务(以Qwen2-7B模型为例)：

```shell
cd /workspace/vLLM
python3 -m vllm.entrypoints.openai.api_server \
    --model /data/Qwen2-7B-Instruct/ \
    --host localhost \
    --port 8000 \
    --served-model-name vllm \
    --enforce-eager \
    --tensor-parallel-size 1 \
    --enable-prefix-caching
```
其中：
--enable-prefix-caching：可选参数，开启prefix-caching功能，默认开启。

2. 运行测试脚本

当服务成功Connected后，打开一个新的终端并进入同一个docker，运行测试脚本。

设计了4个测试问题，分别放在long_question_1.txt，long_question_2.txt，long_question_3.txt，long_question_4.txt文件中。这4个问题以同一篇文章作为背景，在文章末尾给出4个简短的问题。为了模拟长系统提示以及多轮对话的场景，运行脚本将4个测试问题按照顺序发送到推理服务中：

```shell
cd /workspace/vLLM/evaluation/test_prefix_caching/
bash emulate_1_user_prefix.sh
```

可以在vLLM推理服务启动的终端看到类似如下输出:
```shell
INFO 08-01 12:34:33 loggers.py:76] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 24.9 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.1%, Prefix cache hit rate: 74.2%
```

其中`Prefix cache hit rate`表示所有请求累积的Prefix cache命中率，`Prefix cache hit rate>0`证明该功能起作用。

同时，脚本所在目录将生成日志文件run_question*.log。
通过检查日志中的输出结果可以验证功能的正确性：若输出内容通顺、连贯且合理，则证明prefix-caching功能正常。

## Prefill-Chunking功能测试

### 功能介绍
当前推理框架中，Prefill阶段因密集计算表现为Compute-bound，而Decode阶段因频繁访存表现为Memory-bound，导致硬件资源利用不均衡。
针对这一问题，Prefill-Chunking（预填充分块）将Prefill请求的 prompts 切分为多个长短大致相同的 chunks，在这些 chunks 做Prefill的同一时刻捎带处理其他已就绪的 Decode 请求，也就是将处于 Prefill 阶段的请求与处于 Decode 阶段的请求组成一个批次（batch）进行计算，从而实现算力和带宽的利用率最大化。
此外，Prefill-Chunking功能也优化了用户体验。具体来说，传统实现中，TGI服务实例在同一时刻仅能处理单一阶段（Prefill优先），高并发时Decode易被阻塞，引发Token输出延迟。而Prefill-Chunking功能可以实现Decode Token连续输出，无需等待其他Prefill完成。

不过，当前TPU架构暂不支持指令级并行，导致batch内的Decode和Prefill仍需顺序处理。这种做法虽然无法实现带宽和算力的利用率最大化，但能有效缓解Decode阻塞问题，提升用户体验。

### 测试方法

1. 启动推理服务

首先进入docker并使用以下命令启动推理服务（以Qwen2-7B模型为例）：

```shell
python3 -m vllm.entrypoints.openai.api_server \
    --model /data/Qwen2-7B-Instruct/ \
    --host localhost \
    --port 8000 \
    --served-model-name vllm \
    --enforce-eager \
    --tensor-parallel-size 1 \
    --enable-chunked-prefill true \
    --max-num-batched-tokens 100 \
    --max-num-seqs 32 \
```

其中各参数含义如下
--enable-chunked-prefill：可选参数，开启Prefill-Chunking功能，默认开启。
--max-num-batched-tokens：可选参数，多个请求总共可以处理的最大的prefill token数量,默认值为2048。
--max-num-seqs：可选参数，同时处理的最大序列数，每个序列通常对应一个独立的用户请求，默认值为256。


2. 运行测试脚本

服务启动后，在新终端中进入同一Docker容器，执行测试脚本：

```shell
cd /workspace/vLLM/evaluation/test_prefill_chunking/
bash emulate_4_users_prefill_chunking.sh
```
设计了5个测试问题，每个问题的token数量约为600。
将`--max-num-batched-tokens`参数设置为`100`实现对prefill做多次切块，该参数设置得越小，切块数量越多，每块的prefill token数量越少。
运行测试脚本，会同时将这5个问题发送到推理服务中，以模拟多轮对话场景。测试完成后，脚本所在目录将生成日志文件run_*_prefill.log。
通过检查日志中的输出结果可以验证功能的正确性：若输出内容通顺、连贯且合理，则证明prefill-chunking功能正常。


## 开发者指南

本章节介绍在vLLM框架进行调试相关的特性和工具。

### 常用环境变量介绍

| 环境变量名称          | 说明                                                    | 默认值            |
|-----------------------|---------------------------------------------------------|-------------------|
| `DEVICE`              | 指定设备类型，`SOPHTPU`/`GPU`                           | `SOPHTPU`         |
| `DISABLE_CACHE`       | 是否禁用指令缓存。`1` 禁用，`0` 启用。                  | `1`               |
| `CMODEL_FAST_EXEC`    | 是否使用 oneDNN 加速 cmodel。                           | `OFF`             |
| `OMP_NUM_THREADS`     | omp线程数量                                             |cpu thread         |
| `WORLD_SIZE`          | 总进程数，用于分布式训练。                              | `1`               |
| `OMPI_COMM_WORLD_RANK`| OpenMPI 提供的 rank ID（优先级高于 `RANK`）。           | -                 |
| `OMPI_COMM_WORLD_SIZE`| OpenMPI 提供的总进程数（优先级高于 `WORLD_SIZE`）。     | -                 |
| `RANK`                | 当前进程的 rank ID，用于分布式推理/训练。               | `0`               |
| `CHIP_MAP`            | 芯片映射，用于指定使用的芯片 ID。                       | -                 |

### 启动性能分析(TODO)

#### 全流程性能分析
1. 如需对模型推理全流程进行性能分析，`export ENABLE_PROFILE = 1`启用性能分析，单测脚本将会在warmup之后的推理过程进行性能分析。
2. 运行单测，当前工作目录下会生成性能分析中间结果`cdm_profile_data_dev<DEV_ID>-<IDX>`，\
其中`<DEV_ID>`是当前使用的设备ID，`<IDX>`是当前开启profile的ID。
3. 使用[profile工具](#profile工具)进行进一步可视化及分析。

#### 指定模块/算子性能分析
1. 如需对特定算子或推理模块进行性能分析，在需要分析的算子或模块前后添加代码：

    ```python
    torch.ops.my_ops.enable_profile(max_record_num, mode)
    # max_record_num = 1e6, mode = 0/1/2
    # ...... ## 待分析代码块_IDX
    torch.ops.my_ops.disable_profile()
    ```
2. 运行单测，如果在多个位置或代码块开启了profile记录, 可以根据调用顺序与生成目录`cdm_profile_data_dev<DEV_ID>-<IDX>`中的`<IDX>`对应。
3. 使用[profile工具](#profile工具)进行进一步可视化及分析。

### Profile工具

1. 工具简介：

    本工具用于分析模型推理过程中的性能瓶颈，包括算子执行时间、内存占用等。支持对全流程性能分析和指定模块/算子性能分析。

2. 发布地址：[bigTpuProfile · PyPI](https://pypi.org/project/bigTpuProfile/)

    使用详情见发布页。

      ```shell
      # 安装
      pip install bigTpuProfile
      # 使用 bigTpuProfile -h 查看可用参数
      bigTpuProfile cdm_profile_data_devX-X/ result_out --arch BM1690
      # 可视化结果存储于result_out
      ```
