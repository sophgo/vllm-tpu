## 项目介绍

本项目基于 [vLLM v7.3.0](https://github.com/vllm-project/vllm/releases/tag/v0.7.3)，支持在 Sophon TPU SG2260 上运行 LLaMa, Qwen, DeepSeek 等主流大语言模型。

### Release 地址

vLLM项目代码、docker镜像、主要模型权重/数据集等资源在 FTP 服务器上存放路径：
`ftp://172.28.141.89/LLMs/vLLM

### TGI模型支持列表

| 模型名称      | 权重类型    | 模型链接     |
|---------------|-------------|------------|
| Llama2-7B     | FP16        | [Llama2-7B](https://huggingface.co/meta-llama/Llama-2-7b-hf) |
| Llama2-7B     | w4a16       | [Llama-2-7B-Chat-GPTQ](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GPTQ) |
| Qwen2-7B      | BF16        | [Qwen2-7B](https://huggingface.co/Qwen/Qwen2-7B) |
| Qwen2-7B      | w4a16       | [Qwen2-7B](https://huggingface.co/Qwen/Qwen2-7B-Instruct-GPTQ-Int4) |
| Qwen2-72B     | w4a16       | [Qwen2-72B](https://huggingface.co/Qwen/Qwen2-72B) |

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

#### 安装Torch-TPU whl包

从FTP服务器上`torch_tpu/release_build/latest_release`目录下拉取torch-tpu whl包并安装：

```shell
tar -xvf torch-tpu_*.tar.gz 
pip install dist/torch_tpu-*_x86_64.whl --force-reinstall
```


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
    source soph_envsetup.sh
    ```

### test_whole_model.py 单测脚本使用说明

`test_whole_model.py`脚本用于测试SOPH TGI内LLM推理性能，相关参数如下：
  - --model: 必需参数，指定要加载的模型的路径。
  - --quantize: 可选参数，指定强制加载的模型的数据类型，quantize为true时尽量量化模型推理。
  - --batch: 可选参数，默认构造单测批次为1。
  - --tp_size: 可选参数，指定要推理的张量并行数量，默认为1。
  - --mode: 可选参数，指定运行模式，可选值为 chat 和 generate，默认值为 generate。

#### 测试 LLaMA-7B 模型:

  ```shell
  python test_whole_model.py --model /data/llama-2-7b-chat-hf/
  ```

#### 测试多芯并行推理结果:

  ```shell
  CHIP_MAP=0,1 test_whole_model.py --model /data/llama-2-7b-chat-hf/ --batch 4 --tp_size 2
  ```

  CHIP_MAP=0,1：环境变量CHIP_MAP用于指定分布式推理需要使用的芯片编号。


### 测试Llama2-7b/Qwen2-7b模型推理性能

#### 测试Llama2-7b推理性能

```shell
CONTEXT_LEN=4096 DECODE_TOKEN_LEN=32 python test_whole_model.py --model llama2-7b --quantize gptq --batch 8
```

#### Qwen2-7b推理性能

```shell
CONTEXT_LEN=4096 DECODE_TOKEN_LEN=32 python test_whole_model.py --model qwen2-7b --quantize gptq --batch 8
```

> 可通过`CONTEXT_LEN`指定上下文长度，通过`DECODE_TOKEN_LEN`指定最大生成的token数量。

### 测试Llama2-70b/Qwen2-72b模型推理性能

#### 测试Llama2-70b推理性能

```shell
CONTEXT_LEN=1024 DECODE_TOKEN_LEN=32 test_whole_model.py --model llama2-70b --batch 16 --tp_size 2
```

#### Qwen2-72b推理性能

```shell
CONTEXT_LEN=512 DECODE_TOKEN_LEN=32 test_whole_model.py --model qwen2-72b --quantize gptq
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


## 开发者指南

本章节介绍在vLLM框架进行调试相关的特性和工具。

### 常用环境变量介绍

| 环境变量名称          | 说明                                                    | 默认值            |
|-----------------------|---------------------------------------------------------|-------------------|
| `DEVICE`              | 指定设备类型，`SOPHTPU`/`GPU`                           | `SOPHTPU`         |
| `DISABLE_CACHE`       | 是否禁用指令缓存。`1` 禁用，`0` 启用。                  | `1`               |
| `DECODE_TOKEN_LEN`    | 解码时生成的最大 token 数量。                           | `10`              |
| `CONTEXT_LEN`         | 上下文长度，包括输入长度和解码长度。                    | `6`               |
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
