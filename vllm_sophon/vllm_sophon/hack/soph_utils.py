import torch
import torch.nn.functional as F

from vllm.distributed.parallel_state import get_world_group
from vllm.logger import init_logger
import concurrent.futures

logger = init_logger(__name__)

def copy_files(src_dir, dst_dir, exclude_ext=".safetensors"):
    import os, shutil
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for item in os.listdir(src_dir):
        src_path = os.path.join(src_dir, item)
        dst_path = os.path.join(dst_dir, item)
        if src_path.endswith(exclude_ext):
            continue
        if os.path.isdir(src_path):
            copy_files(src_path, dst_path, exclude_ext)
        else:
            shutil.copy2(src_path, dst_path)

def reorder_mm_weight(tensor):
    tensor = tensor.t().contiguous().view(dtype=torch.uint8)
    return tensor

def reorder_mm_zeros(tensor):
    tensor = tensor.view(dtype=torch.int8)

    # seperate low 4bit and high and convert to int8
    low_4bits = tensor & 0xf
    high_4_bits = (tensor >> 4) & 0xf

    # concat low 4bit and high 4bit
    flatten_tensor = torch.zeros(tensor.shape[0], tensor.shape[1] * 2, dtype=torch.int8)
    flatten_tensor[:, ::2] = low_4bits
    flatten_tensor[:, 1::2] = high_4_bits

    # qzeros + 1 and transpose
    flatten_tensor = (flatten_tensor + 1).t()

    # convert int8 to int4
    qzeros = torch.zeros(flatten_tensor.shape[0], flatten_tensor.shape[1] // 2, dtype=torch.uint8)
    qzeros[:] = (flatten_tensor[:, ::2] & 0xf) | ((flatten_tensor[:, 1::2] & 0xf) << 4)

    return qzeros

def reorder_mm_scale(tensor, dtype=torch.float16):
    tensor = tensor.t().to(dtype).contiguous().view(dtype=torch.uint8)
    return tensor

def reorder_mm_bias(tensor, dtype=torch.float16):
    tensor = tensor.to(dtype).t().contiguous()
    return tensor

def reorder_mlp_qweight(tensor):
    tensor = tensor.t().contiguous().view(dtype=torch.uint8)
    return tensor

def reorder_mlp_qweight_down(tensor, groupsize):
    # Transpose it into byte-wise shape of r/groupsize, c, groupsize
    groupsize //= torch.iinfo(tensor.dtype).bits // 8
    r, c = tensor.shape
    tensor = tensor.view(-1, groupsize, c).permute(0, 2, 1).contiguous()
    tensor = tensor.view(-1, c * groupsize).view(dtype=torch.uint8)
    return tensor

def reorder_mlp_scales_up_gate(tensor, dtype=torch.float16):
    tensor = tensor.to(dtype).t().contiguous()
    return tensor

def reorder_mlp_qzeros_up_gate(tensor):
    return reorder_mm_zeros(tensor)

def reorder_mlp_scales_down(tensor, dtype=torch.float16):
    tensor = tensor.to(dtype).contiguous()
    return tensor

def reorder_mlp_qzeros_down(tensor):
    tensor = tensor.view(dtype=torch.int8)

    # seperate low 4bit and high and convert to int8
    low_4bits = tensor & 0xf
    high_4_bits = (tensor >> 4) & 0xf

    # concat low 4bit and high 4bit
    flatten_tensor = torch.zeros(tensor.shape[0], tensor.shape[1] * 2, dtype=torch.int8)
    flatten_tensor[:, ::2] = low_4bits
    flatten_tensor[:, 1::2] = high_4_bits

    # qzeros + 1 and transpose
    flatten_tensor = flatten_tensor + 1

    # convert int8 to int4
    qzeros = torch.zeros(flatten_tensor.shape[0] // 2, flatten_tensor.shape[1], dtype=torch.uint8)
    qzeros[:] = (flatten_tensor[::2, :] & 0xf) | ((flatten_tensor[1::2, :] & 0xf) << 4)

    return qzeros

def gptq_reorder_kernel(out_tensors: list, dtype, name, file, groupsize):
    import re
    tensor = file.get_tensor(name)
    if re.match('.*\.self_attn.*\.qweight$', name):
        tensor = reorder_mm_weight(tensor)
    elif re.match('.*\.self_attn.*\.scales$', name):
        tensor = reorder_mm_scale(tensor, dtype)
    elif re.match('.*\.self_attn.*\.qzeros$', name):
        tensor = reorder_mm_zeros(tensor)
    elif re.match('.*\.self_attn.*\.bias$', name):
        tensor = reorder_mm_bias(tensor, dtype)
    elif re.match('.*\.mlp\.down_proj\.qweight$', name):
        tensor = reorder_mlp_qweight_down(tensor, groupsize)
    elif re.match('.*\.mlp.*\.qweight$', name):
        tensor = reorder_mlp_qweight(tensor)
    elif re.match('.*\.mlp\.(up|gate)_proj\.qzeros$', name):
        tensor = reorder_mlp_qzeros_up_gate(tensor)
    elif re.match('.*\.mlp\.down_proj\.qzeros$', name):
        tensor = reorder_mlp_qzeros_down(tensor)
    elif re.match('.*\.mlp\.(up|gate)_proj\.scales$', name):
        tensor = reorder_mlp_scales_up_gate(tensor, dtype)
    elif re.match('.*\.mlp\.down_proj\.scales$', name):
        tensor = reorder_mlp_scales_down(tensor, dtype)
    if tensor.dtype not in (torch.uint8, torch.int32) and tensor.dtype!= dtype:
        tensor = tensor.to(dtype)
    out_tensors[name] = tensor

def llama_w4a16_reorder(src_weight_file, dst_weight_file, groupsize, dtype):
    from safetensors import safe_open
    from safetensors.torch import save_file
    new_tensors = {}
    with safe_open(src_weight_file, framework="pt", device="cpu") as f:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(gptq_reorder_kernel, new_tensors, dtype, key, f, groupsize) for key in f.keys()]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    save_file(new_tensors, dst_weight_file)

def weight_reorder(model_id, quantize, dtype, groupsize):
    import os
    from vllm.distributed import get_tensor_model_parallel_rank
    if quantize in ["gptq", 'awq']:
        tp_rank = get_tensor_model_parallel_rank()
        reorder_cache_pth = os.path.join('/data', '.reorder_cache')
        reorder_path = os.path.join(reorder_cache_pth, model_id.rstrip('/').split('/')[-1])
        if not os.path.exists(reorder_path) and tp_rank == 0:
            logger.warning(f"Start weight reorder process...")
            copy_files(model_id, reorder_path, ".safetensors")
            for filename in os.listdir(model_id):
                if filename.endswith(".safetensors"):
                    llama_w4a16_reorder(
                        os.path.join(model_id, filename),
                        os.path.join(reorder_path, filename),
                        groupsize,
                        dtype
                    )
        get_world_group().barrier()
        logger.warning(f"Weight reorder success.")
        return reorder_path
    else:
        return None

class LLMQualityChecker:
    def __init__(self, base_url="https://www.sophnet.com/api/open-apis/v1"):
        from openai import OpenAI
        api_key = os.getenv("API_KEY", "")
        assert len(api_key) > 0, "Environment \"API_KEY\" must be provided."
        self.client = OpenAI(api_key=api_key, base_url=base_url)

        import logging
        logging.getLogger("httpx").setLevel(logging.WARNING)


    def check_quality(self, prompt, generated):
        """
        Check the quality of generated text by LLMs using Sophnet API.

        :param prompt: The original prompt given to the LLM.
        :param generated: The text generated by the LLM.
        :return: A dictionary containing the evaluation results.
        """
        # prefix = 'Please evaluating the coherence and format of generated text from my LLM.'
        # post_fix = 'Respond should as the following format: "@Score: <coherence and format score(out of 100, 60 indicates a normal sentence)>\n@Abnormal Words: <abnormal words list>\n@Abnormal Repeat: <abnormal repeat list>\n@Reason: <shot reason>"'

        # prefix = 'Please evaluating the quality(relevance and formatting) of generated text by my LLM.'
        post_fix = 'Respond should as the following format: "@ Score: <quality score(out of 100, 60 indicates a normal sentence)>\n@ Abnormal Words: <abnormal words list>\n@ Abnormal Repeat: <abnormal repeat list>\n@ Reason: <short reason>"'

        response = self.client.chat.completions.create(
            model="Qwen2.5-7B-Instruct",
            messages=[
                {"role": "system", "content": "You are a helpful assistant to evaluate the quality of completion text by LLMs."},
                {"role": "user", "content": f"Prompt: {prompt}\nCompletion text: {generated}\n{post_fix}"}
            ],
            max_tokens=200,
            temperature=0,
            top_p=0,
        )

        return response.choices[0].message.content.strip("\"")

def sanity_check_mm_encoder_outputs(
    mm_embeddings: object,
    expected_num_items: int,
) -> None:
    """
    Perform sanity checks for the result of
    [`vllm.model_executor.models.SupportsMultiModal.get_multimodal_embeddings`][].
    """
    assert isinstance(mm_embeddings, (list, tuple, torch.Tensor)), (
        "Expected multimodal embeddings to be a list/tuple of 2D tensors, "
        f"or a single 3D tensor, but got {type(mm_embeddings)} "
        "instead. This is most likely due to incorrect implementation "
        "of the model's `get_multimodal_embeddings` method.")

    assert len(mm_embeddings) == expected_num_items, (
        "Expected number of multimodal embeddings to match number of "
        f"input items: {expected_num_items}, but got {len(mm_embeddings)=} "
        "instead. This is most likely due to incorrect implementation "
        "of the model's `get_multimodal_embeddings` method.")

    assert all(e.ndim == 2 for e in mm_embeddings), (
        "Expected multimodal embeddings to be a sequence of 2D tensors, "
        f"but got tensors with shapes {[e.shape for e in mm_embeddings]} "
        "instead. This is most likely due to incorrect implementation "
        "of the model's `get_multimodal_embeddings` method.")
