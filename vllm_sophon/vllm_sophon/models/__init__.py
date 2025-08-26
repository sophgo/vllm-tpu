from vllm import ModelRegistry


def register_model():
    ModelRegistry.register_model(
        "DeepseekV3ForCausalLM",
        "vllm_sophon.models.soph_deepseek_v3:DeepseekV3ForCausalLM")

    ModelRegistry.register_model(
        "LlamaForCausalLM",
        "vllm_sophon.models.soph_llama:LlamaForCausalLM")

    ModelRegistry.register_model(
        "Qwen2ForCausalLM",
        "vllm_sophon.models.soph_qwen2:Qwen2ForCausalLM")

    ModelRegistry.register_model(
        "Qwen3ForCausalLM",
        "vllm_sophon.models.soph_qwen3:Qwen3ForCausalLM")
