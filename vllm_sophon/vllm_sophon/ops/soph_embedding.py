
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

from vllm.model_executor.layers.vocab_parallel_embedding import pad_vocab_size
from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod
from vllm.model_executor.layers.vocab_parallel_embedding import DEFAULT_VOCAB_PADDING_SIZE

class SophEmbedding(torch.nn.Module):
    """VocabEmbedding without TP on SophTPU."""

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 params_dtype: Optional[torch.dtype] = None,
                 org_num_embeddings: Optional[int] = None,
                 padding_size: int = DEFAULT_VOCAB_PADDING_SIZE,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.org_vocab_size = org_num_embeddings or num_embeddings
        self.padding_size = padding_size

        self.num_added_embeddings = num_embeddings - self.org_vocab_size
        self.org_vocab_size_padded = pad_vocab_size(
            self.org_vocab_size, padding_size
        )
        self.num_embeddings_padded = pad_vocab_size(
            self.org_vocab_size_padded + self.num_added_embeddings,
            padding_size
        )

        self.quant_method = None
        if quant_config is not None:
            self.quant_method = quant_config.get_quant_method(self, prefix=prefix)
        if self.quant_method is None:
            self.quant_method = UnquantizedEmbeddingMethod()

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        self.quant_method.create_weights(
            self,
            self.embedding_dim,
            [self.num_embeddings_padded],
            self.embedding_dim,
            self.num_embeddings_padded,
            params_dtype=params_dtype
        )

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        return self.quant_method.embedding(
            self,
            input_.int()
        )

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

class SophParallelLMHead(SophEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        org_num_embeddings: Optional[int] = None,
        padding_size: int = 128, 
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            params_dtype=params_dtype,
            org_num_embeddings=org_num_embeddings,
            padding_size=padding_size,
            quant_config=quant_config,
            prefix=prefix
        )
        self.bias = None
        if bias:
            self.bias = nn.Parameter(
                torch.empty(num_embeddings, dtype=params_dtype)
            )
            set_weight_attrs(self.bias, {
                "output_dim": 0,
                "weight_loader": self.weight_loader
            })
        else:
            if not hasattr(self, 'bias'):
                self.register_parameter("bias", None)

    def tie_weights(self, embed_tokens: SophEmbedding):
        if self.quant_config and self.quant_config.get_name() == "gguf":
            return embed_tokens
        else:
            self.weight = embed_tokens.weight
            return self

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        output = super().forward(input_)
        if self.bias is not None:
            output += self.bias
        return output
