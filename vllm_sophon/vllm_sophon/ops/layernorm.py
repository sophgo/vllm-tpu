from typing import Optional, Tuple
import torch
import torch_tpu
from vllm.model_executor.layers.layernorm import RMSNorm

class SophonRMSNorm(RMSNorm):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: Optional[int] = None,
        has_weight: bool = True,
    ) -> None:
        super().__init__(
            hidden_size,
            eps,
            var_hidden_size,
            has_weight
        )

    def forward(
        self,
        x: torch.Tensor,
        output: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            x = x + residual

        residual = x
        torch.ops.my_ops.rmsnorm_forward(
            x,
            self.weight,
            None,
            output,
            x.dim() - 1,
            self.variance_epsilon
        )
        return output, residual
