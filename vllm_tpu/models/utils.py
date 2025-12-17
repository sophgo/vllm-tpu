import torch

from vllm.multimodal import NestedTensors

def soph_flatten_embeddings(embeddings: NestedTensors, batch_size=1024) -> torch.Tensor:
    """
    Recursively flattens and concatenates NestedTensors on all but the last
    dimension.
    """

    leaves = []
    def collect_leaves(e):
        if isinstance(e, torch.Tensor):
            leaves.append(e.flatten(0, -2))
        else:
            for t in e:
                collect_leaves(t)

    collect_leaves(embeddings)

    if len(leaves) > batch_size:
        chunks = [torch.cat(leaves[i:i+batch_size], dim=0) for i in range(0, len(leaves), batch_size)]
        return torch.cat(chunks, dim=0)
    else:
        return torch.cat(leaves, dim=0)

def soph_embed_multimodal(
    input_ids: torch.Tensor,
    multimodal_token_id: int,
    inputs_embeds,
    multimodal_embeds: NestedTensors,
) -> torch.Tensor:
    """
    Embed token IDs and multimodal inputs and combine their embeddings.

    ``multimodal_token_id`` is used to determine whether a token ID should
    be embedded using ``get_text_embeds`` or ``get_multimodal_embeds``.

    Compared to ``merge_multimodal_embeddings`, this avoids running
    ``get_text_embeds`` on ``input_ids[input_ids == multimodal_token_id]``
    which causes issues when the placeholder token ID exceeds the
    vocabulary size of the language model.
    """
    is_multimodal = input_ids == multimodal_token_id

    try:
        flattened = soph_flatten_embeddings(multimodal_embeds)
    except Exception as e:
        raise RuntimeError(
            f"Cannot fill images right now. If error happens at warmup, make sure you have enough `--max-input-tokens`  to handle images. If error happens at regular runtime, please fill in an issue: {e}"
        )

    diffs = is_multimodal[1:].int() - is_multimodal[:-1].int()
    starts = torch.where(diffs == 1)[0]
    ends = torch.where(diffs == -1)[0]
    if len(starts) != len(ends):
        inputs_embeds[is_multimodal] = flattened.view(-1, flattened.shape[-1])
    else:
        current_ptr = 0
        for start_idx, end_idx in zip(starts, ends):
            region_length = end_idx - start_idx
            inputs_embeds[start_idx+1:end_idx+1] = flattened[current_ptr:current_ptr + region_length]
            current_ptr += region_length

    return inputs_embeds


