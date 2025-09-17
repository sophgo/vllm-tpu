# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from functools import cached_property
from typing import (Final, Iterable, List, Literal, Mapping, Optional,
                    Protocol, Set, Tuple, TypedDict, TypeVar, Union)

import torch
import torch.nn as nn
from transformers import BatchFeature, LlavaNextConfig, LlavaNextProcessor
from typing_extensions import NotRequired

from vllm.attention import AttentionMetadata
from vllm.config import VllmConfig
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, NestedTensors
from vllm.multimodal.parse import ImageSize
from vllm.sequence import IntermediateTensors

from .soph_clip import CLIPVisionModel
from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsPP
from vllm.model_executor.models.llava import (BaseLlavaMultiModalProcessor, BaseLlavaProcessingInfo,
                    LlavaDummyInputsBuilder, LlavaLikeConfig)
from vllm.model_executor.models.utils import (AutoWeightsLoader, flatten_bn,
                    init_vllm_registered_model, maybe_prefix)

from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.distributed import (get_tensor_model_parallel_world_size,
                              get_tensor_model_parallel_rank,
                              tensor_model_parallel_all_reduce)
from vllm_sophon.ops.soph_linear import (SophColumnParallelLinear,
                                         SophRowParallelLinear)

class LlavaNextImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    data: Union[torch.Tensor, List[torch.Tensor]]
    """
    Shape:
    `(batch_size * num_images, 1 + num_patches, num_channels, height, width)`

    Note that `num_patches` may be different per batch and image,
    in which case the data is passed as a list instead of a batched tensor.
    """

    image_sizes: NotRequired[torch.Tensor]
    """
    Shape: `(batch_size * num_images, 2)`

    This should be in `(height, width)` format.
    """


class LlavaNextImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    data: torch.Tensor
    """Shape: `(batch_size * num_images, image_feature_size, hidden_size)`

    `hidden_size` must match the hidden size of language model backbone.
    """


LlavaNextImageInputs = Union[LlavaNextImagePixelInputs,
                             LlavaNextImageEmbeddingInputs]


class LlavaNextLikeConfig(LlavaLikeConfig, Protocol):
    image_grid_pinpoints: Final[list[list[int]]]

from transformers.image_processing_utils import select_best_resolution
def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    """
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.

    Args:
        image_size (`tuple`):
            The size of the input image in the format (height, width).
        grid_pinpoints (`List`):
            A list containing possible resolutions. Each item in the list should be a tuple or list
            of the form `(height, width)`.
        patch_size (`int`):
            The size of each image patch.

    Returns:
        tuple: The shape of the image patch grid in the format (height, width).
    """
    if not isinstance(grid_pinpoints, list):
        raise ValueError("grid_pinpoints should be a list of tuples or lists")

    height, width = select_best_resolution(image_size, grid_pinpoints)
    return height // patch_size, width // patch_size

def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
        tensor (`torch.Tensor`):
            The image tensor, assumed to be of shape (height, width, num_channels).
        original_size (`tuple`):
            The original size of the image (height, width).

    Returns:
        `torch.Tensor`: The unpadded image tensor.
    """
    original_height, original_width = original_size
    current_height, current_width = tensor.shape[:-1]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[padding : current_height - padding, :, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, padding : current_width - padding, :]

    return unpadded_tensor

class LlavaNextProcessingInfo(BaseLlavaProcessingInfo):

    def get_hf_config(self) -> LlavaNextLikeConfig:
        return self.ctx.get_hf_config(LlavaNextConfig)

    def get_hf_processor(self, **kwargs: object):
        hf_processor = self.ctx.get_hf_processor(LlavaNextProcessor, **kwargs)

        # In case patch_size is omitted from `processor_config.json`
        # e.g. for E5-V: https://huggingface.co/royokong/e5-v
        if hf_processor.patch_size is None:
            patch_size = self.get_vision_encoder_info().get_patch_size()
            hf_processor.patch_size = patch_size

        return hf_processor

    # Based on: https://github.com/huggingface/text-generation-inference/blob/v3.0.1/server/text_generation_server/models/vlm_causal_lm.py#L113
    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        hf_config = self.get_hf_config()
        vision_encoder_info = self.get_vision_encoder_info()

        base_feature_size = self._apply_feature_select_strategy(
            hf_config.vision_feature_select_strategy,
            vision_encoder_info.get_num_image_tokens(
                image_width=image_width,
                image_height=image_height,
            ),
        )

        num_patch_height, num_patch_width = get_anyres_image_grid_shape(
            image_size=(image_height, image_width),
            grid_pinpoints=hf_config.image_grid_pinpoints,
            patch_size=vision_encoder_info.get_image_size(),
        )

        (
            unpadded_feature_size,
            newline_feature_size,
        ) = self._get_num_unpadded_features(
            original_height=image_height,
            original_width=image_width,
            npatches=vision_encoder_info.get_patch_grid_length(),
            num_patch_height=num_patch_height,
            num_patch_width=num_patch_width,
        )

        return unpadded_feature_size + newline_feature_size + base_feature_size

    # Based on: https://github.com/huggingface/text-generation-inference/blob/v3.0.1/server/text_generation_server/models/vlm_causal_lm.py#L86
    def _get_num_unpadded_features(
        self,
        *,
        original_height: int,
        original_width: int,
        npatches: int,
        num_patch_height: int,
        num_patch_width: int,
    ) -> tuple[int, int]:
        current_height = npatches * num_patch_height
        current_width = npatches * num_patch_width

        aspect_ratio = original_width / original_height
        current_aspect_ratio = current_width / current_height

        if aspect_ratio > current_aspect_ratio:
            new_height = (original_height * current_width) // original_width
            padding = (current_height - new_height) // 2
            current_height = current_height - (2 * padding)
        else:
            new_width = (original_width * current_height) // original_height
            padding = (current_width - new_width) // 2
            current_width = current_width - (2 * padding)

        unpadded_features = current_height * current_width
        newline_features = current_height

        return (unpadded_features, newline_features)

    def get_image_size_with_most_features(self) -> ImageSize:
        hf_config = self.get_hf_config()

        largest_feature_size, largest_feature_pinpoint = 0, None
        for (height, width) in hf_config.image_grid_pinpoints:
            feat_size = self.get_num_image_tokens(image_width=width,
                                                  image_height=height)
            if feat_size > largest_feature_size:
                largest_feature_size = feat_size
                largest_feature_pinpoint = ImageSize(width=width,
                                                     height=height)

        if largest_feature_size == 0 or largest_feature_pinpoint is None:
            raise ValueError("Cannot have a largest feature size of 0!")

        return largest_feature_pinpoint


_I = TypeVar("_I", bound=LlavaNextProcessingInfo)


class BaseLlavaNextMultiModalProcessor(BaseLlavaMultiModalProcessor[_I]):

    # Copied from BaseMultiModalProcessor
    @abstractmethod
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        raise NotImplementedError


class LlavaNextMultiModalProcessor(
        BaseLlavaNextMultiModalProcessor[LlavaNextProcessingInfo]):

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
            image_sizes=MultiModalFieldConfig.batched("image"),
            image_embeds=MultiModalFieldConfig.batched("image"),
        )


class SophLlavaNextMultiModalProjector(nn.Module):

    def __init__(self,
                 vision_hidden_size: int,
                 text_hidden_size: int,
                 projector_hidden_act: str,
                 multimodal_projector_bias: bool,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()

        self.tp_size = get_tensor_model_parallel_world_size()
        self.rank = get_tensor_model_parallel_rank()
        self.linear_1 = SophColumnParallelLinear(vision_hidden_size,
                                             text_hidden_size,
                                             bias=multimodal_projector_bias,
                                            skip_bias_add = False,
                                             quant_config=quant_config,
                                             prefix=f"{prefix}.linear_1")

        self.linear_2 = SophRowParallelLinear(text_hidden_size,
                                          text_hidden_size,
                                          bias=multimodal_projector_bias,
                                          skip_bias_add = False,
                                          quant_config=quant_config,
                                          prefix=f"{prefix}.linear_2")

        self.register_buffer("w1", self.linear_1.weight.data.transpose(0,1).contiguous())
        self.register_buffer("w2", self.linear_2.weight.data.transpose(0,1).contiguous())
        self.register_buffer("b1", self.linear_1.bias.data.contiguous())
        self.register_buffer("b2", None)
        if self.rank == 0:
            self.register_buffer("b2", self.linear_2.bias.data.contiguous())

        self.out_buffer = None

    def forward(self, out_buffer, image_features: torch.Tensor) -> torch.Tensor:

        image_features = image_features.contiguous()
        self.out_buffer = out_buffer
        torch.ops.my_ops.llava_mlp(image_features,
                                   self.w1, self.w2,
                                   self.b1, self.b2,
                                   self.out_buffer)
        if self.tp_size > 1:
            self.out_buffer = tensor_model_parallel_all_reduce(self.out_buffer)

        return self.out_buffer

def _get_num_hidden_layers(hf_config: LlavaLikeConfig) -> int:
    """Determine the number of hidden layers to initialize up to in the
    visual encoder.
    
    Args:
        hf_config: Model config with vision feature layer(s).
    """
    feature_layers = hf_config.vision_feature_layer
    num_hidden_layers = hf_config.vision_config.num_hidden_layers
    # If we have one feature layer, initialize up to that layer
    if isinstance(feature_layers, int):
        return _get_layer_index(feature_layers, num_hidden_layers)
    # If we have multiple feature layers, initialize up to the deepest one
    elif isinstance(feature_layers, (list, tuple)):
        return max(
            _get_layer_index(idx, num_hidden_layers) for idx in feature_layers)
    raise TypeError(f"vision_layer_feature type: {type(feature_layers)}"
                    " is not supported")

def _get_layer_index(feature_layer_index: int, num_hidden_layers: int) -> int:
    """Given a signed vision feature layer, get the number of hidden layers
    needed to leverage it.

    Args:
        feature_layer_index: Index of a required layer in the visual encoder.
        num_hidden_layers: The total number of hidden layers in the visual
            encoder.
    """
    if feature_layer_index < 0:
        return num_hidden_layers + feature_layer_index + 1
    return feature_layer_index


def _flatten_embeddings(embeddings: NestedTensors, batch_size=1024) -> torch.Tensor:
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

def _embed_multimodal(
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
        flattened = _flatten_embeddings(multimodal_embeds)
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

from transformers import (BatchFeature, CLIPVisionConfig)
def init_vision_tower_for_llava_next(
    hf_config: LlavaLikeConfig,
    quant_config: Optional[QuantizationConfig],
    *,
    require_post_norm: Optional[bool] = None,
    prefix: str = "",
):
    vision_config = hf_config.vision_config

    # Initialize the vision tower only up to the deepest required feature layer
    num_hidden_layers = _get_num_hidden_layers(hf_config)

    if isinstance(vision_config, CLIPVisionConfig):
        return CLIPVisionModel(
            vision_config,
            quant_config=quant_config,
            num_hidden_layers_override=num_hidden_layers,
            require_post_norm=require_post_norm,
            prefix=prefix,
        )

    msg = f"Unsupported vision config: {type(vision_config)}"
    raise NotImplementedError(msg)

import time
import torch_tpu
import os
@MULTIMODAL_REGISTRY.register_processor(LlavaNextMultiModalProcessor,
                                        info=LlavaNextProcessingInfo,
                                        dummy_inputs=LlavaDummyInputsBuilder)
class LlavaNextForConditionalGeneration(nn.Module, SupportsMultiModal,
                                        SupportsPP):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        vision_feature_layer = config.vision_feature_layer
        # Determine the layer up to which we will initialize the vision tower
        if isinstance(vision_feature_layer, int):
            vision_hidden_size = config.vision_config.hidden_size
            self.feature_sample_layers = None
        # Used for multimodal granite models to control encoder outputs
        elif isinstance(vision_feature_layer, (list, tuple)):
            vision_hidden_size = config.vision_config.hidden_size * len(
                vision_feature_layer)
            self.feature_sample_layers = vision_feature_layer
        else:
            raise TypeError(
                f"vision_layer_feature type: {type(vision_feature_layer)}"
                " is not supported")

        self.config = config
        self.multimodal_config = multimodal_config

        # TODO: Optionally initializes this for supporting embeddings.
        self.vision_tower = init_vision_tower_for_llava_next(
            config,
            quant_config,
            require_post_norm=False,
            prefix=maybe_prefix(prefix, "vision_tower"))
        self.image_newline = nn.Parameter(
            torch.empty(config.text_config.hidden_size))
        self.multi_modal_projector = SophLlavaNextMultiModalProjector(
            vision_hidden_size=vision_hidden_size,
            text_hidden_size=config.text_config.hidden_size,
            projector_hidden_act=config.projector_hidden_act,
            multimodal_projector_bias=config.multimodal_projector_bias)

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)

    @cached_property
    def sampler(self):
        if hasattr(self.language_model, "sampler"):
            return self.language_model.sampler

        return get_sampler()

    def _validate_image_sizes(self, data: torch.Tensor) -> torch.Tensor:
        expected_dims = (2, )

        def _validate_shape(d: torch.Tensor):
            actual_dims = tuple(d.shape)

            if actual_dims != expected_dims:
                expected_expr = str(expected_dims)
                raise ValueError(
                    f"The expected shape of image sizes per image per batch "
                    f"is {expected_expr}. You supplied {tuple(d.shape)}.")

        for d in data:
            _validate_shape(d)

        return data

    def _validate_pixel_values(
        self, data: Union[torch.Tensor, List[torch.Tensor]]
    ) -> Union[torch.Tensor, List[torch.Tensor]]:

        h = w = self.config.vision_config.image_size
        expected_dims = (3, h, w)

        def _validate_shape(d: torch.Tensor):
            actual_dims = tuple(d.shape[1:])

            if actual_dims != expected_dims:
                expected_expr = ("num_patches", *map(str, expected_dims))
                raise ValueError(
                    "The expected shape of pixel values per image per batch "
                    f"is {expected_expr}. You supplied {tuple(d.shape)}.")

        for d in data:
            _validate_shape(d)

        return data

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[LlavaNextImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_sizes = kwargs.pop("image_sizes", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            if not isinstance(pixel_values, (torch.Tensor, list)):
                raise ValueError("Incorrect type of pixel values. "
                                 f"Got type: {type(pixel_values)}")

            if not isinstance(image_sizes, (torch.Tensor, list)):
                raise ValueError("Incorrect type of image sizes. "
                                 f"Got type: {type(image_sizes)}")

            return LlavaNextImagePixelInputs(
                type="pixel_values",
                data=self._validate_pixel_values(
                    flatten_bn(pixel_values)),
                image_sizes=self._validate_image_sizes(
                    flatten_bn(image_sizes, concat=True)),
            )

        if image_embeds is not None:
            if not isinstance(image_embeds, torch.Tensor):
                raise ValueError("Incorrect type of image embeds. "
                                 f"Got type: {type(image_embeds)}")

            return LlavaNextImageEmbeddingInputs(
                type="image_embeds",
                data=flatten_bn(image_embeds),
            )

        raise AssertionError("This line should be unreachable.")

    def _select_image_features(self, image_features: torch.Tensor, *,
                               strategy: str) -> torch.Tensor:
        # Copied from https://github.com/huggingface/transformers/blob/39c3c0a72af6fbda5614dde02ff236069bb79827/src/transformers/models/llava/modeling_llava.py#L421  # noqa
        if strategy == "default":
            return image_features[:, 1:]
        elif strategy == "full":
            return image_features

        raise ValueError(f"Unexpected select feature strategy: {strategy}")

    def _image_pixels_to_features(
        self,
        vision_tower: Union[CLIPVisionModel],
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:

        # NOTE: we skip the step to select the vision feature layer since
        # this is already done inside the vision tower

        pixel_values = pixel_values.contiguous()
        image_features = vision_tower(
            pixel_values, feature_sample_layers=self.feature_sample_layers)

        return self._select_image_features(
            image_features,
            strategy=self.config.vision_feature_select_strategy,
        )

    # Based on: https://github.com/haotian-liu/LLaVA/blob/main/llava/model/llava_arch.py
    def _merge_image_patch_embeddings(self, image_size: torch.Tensor,
                                      patch_embeddings: torch.Tensor, *,
                                      strategy: str) -> torch.Tensor:
        if strategy == "flat":
            return patch_embeddings.flatten(0, 1)

        if strategy.startswith("spatial"):
            height = width = self.config.vision_config.image_size \
                // self.config.vision_config.patch_size

            base_patch_embeds = patch_embeddings[0]
            if height * width != base_patch_embeds.shape[0]:
                raise ValueError(
                    "The number of patches is not consistent with the "
                    "image size.")

            if patch_embeddings.shape[0] > 1:
                other_patch_embeds = patch_embeddings[1:]

                # Move to CPU to avoid floating-point errors
                orig_height, orig_width = image_size.tolist()

                num_patch_height, num_patch_width = get_anyres_image_grid_shape(
                    (orig_height, orig_width),
                    self.config.image_grid_pinpoints,
                    self.config.vision_config.image_size,
                )
                num_patches = num_patch_height * num_patch_width

                # Image patches might be padded for batch processing
                other_patch_embeds = other_patch_embeds[:num_patches] \
                    .view(num_patch_height, num_patch_width, height, width, -1)

                if "unpad" in strategy:
                    other_patch_embeds = other_patch_embeds.permute(0, 2, 1, 3, 4).contiguous()
                    other_patch_embeds = other_patch_embeds.flatten(0, 1).flatten(1, 2)
                    other_patch_embeds = unpad_image(other_patch_embeds,
                                                     (orig_height, orig_width))

                    other_patch_embeds = torch.cat(
                        (
                            other_patch_embeds,
                            self.image_newline[None, None, :].expand(
                                other_patch_embeds.shape[0], 1, other_patch_embeds.shape[-1]
                            ),
                        ),
                        dim=1,
                    )
                    other_patch_embeds = other_patch_embeds.flatten(0, 1)
                else:
                    other_patch_embeds = other_patch_embeds \
                        .permute(0, 2, 1, 3, 4).contiguous() \
                        .flatten(0, 3)

                merged_patch_embeddings = torch.cat(
                    (base_patch_embeds, other_patch_embeds), dim=0)
            else:
                if "unpad" in strategy:
                    merged_patch_embeddings = torch.cat(
                        (base_patch_embeds,
                         self.image_newline[None] \
                            .to(base_patch_embeds.device)
                    ), dim=0)
                else:
                    merged_patch_embeddings = base_patch_embeds

            return merged_patch_embeddings

        raise ValueError(f"Unexpected patch merge strategy: {strategy}")

    def _process_image_pixels(
        self,
        inputs: LlavaNextImagePixelInputs,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
        assert self.vision_tower is not None

        pixel_values = inputs["data"]
        if isinstance(pixel_values, torch.Tensor):
            b, num_patches, c, h, w = pixel_values.shape
            stacked_pixel_values = pixel_values.view(b * num_patches, c, h, w)
            stacked_image_features = self._image_pixels_to_features(
                self.vision_tower, stacked_pixel_values)

            mlp_out = torch.empty(stacked_image_features.shape[0], stacked_image_features.shape[1], self.multi_modal_projector.w2.shape[1], dtype=self.multi_modal_projector.w1.dtype, device = stacked_image_features.device)
            stacked_patch_embeddings = self.multi_modal_projector(
                mlp_out,
                stacked_image_features)

            return stacked_patch_embeddings.view(
                b, num_patches, *stacked_patch_embeddings.shape[1:])

        num_patches_per_batch = [v.shape[0] for v in pixel_values]
        stacked_pixel_values = torch.cat(pixel_values)
        stacked_image_features = self._image_pixels_to_features(
            self.vision_tower, stacked_pixel_values)
        mlp_out = torch.empty(stacked_image_features.shape[0], stacked_image_features.shape[1], self.multi_modal_projector.w2.shape[1], dtype=self.multi_modal_projector.w1.dtype, device = stacked_image_features.device)

        return torch.split(self.multi_modal_projector(mlp_out, stacked_image_features),
                           num_patches_per_batch)

    def _process_image_input(
        self,
        image_input: LlavaNextImageInputs,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        if image_input["type"] == "image_embeds":
            return [image_input["data"]]

        patch_embeddings = self._process_image_pixels(image_input)

        image_sizes = image_input.get("image_sizes")

        if image_sizes is None:
            batch_size = len(image_input["data"])
            vision_config = self.config.vision_config
            default_height = default_width = vision_config.image_size
            image_sizes = torch.as_tensor([[default_height, default_width]
                                           for _ in range(batch_size)])

        return [
            self._merge_image_patch_embeddings(image_sizes[i],
                                               patch_features_batch,
                                               strategy="spatial_unpad")
            for i, patch_features_batch in enumerate(patch_embeddings)
        ]
    
    def merge_input_ids_with_image_features(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        image_features: torch.Tensor,
    ):
        """In place merges in vision_embeddings with inputs_embeds."""
        mask = input_ids == self.config.image_token_index
        # Let's pray we have enabled enough slots !
        try:
            inputs_embeds[mask] = image_features.view(-1, image_features.shape[-1])
            """
            #Note: replace the above line with following code to avoid moving data to cpu!
            start_idx = (mask == True).argmax().item()
            insert_image_feature = image_features.view(-1, image_features.shape[-1])
            logger.info(f"image_features.view(-1, image_features.shape[-1]).shape: {image_features.view(-1, image_features.shape[-1]).shape}")
            end_idx = start_idx + insert_image_feature.shape[0]
            inputs_embeds = torch.cat((inputs_embeds[:start_idx], insert_image_feature, inputs_embeds[end_idx:]), dim=0)
            #inputs_embeds[start_idx:end_idx] = insert_image_feature
            """
        except Exception as e:
            raise RuntimeError(
                f"Cannot fill images right now. If error happens at warmup, make sure you have enough `--max-input-tokens`  to handle images. If error happens at regular runtime, please fill in an issue: {e}"
            )
        return inputs_embeds
    
    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def get_multimodal_embeddings(
            self, **kwargs) -> Optional[NestedTensors]:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None
        vision_embeddings = self._process_image_input(image_input)
        return vision_embeddings

    def _merge_input_ids_with_image_features(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        image_features: torch.Tensor,
    ):
        """In place merges in vision_embeddings with inputs_embeds."""
        image_features = torch.stack(image_features, dim=0)
        mask = input_ids == self.config.image_token_index
        # Let's pray we have enabled enough slots !
        try:
            # logger.info(f"***inputs_embeds.shape: {inputs_embeds.shape}, mask.shape: {mask.shape}, image_features.shape: {image_features.shape}")
            
            inputs_embeds[mask] = image_features.view(-1, image_features.shape[-1])
            """
            #Note: replace the above line with following code to avoid moving data to cpu!
            start_idx = (mask == True).argmax().item()
            insert_image_feature = image_features.view(-1, image_features.shape[-1])
            logger.info(f"image_features.view(-1, image_features.shape[-1]).shape: {image_features.view(-1, image_features.shape[-1]).shape}")
            end_idx = start_idx + insert_image_feature.shape[0]
            inputs_embeds = torch.cat((inputs_embeds[:start_idx], insert_image_feature, inputs_embeds[end_idx:]), dim=0)
            #inputs_embeds[start_idx:end_idx] = insert_image_feature
            """
        except Exception as e:
            raise RuntimeError(
                f"Cannot fill images right now. If error happens at warmup, make sure you have enough `--max-input-tokens`  to handle images. If error happens at regular runtime, please fill in an issue: {e}"
            )
        return inputs_embeds
    
    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: list[torch.Tensor] = None,
    ) -> torch.Tensor:

        if multimodal_embeddings is None \
            or len(multimodal_embeddings) == 0:
            return self.language_model.get_input_embeddings(input_ids)

        inputs_embeds = self.language_model.model.get_input_embeddings(input_ids)
        inputs_embeds = _embed_multimodal(
            input_ids,
            self.config.image_token_index,
            inputs_embeds,
            multimodal_embeddings,
        )

        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        """Run forward pass for LlaVA-NeXT.

        One key thing to understand is the `input_ids` already accounts for the
        positions of the to-be-inserted image embeddings.

        Concretely, consider a text prompt:
        `"A chat between a curious human and an artificial intelligence
        assistant. The assistant gives helpful, detailed, and polite answers to
        the human's questions.
        USER: <image>\\nWhat is shown in this image? ASSISTANT:"`.

        Tokenizer outputs:
        `[1, 319, 13563, 1546, 263, 12758, 5199, 322, 385, 23116, 21082, 20255,
        29889, 450, 20255, 4076, 8444, 29892, 13173, 29892, 322, 1248, 568,
        6089, 304, 278, 5199, 29915, 29879, 5155, 29889, 3148, 1001, 29901,
        29871, 32000, 13, 5618, 338, 4318, 297, 445, 1967, 29973, 319, 1799,
        9047, 13566, 29901]`.

        To reserve space in KV cache, we have to insert placeholder tokens
        before they are inputted to the model, so the input processor prepends
        additional image tokens (denoted as `32000`), resulting in:
        `[1, 319, 13563, 1546, 263, 12758, 5199, 322, 385, 23116, 21082, 20255,
        29889, 450, 20255, 4076, 8444, 29892, 13173, 29892, 322, 1248, 568,
        6089, 304, 278, 5199, 29915, 29879, 5155, 29889, 3148, 1001, 29901,
        29871, 32000, ..., 32000, 13, 5618, 338, 4318, 297, 445, 1967, 29973,
        319, 1799, 9047, 13566, 29901]`.

        Unlike in LLaVA-1.5, the number of image tokens inputted to the language
        model depends on the original size of the input image. Including the
        original image token in the input, the required number of image tokens
        is given by :func:`get_llava_next_image_feature_size`.

        This way, the `positions` and `attn_metadata` are consistent
        with the `input_ids`.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            pixel_values: The pixels in each grid patch for each input image.
            image_sizes: The original `(height, width)` for each input image.

        See also:
            :class:`LlavaNextImageInputs`
        """

        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility.
        elif inputs_embeds is None:
            vision_embeddings = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids,
                                                      vision_embeddings)
            input_ids = None
        elif inputs_embeds is not None:
            input_ids = None

        hidden_states = self.language_model.model(input_ids,
                                                  positions,
                                                  kv_caches,
                                                  attn_metadata,
                                                  intermediate_tensors,
                                                  inputs_embeds=inputs_embeds)

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states,
                                                  sampling_metadata)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        return self.language_model.sample(logits, sampling_metadata)

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)