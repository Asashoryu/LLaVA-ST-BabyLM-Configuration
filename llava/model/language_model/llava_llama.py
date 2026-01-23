#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig

from torch.nn import CrossEntropyLoss


# We use the standard HF Llama classes and extend them with LLaVA-specific
# multimodal functionality implemented in `llava_arch`.
# The file defines lightweight wrappers so HuggingFace `AutoModel*` APIs
# can construct LLaVA-capable models that accept images + text.
from transformers import LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaConfig(LlamaConfig):
    """Configuration for LLaVA-wrapped Llama models.

    Inherits all standard `LlamaConfig` options and adds a few convenient
    generation defaults so the model config is ready for inference out of
    the box.
    """
    model_type = "llava_llama"
    # Generation defaults stored on the config for convenience.
    temperature: float = 0.0  # reset to 0.0, previously 0.9 for Vicuna
    max_new_tokens: int = 1024
    do_sample: bool = False
    top_p: Optional[float] = None
    # rope_scaling: Optional[dict] = {}


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    """Combined model class.

    This class uses multiple inheritance to merge LLaVA's multimodal helpers
    (from `LlavaMetaModel`) with the Llama transformer implementation
    (`LlamaModel`). The result is a backbone that understands image tokens
    and the associated spatial/temporal embeddings.
    """
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        # Cooperative initialization via the MRO ensures both mixins are set up.
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    """Top-level model class exposed to HuggingFace APIs.

    Behaves like a standard `LlamaForCausalLM` but supports multimodal inputs
    (images + text). It wraps a `LlavaLlamaModel` backbone and a language-model
    head. Custom forwarding logic prepares multimodal embeddings before
    delegating to the transformer.
    """
    config_class = LlavaConfig

    def __init__(self, config):
        # Initialize HF wrapper which prepares generation utilities.
        LlamaForCausalLM.__init__(self, config)

        # Ensure the model type is set so HF serialization/registration works
        config.model_type = "llava_llama"

        # Build the combined backbone and an LM head
        self.model = LlavaLlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Final HF initialization step (weight init, tie weights, etc.)
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = None,
        cache_position=None,
        variables: Optional[list] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        # Prepare multimodal inputs when `inputs_embeds` is not already provided.
        # This converts images (or video frames) into a sequence of embeddings
        # and computes the corresponding position/attention tensors so the
        # Llama backbone can consume them as if they were token embeddings.
        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal_video(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes)

        if dpo_forward:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            return logits, labels

        else:
            # Transformers' forward accepts either `input_ids` or `inputs_embeds`, not both.
            # If `inputs_embeds` has been prepared (e.g., after multimodal processing), pass it
            # and clear `input_ids` to avoid conflicts.
            if inputs_embeds is not None:
                return super().forward(
                    input_ids=None,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
            else:
                return super().forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        modalities = kwargs.pop("modalities", None) if "modalities" in kwargs and modalities is None else modalities
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        # During generation we accept `images` as well; if present convert them
        # into `inputs_embeds` using the same multimodal preprocessing so the
        # generation routine runs on embeddings rather than raw token ids.
        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal_video(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
        else:
            # Text-only generation: obtain token embeddings from the embed layer.
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


# Register the custom config and model implementations with HuggingFace so
# calls like `AutoModelForCausalLM.from_pretrained(..., config=...)` will
# instantiate the LLaVA-aware classes when appropriate.
AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
