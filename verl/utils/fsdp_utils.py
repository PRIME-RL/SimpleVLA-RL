# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy
from transformers.trainer_pt_utils import get_module_class_from_name
import torch
import torch.nn as nn
from verl.utils.vla_utils.openvla_oft.modeling_prismatic import  PrismaticProjector

def init_fn(x: torch.nn.Module):
    if not torch.distributed.get_rank() == 0:
        x = x.to_empty(device=torch.cuda.current_device(), recurse=False)
        torch.cuda.empty_cache()
    return x


def get_init_weight_context_manager(use_meta_tensor=True):
    from accelerate import init_empty_weights
    cpu_init_weights = lambda: torch.device('cpu')
    if use_meta_tensor:
        init_context = init_empty_weights if torch.distributed.get_rank() != 0 else cpu_init_weights
    else:
        init_context = cpu_init_weights
    return init_context


# Copyright 2020-present the HuggingFace Inc. team.
# Adapted from https://github.com/huggingface/transformers/src/transformers/trainer.py
def get_fsdp_wrap_policy(module, config=None):
    # TODO
    if config is None:
        config = {}

    if config.get('disable', False):
        return None

    default_transformer_cls_names_to_wrap = getattr(module, "_no_split_modules", None)
    fsdp_transformer_layer_cls_to_wrap = config.get("transformer_layer_cls_to_wrap",
                                                    default_transformer_cls_names_to_wrap)
    min_num_params = config.get('min_num_params', 0)
    auto_wrap_policy = None
    if min_num_params > 0:
        auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=min_num_params)
    elif fsdp_transformer_layer_cls_to_wrap is not None:
        transformer_cls_to_wrap = set()
        for layer_class in fsdp_transformer_layer_cls_to_wrap:
            transformer_cls = get_module_class_from_name(module, layer_class)
            if transformer_cls is None:
                raise Exception("Could not find the transformer layer class to wrap in the model.")
            else:
                transformer_cls_to_wrap.add(transformer_cls)

        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            # Transformer layer class to wrap
            transformer_layer_cls=transformer_cls_to_wrap,
        )
    return auto_wrap_policy

def get_fsdp_wrap_policy_vla(module, config=None, is_lora=False):
    
    from timm.models.vision_transformer import Block, VisionTransformer
    from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy, transformer_auto_wrap_policy, lambda_auto_wrap_policy
    vit_wrap_policy = functools.partial(_module_wrap_policy, module_classes={VisionTransformer})
    transformer_block_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})
    vision_fsdp_wrapping_policy = functools.partial(_or_policy, policies=[vit_wrap_policy, transformer_block_policy])

    # transformer_block_policy = functools.partial(
    #         transformer_auto_wrap_policy, transformer_layer_cls={self.transformer_layer_cls}
    #     )\
        
    #default_transformer_cls_names_to_wrap = getattr(module, "_no_split_modules", None)
    default_transformer_cls_names_to_wrap = getattr(module.language_model, "_no_split_modules", None)
    
    fsdp_transformer_layer_cls_to_wrap = default_transformer_cls_names_to_wrap

    llm_wrap_policy = None
    
    if fsdp_transformer_layer_cls_to_wrap is not None:
        transformer_cls_to_wrap = set()
        for layer_class in fsdp_transformer_layer_cls_to_wrap:
            print("layer_class is :", layer_class)
            transformer_cls = get_module_class_from_name(module, layer_class)
            if transformer_cls is None:
                raise Exception("Could not find the transformer layer class to wrap in the model.")
            else:
                transformer_cls_to_wrap.add(transformer_cls)

        llm_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            # Transformer layer class to wrap
            transformer_layer_cls=transformer_cls_to_wrap,
        )
    print("llm_wrap_policy:",llm_wrap_policy)
    assert llm_wrap_policy is not None

      




    # Get Prismatic Wrapping Policy =>> just a module wrapping policy around `self.projector`
    # prismatic_fsdp_wrapping_policy = functools.partial(
    #     _module_wrap_policy,
    #     module_classes={LinearProjector, MLPProjector, FusedMLPProjector},
    # )
    prismatic_fsdp_wrapping_policy = functools.partial(
        _module_wrap_policy,
        module_classes={PrismaticProjector},
    )

    
    # Add lambda policy for LoRA modules if is_lora is True
    if is_lora:
        def lambda_policy_fn(module):
            return bool(
                len(list(module.named_children())) == 0
                and getattr(module, "weight", None) is not None
                and module.weight.requires_grad
            )
        lambda_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)


    # Return union (_or_) over constituent policies
    #   => Note: there is *not* a fall-through policy; any module that isn't covered by the above constituents will
    #            automatically be folded into the root VLM FSDP instance.
    if is_lora:
        vla_policies=[
                vision_fsdp_wrapping_policy,
                llm_wrap_policy,
                prismatic_fsdp_wrapping_policy,
                lambda_policy
            ]
    else:
        vla_policies=[
            vision_fsdp_wrapping_policy,
            llm_wrap_policy,
            prismatic_fsdp_wrapping_policy,
        ]
    
    return functools.partial(
        _or_policy,
        policies=vla_policies
    )


def offload_fsdp_grad(module):
    for _, param in module.named_parameters():
        if param.grad is not None:
            param.grad = param.grad.to("cpu", non_blocking=True)
    torch.cuda.empty_cache()


def load_fsdp_grad(module, device_id):
    for _, param in module.named_parameters():
        if param.grad is not None:
            param.grad = param.grad.to(device_id, non_blocking=True)
    torch.cuda.empty_cache()


def offload_fsdp_param_and_grad(module, offload_grad=False):
    for _, param in module.named_parameters():
        if hasattr(param, "_local_shard"):
            param._local_shard = param._local_shard.to("cpu", non_blocking=True)
        param.data = param.data.to('cpu', non_blocking=True)
        if offload_grad and param.grad is not None:
            param.grad = param.grad.to("cpu", non_blocking=True)
    torch.cuda.empty_cache()


def load_fsdp_param_and_grad(module, device_id, load_grad=False):
    for _, param in module.named_parameters():
        if hasattr(param, "_local_shard"):
            param._local_shard = param._local_shard.to(device_id, non_blocking=True)
        param.data = param.data.to(device_id, non_blocking=True)
        if load_grad and param.grad is not None:
            param.grad = param.grad.to(device_id, non_blocking=True)
    torch.cuda.empty_cache()


def offload_fsdp_optimizer(optimizer):
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to("cpu", non_blocking=True)
    torch.cuda.empty_cache()


def load_fsdp_optimizer(optimizer, device_id):
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device_id, non_blocking=True)
    torch.cuda.empty_cache()
