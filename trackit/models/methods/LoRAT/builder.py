import torch
from trackit.miscellanies.torch.dtype import set_default_dtype
from trackit.models import ModelBuildingContext, ModelImplSuggestions
from trackit.models.backbone.builder import build_backbone
from trackit.miscellanies.pretty_format import pretty_format
import torch.nn as nn
import safetensors.torch
from .sample_data_generator import build_sample_input_data_generator


def create_LoRAT_build_context(config: dict):
    print('LoRAT model config:\n' + pretty_format(config['model']))
    return ModelBuildingContext(lambda impl_advice: build_LoRAT_model(config, impl_advice),
                             lambda impl_advice: get_LoRAT_build_string(config['model']['type'], impl_advice),
                             build_sample_input_data_generator(config))


def build_LoRAT_model(config: dict, model_impl_suggestions: ModelImplSuggestions):
    model_config = config['model']
    common_config = config['common']
    backbone = build_backbone(model_config['backbone'],
                              torch_jit_trace_compatible=model_impl_suggestions.torch_jit_trace_compatible)
    model_type = model_config['type']
    
    if model_type == 'dinov2':
        from .lorat import LoRAT_DINOv2
        model = LoRAT_DINOv2(backbone, common_config['template_feat_size'],
                                common_config['search_region_feat_size'],
                                model_config.get('target_landmark_relation'))
    else:
        raise NotImplementedError(f"Model type '{model_type}' is not supported.")
    
    from .funcs.vit_backbone_freeze import freeze_vit_backbone_
    from .funcs.vit_lora_utils import enable_lora_
    freeze_vit_backbone_(model)
    enable_lora_(
        model,
        model_config['lora']['r'],
        model_config['lora']['alpha'],
        model_config['lora']['dropout'],
        model_config['lora']['use_rslora'],
    )
    return model


def get_LoRAT_build_string(model_type: str, model_impl_suggestions: ModelImplSuggestions):
    build_string = 'LORAT_lora'
    if model_impl_suggestions.optimize_for_inference:
        build_string += '_merged'
    if model_impl_suggestions.torch_jit_trace_compatible:
        build_string += '_disable_flash_attn'
    return build_string

def load_model_weight(model: nn.Module, weight_path: str, use_safetensors=False, strict=True):
    if use_safetensors:
        return safetensors.torch.load_model(model, weight_path, strict=strict)
    else:
        return model.load_state_dict(torch.load(weight_path, map_location='cpu', weights_only=False), strict=strict)
