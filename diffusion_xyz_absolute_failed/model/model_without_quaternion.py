# this file shuold implement the main model of diffusion policy
from typing import Tuple, Sequence, Dict, Union, Optional, Callable
import numpy as np
import math
import torch
import torch.nn as nn
import torchvision
import collections
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import data_preprocess
from .conditional_unet1d import ConditionalUnet1D # same folder, use "."

pred_horizon = 16
obs_horizon = 2
action_horizon = 8

# RGB_Vision_Encoder: a ResNet where all BatchNorm layers are modified to GroupNorm layers
# For stable training, especially when the normalization layer is used in conjunction with EMA(Exponential Moving Average)

def get_resnet(name:str, weights=None, **kwargs) -> nn.Module:
    func = getattr(torchvision.models, name) # get actual function corresponds to that name
    resnet = func(weights=weights, **kwargs)

    # remove the final Fully Connected Layer
    resnet.fc = torch.nn.Identity()
    return resnet # the output dim of resnet18 is 512

def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules 
    selected by the predicate with the output of func.
    
    Predicate: return true if the module is to be replaced.
    func: return new module to use.
    """
    if predicate(root_module):
        return func(root_module)
    
    bn_list = [k.split('.') for k, m 
                       in root_module.named_modules(remove_duplicate=True) if predicate(m)]
    for *parent, k in bn_list: 
        # `k` is like `car` in Lisp.
        # it will be either a index (if the parent module is a nn.Sequential) or a string name (if the parent module is reguar nn.Module)
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced.
    bn_list = [k.split('.') for k, m 
               in root_module.named_modules(remove_duplicate=True) if predicate(m)]
    
    assert len(bn_list) == 0
    return root_module


def replace_bn_with_gn(
        root_module: nn.Module,
        features_per_group: int=16) -> nn.Module:
    """
    Repalce all BatchNorm layers with GroupNorm
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module

# # DEPTH_VISION_ENCODER: Use a small CNN to encode the depth image.(DEPRECATED)

# class DepthCNN(torch.nn.Module):
#     def __init__(self, out_dim=512):
#         super().__init__()
#         self.encoder = torch.nn.Sequential(
#             torch.nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
#             torch.nn.ReLU(),
#             torch.nn.MaxPool2d(2),

#             torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
#             torch.nn.ReLU(),
#             torch.nn.MaxPool2d(2),

#             torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), 
#             torch.nn.ReLU(),

#             torch.nn.AdaptiveAvgPool2d((1, 1)), 
#             torch.nn.Flatten(),
#             torch.nn.Linear(128, out_dim)
#         )

#     def forward(self, x):
#         return self.encoder(x)



def nets():

    # Construct ResNet-18 as RGB_vision_encoder:
    rgb_vision_encoder = get_resnet('resnet18')

    # Construct a small CNN as depth_vision_encoder:
    # depth_vision_encoder = DepthCNN()

    # Replace all BatchNorm with GroupNorm to work with EMA, otherwise performance will tank
    rgb_vision_encoder = replace_bn_with_gn(rgb_vision_encoder)

    rgb_vision_feature_dim = 512 # ResNet18 has output dim of 512
    obs_proprio_dim = 3 # xyz q(wxyz) 
    # depth_vision_feature_dim = 512
    # obs_dim = rgb_vision_feature_dim + obs_proprio_dim + depth_vision_feature_dim
    obs_dim = rgb_vision_feature_dim + obs_proprio_dim

    action_dim = 3

    # create noise prediction network object
    noise_pred_net = ConditionalUnet1D(
        input_dim = action_dim, 
        global_cond_dim=obs_dim * obs_horizon
    )

    # the final arch has 3 parts
    return nn.ModuleDict({
        "rgb_encoder": rgb_vision_encoder,
        # "depth_encoder": depth_vision_encoder,
        "noise_pred_net": noise_pred_net
    })