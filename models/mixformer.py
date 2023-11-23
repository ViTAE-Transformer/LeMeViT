"""
MixFormer impl.

author: Jiang Wentao
github: 
email: 

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import math
from collections import OrderedDict
from functools import partial
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fairscale.nn.checkpoint import checkpoint_wrapper
from timm.models import register_model
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import _cfg

from ops.bra_legacy import BiLevelRoutingAttention

from ._common import Attention, AttentionLePE, DWConv

# from positional_encodings import PositionalEncodingPermute2D, Summer
# from siren_pytorch import SirenNet
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

try:
    import xformers.ops as xops

    has_xformers = True
except ImportError:
    has_xformers = False

from .layers import PatchEmbed, PatchMerging, Conv2d_BN, LayerNorm

class Attention(nn.Module):
    """Patch-to-Cluster Attention Layer"""
    
    def __init__(
        self,
        dim,
        num_heads,
        attn_drop=0.0,
        proj_drop=0.0,
        **kwargs,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} not divisible by num_heads {num_heads}"
        self.num_heads = num_heads

        self.use_xformers = has_xformers and (dim // num_heads) % 32 == 0

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.attn_drop = attn_drop

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attn_viz = nn.Identity() 

    def forward(self, x):
        if self.use_xformers:
            q = self.q(x)  # B N C
            k = self.k(x)  # B N C
            v = self.v(x)
            q = rearrange(q, "B N (h d) -> B N h d", h=self.num_heads)
            k = rearrange(k, "B N (h d) -> B N h d", h=self.num_heads)
            v = rearrange(v, "B N (h d) -> B N h d", h=self.num_heads)

            x = xops.memory_efficient_attention(q, k, v)  # B N h d
            x = rearrange(x, "B N h d -> B N (h d)")

            x = self.proj(x)
        else:
            x = rearrange(x, "B N C -> N B C")

            x, attn = F.multi_head_attention_forward(
                query=x,
                key=x,
                value=x,
                embed_dim_to_check=x.shape[-1],
                num_heads=self.num_heads,
                q_proj_weight=self.q.weight,
                k_proj_weight=self.k.weight,
                v_proj_weight=self.v.weight,
                in_proj_weight=None,
                in_proj_bias=torch.cat([self.q.bias, self.k.bias, self.v.bias]),
                bias_k=None,
                bias_v=None,
                add_zero_attn=False,
                dropout_p=self.attn_drop,
                out_proj_weight=self.proj.weight,
                out_proj_bias=self.proj.bias,
                use_separate_proj_weight=True,
                training=self.training,
                need_weights=not self.training,  # for visualization
                average_attn_weights=False,
            )

            x = rearrange(x, "N B C -> B N C")

            if not self.training:
                attn = self.attn_viz(attn)

        x = self.proj_drop(x)

        return x

class MixAttention(nn.Module):
    """Patch-to-Cluster Attention Layer"""
    
    def __init__(
        self,
        dim,
        num_heads,
        scale = None,
        bias = False,
        attn_drop=0.0,
        proj_drop=0.0,
        **kwargs,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} not divisible by num_heads {num_heads}"
        self.num_heads = num_heads

        self.use_xformers = has_xformers and (dim // num_heads) % 32 == 0

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.attn_drop = attn_drop

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attn_viz = nn.Identity() 

    def forward(self, x):
        if self.use_xformers:
            q = self.q(x)  # B N C
            k = self.k(x)  
            v = self.v(x)

            q = rearrange(q, "B N (h d) -> B N h d", h=self.num_heads)
            k = rearrange(k, "B N (h d) -> B N h d", h=self.num_heads)
            v = rearrange(v, "B N (h d) -> B N h d", h=self.num_heads)

            x = xops.memory_efficient_attention(q, k, v)  # B N h d
            x = rearrange(x, "B N h d -> B N (h d)")

            x = self.proj(x)
        else:
            x = rearrange(x, "B N C -> N B C")

            x = rearrange(x, 'n h w c -> n (h w) c')
            _, H, W, _ = x.size()
            #######################################
            B, N, C = x.shape        
            q = self.q(x)  # B N C
            k = self.k(x)  # B N C
            v = self.v(x)
            q = rearrange(q, "B N (h d) -> B N h d", h=self.num_heads)
            k = rearrange(k, "B N (h d) -> B N h d", h=self.num_heads)
            v = rearrange(v, "B N (h d) -> B N h d", h=self.num_heads)

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            #######################################

            x = rearrange(x, 'n (h w) c -> n h w c', h=H, w=W)

            x = rearrange(x, "N B C -> B N C")

            if not self.training:
                attn = self.attn_viz(attn)

        x = self.proj_drop(x)

        return x


class MixBlock(nn.Module):
    def __init__(self, dim, 
                 attn_drop, proj_drop, drop_path=0., 
                 layer_scale_init_value=-1, num_heads=8, qk_dim=None, mlp_ratio=4, mlp_dwconv=False,
                 cpe_ks=3, pre_norm=True):
        super().__init__()
        qk_dim = qk_dim or dim

        # modules
        if cpe_ks > 0:
            self.pos_embed = nn.Conv2d(dim, dim,  kernel_size=cpe_ks, padding=1, groups=dim)
        else:
            self.pos_embed = lambda x: 0
        self.norm1 = nn.LayerNorm(dim, eps=1e-6) # important to avoid attention collapsing
        self.attn = MixAttention(dim=dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=proj_drop)

        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, int(mlp_ratio*dim)),
                                 DWConv(int(mlp_ratio*dim)) if mlp_dwconv else nn.Identity(),
                                 nn.GELU(),
                                 nn.Linear(int(mlp_ratio*dim), dim)
                                )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # tricks: layer scale & pre_norm/post_norm
        if layer_scale_init_value > 0:
            self.use_layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        else:
            self.use_layer_scale = False
        self.pre_norm = pre_norm
            

    def forward(self, x):
        """
        x: NCHW tensor
        """
        
        _, C, H, W = x.shape
        # conv pos embedding
        x = x + self.pos_embed(x)
        # permute to NHWC tensor for attention & mlp
        x = rearrange(x, "N C H W -> N (H W) C")
        # x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)

        # attention & mlp
        if self.pre_norm:
            if self.use_layer_scale:
                x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x))) # (N, H, W, C)
                x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x))) # (N, H, W, C)
            else:
                x = x + self.drop_path(self.attn(self.norm1(x))) # (N, H, W, C)
                x = x + self.drop_path(self.mlp(self.norm2(x))) # (N, H, W, C)
        else: # https://kexue.fm/archives/9009
            if self.use_layer_scale:
                x = self.norm1(x + self.drop_path(self.gamma1 * self.attn(x))) # (N, H, W, C)
                x = self.norm2(x + self.drop_path(self.gamma2 * self.mlp(x))) # (N, H, W, C)
            else:
                x = self.norm1(x + self.drop_path(self.attn(x))) # (N, H, W, C)
                x = self.norm2(x + self.drop_path(self.mlp(x))) # (N, H, W, C)

        x = rearrange(x, "N (H W) C -> N C H W",H=H,W=W)
        # permute back
        # x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        return x


class MixFormer(nn.Module):
    def __init__(self, 
                 depth=[3, 4, 8, 3], 
                 in_chans=3, 
                 num_classes=1000, 
                 embed_dim=[64, 128, 320, 512],
                 head_dim=64, 
                 mlp_ratios=[4, 4, 4, 4],
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop=0., 
                 drop_path_rate=0.,
                 # <<<------
                 qk_dims=None,
                 cpe_ks=3,
                 pre_norm=True,
                 mlp_dwconv=False,
                 representation_size=None,
                 layer_scale_init_value=-1,
                 use_checkpoint_stages=[],
                 # ------>>>
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        qk_dims = qk_dims or embed_dim
        
        ############ downsample layers (patch embeddings) ######################
        self.downsample_layers = nn.ModuleList()
        # NOTE: uniformer uses two 3*3 conv, while in many other transformers this is one 7*7 conv 
        stem = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim[0] // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(embed_dim[0] // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim[0] // 2, embed_dim[0], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(embed_dim[0]),
        )

        if use_checkpoint_stages:
            stem = checkpoint_wrapper(stem)
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.Conv2d(embed_dim[i], embed_dim[i+1], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.BatchNorm2d(embed_dim[i+1])
            )
            if use_checkpoint_stages:
                downsample_layer = checkpoint_wrapper(downsample_layer)
            self.downsample_layers.append(downsample_layer)
        ##########################################################################

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        nheads= [dim // head_dim for dim in qk_dims]
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[MixBlock(dim=embed_dim[i], 
                           attn_drop=attn_drop, proj_drop=drop_rate,
                           drop_path=dp_rates[cur + j],
                           layer_scale_init_value=layer_scale_init_value,
                           num_heads=nheads[i],
                           qk_dim=qk_dims[i],
                           mlp_ratio=mlp_ratios[i],
                           mlp_dwconv=mlp_dwconv,
                           cpe_ks=cpe_ks,
                           pre_norm=pre_norm
                    ) for j in range(depth[i])],
            )
            if i in use_checkpoint_stages:
                stage = checkpoint_wrapper(stage)
            self.stages.append(stage)
            cur += depth[i]

        ##########################################################################
        self.norm = nn.BatchNorm2d(embed_dim[-1])
        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = nn.Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = self.norm(x)
        x = self.pre_logits(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x.flatten(2).mean(-1)
        x = self.head(x)
        return x



#################### model variants #######################


# model_urls = {
#     "biformer_tiny_in1k": "https://api.onedrive.com/v1.0/shares/s!AkBbczdRlZvChHEOoGkgwgQzEDlM/root/content",
#     "biformer_small_in1k": "https://api.onedrive.com/v1.0/shares/s!AkBbczdRlZvChHDyM-x9KWRBZ832/root/content",
#     "biformer_base_in1k": "https://api.onedrive.com/v1.0/shares/s!AkBbczdRlZvChHI_XPhoadjaNxtO/root/content",
# }


# https://github.com/huggingface/pytorch-image-models/blob/4b8cfa6c0a355a9b3cb2a77298b240213fb3b921/timm/models/_factory.py#L93

@register_model
def mixformer_tiny(pretrained=False, pretrained_cfg=None,
                  pretrained_cfg_overlay=None, **kwargs):
    model = MixFormer(
        depth=[2, 2, 4, 2],
        embed_dim=[64, 128, 256, 512], 
        head_dim=32,
        mlp_ratios=[3, 3, 3, 3],
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.,
        qk_dims=None,
        cpe_ks=3,
        pre_norm=True,
        mlp_dwconv=False,
        representation_size=None,
        layer_scale_init_value=1,
        use_checkpoint_stages=[],
        **kwargs)
    model.default_cfg = _cfg()

    # if pretrained:
    #     model_key = 'biformer_tiny_in1k'
    #     url = model_urls[model_key]
    #     checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True, file_name=f"{model_key}.pth")
    #     model.load_state_dict(checkpoint["model"])

    return model

@register_model
def mixformer_small(pretrained=False, pretrained_cfg=None,
                  pretrained_cfg_overlay=None, **kwargs):
    model = MixFormer(
        depth=[4, 4, 18, 4],
        embed_dim=[64, 128, 256, 512], 
        head_dim=32,
        mlp_ratios=[3, 3, 3, 3],
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.,
        qk_dims=None,
        cpe_ks=3,
        pre_norm=True,
        mlp_dwconv=False,
        representation_size=None,
        layer_scale_init_value=1,
        use_checkpoint_stages=[],
        **kwargs)
    model.default_cfg = _cfg()

    # if pretrained:
    #     model_key = 'biformer_tiny_in1k'
    #     url = model_urls[model_key]
    #     checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True, file_name=f"{model_key}.pth")
    #     model.load_state_dict(checkpoint["model"])

    return model

# @register_model
# def biformer_small(pretrained=False, pretrained_cfg=None,
#                    pretrained_cfg_overlay=None, **kwargs):
#     model = BiFormer(
#         depth=[4, 4, 18, 4],
#         embed_dim=[64, 128, 256, 512], mlp_ratios=[3, 3, 3, 3],
#         #------------------------------
#         n_win=7,
#         kv_downsample_mode='identity',
#         kv_per_wins=[-1, -1, -1, -1],
#         topks=[1, 4, 16, -2],
#         side_dwconv=5,
#         before_attn_dwconv=3,
#         layer_scale_init_value=-1,
#         qk_dims=[64, 128, 256, 512],
#         head_dim=32,
#         param_routing=False, diff_routing=False, soft_routing=False,
#         pre_norm=True,
#         pe=None,
#         #-------------------------------
#         **kwargs)
#     model.default_cfg = _cfg()

#     if pretrained:
#         model_key = 'biformer_small_in1k'
#         url = model_urls[model_key]
#         checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True, file_name=f"{model_key}.pth")
#         model.load_state_dict(checkpoint["model"])

#     return model


# @register_model
# def biformer_base(pretrained=False, pretrained_cfg=None,
#                   pretrained_cfg_overlay=None, **kwargs):
#     model = BiFormer(
#         depth=[4, 4, 18, 4],
#         embed_dim=[96, 192, 384, 768], mlp_ratios=[3, 3, 3, 3],
#         # use_checkpoint_stages=[0, 1, 2, 3],
#         use_checkpoint_stages=[],
#         #------------------------------
#         n_win=7,
#         kv_downsample_mode='identity',
#         kv_per_wins=[-1, -1, -1, -1],
#         topks=[1, 4, 16, -2],
#         side_dwconv=5,
#         before_attn_dwconv=3,
#         layer_scale_init_value=-1,
#         qk_dims=[96, 192, 384, 768],
#         head_dim=32,
#         param_routing=False, diff_routing=False, soft_routing=False,
#         pre_norm=True,
#         pe=None,
#         #-------------------------------
#         **kwargs)
#     model.default_cfg = _cfg()

#     if pretrained:
#         model_key = 'biformer_base_in1k'
#         url = model_urls[model_key]
#         checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True, file_name=f"{model_key}.pth")
#         model.load_state_dict(checkpoint["model"])

#     return model

