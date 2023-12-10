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

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fairscale.nn.checkpoint import checkpoint_wrapper
from timm.models import register_model
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import _cfg


# from positional_encodings import PositionalEncodingPermute2D, Summer
# from siren_pytorch import SirenNet
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


from mmseg.utils import get_root_logger
from mmcv.runner import _load_checkpoint, BaseModule

from ..builder import BACKBONES


try:
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_kvpacked_func, flash_attn_func
    has_flash_attn = True
except ImportError:
    has_flash_attn = False
    
try:
    import xformers.ops as xops
    has_xformers = True
except ImportError:
    has_xformers = False


has_flash_attn = False
has_xformers = False

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        """
        x: NHWC tensor
        """
        x = x.permute(0, 3, 1, 2) #NCHW
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) #NHWC

        return x

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

class StandardAttention(nn.Module):
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

        self.use_flash_attn = has_flash_attn
        self.use_xformers = has_xformers and (dim // num_heads) % 32 == 0

        self.qkv = nn.Linear(dim, 3 * dim)
        self.attn_drop = attn_drop

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attn_viz = nn.Identity() 

    def forward(self, x):
        if self.use_flash_attn:
            qkv = self.qkv(x)
            qkv = rearrange(qkv, "B N (x h d) -> B N x h d", x=3, h=self.num_heads).contiguous()
            x = flash_attn_qkvpacked_func(qkv)
            x = rearrange(x, "B N h d -> B N (h d)").contiguous()
            x = self.proj(x)
        elif self.use_xformers:
            qkv = self.qkv(x)
            qkv = rearrange(qkv, "B N (x h d) -> x B N h d", x=3, h=self.num_heads).contiguous()
            q, k, v = qkv[0], qkv[1], qkv[2]
            x = xops.memory_efficient_attention(q, k, v)  # B N h d
            x = rearrange(x, "B N h d -> B N (h d)").contiguous()
            x = self.proj(x)
        else:
            qkv = self.qkv(x)
            qkv = rearrange(qkv, "B N (x h d) -> x B h N d", x=3, h=self.num_heads).contiguous()
            q, k, v = qkv[0], qkv[1], qkv[2]
            x = F.scaled_dot_product_attention(q, k, v)  # B N h d
            x = rearrange(x, "B h N d -> B N (h d)").contiguous()
            x = self.proj(x)
        return x

class MixAttention(nn.Module):
    
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
        self.scale = scale or dim**(-0.5)

        self.use_flash_attn = has_flash_attn
        self.use_xformers = has_xformers and (dim // num_heads) % 32 == 0

        self.qkv1 = nn.Linear(dim, 3 * dim)
        self.qkv2 = nn.Linear(dim, 3 * dim)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj_x = nn.Linear(dim, dim)
        self.proj_c = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attn_viz = nn.Identity() 

    def forward(self, x, c):
        B, N, C = x.shape        
        B, M, _ = c.shape 
        scale_x = math.log(M, N) * self.scale
        scale_c = math.log(N, N) * self.scale
        
        if self.use_flash_attn:
            qkv1 = self.qkv1(x)
            qkv1 = rearrange(qkv1, "B N (x h d) -> B N x h d", x=3, h=self.num_heads).contiguous()
            qkv2 = self.qkv2(c)
            qkv2 = rearrange(qkv2, "B M (x h d) -> B M x h d", x=3, h=self.num_heads).contiguous()
            
            q1, kv1 = qkv1[:,:,0], qkv1[:,:,1:]
            q2, kv2 = qkv2[:,:,0], qkv2[:,:,1:]
            
            x = flash_attn_kvpacked_func(q1, kv2, softmax_scale=scale_x)
            x = rearrange(x, "B N h d -> B N (h d)").contiguous()
            x = self.proj_x(x)
            c = flash_attn_kvpacked_func(q2, kv1, softmax_scale=scale_c)
            c = rearrange(c, "B M h d -> B M (h d)").contiguous()
            c = self.proj_c(c)
        elif self.use_xformers:
            qkv1 = self.qkv1(x)
            qkv1 = rearrange(qkv1, "B N (x h d) -> x B N h d", x=3, h=self.num_heads).contiguous()
            qkv2 = self.qkv2(c)
            qkv2 = rearrange(qkv2, "B M (x h d) -> x B M h d", x=3, h=self.num_heads).contiguous()
            
            q1, k1, v1 = qkv1[0], qkv1[1], qkv1[2]
            q2, k2, v2 = qkv2[0], qkv2[1], qkv2[2]
            
            x = xops.memory_efficient_attention(q1, k2, v2, scale=scale_x)  # B N h d
            x = rearrange(x, "B N h d -> B N (h d)").contiguous()
            x = self.proj_x(x)
            c = xops.memory_efficient_attention(q2, k1, v1, scale=scale_c)  # B N h d
            c = rearrange(c, "B M h d -> B M (h d)").contiguous()
            c = self.proj_c(c)
        else:
            qkv1 = self.qkv1(x)
            qkv1 = rearrange(qkv1, "B N (x h d) -> x B h N d", x=3, h=self.num_heads).contiguous()
            qkv2 = self.qkv2(c)
            qkv2 = rearrange(qkv2, "B M (x h d) -> x B h M d", x=3, h=self.num_heads).contiguous()
            
            q1, k1, v1 = qkv1[0], qkv1[1], qkv1[2]
            q2, k2, v2 = qkv2[0], qkv2[1], qkv2[2]
            
            x = F.scaled_dot_product_attention(q1, k2, v2, scale=scale_x)  # B N h d
            x = rearrange(x, "B h N d -> B N (h d)").contiguous()
            x = self.proj_x(x)
            c = F.scaled_dot_product_attention(q2, k1, v1, scale=scale_c)  # B N h d
            c = rearrange(c, "B h M d -> B M (h d)").contiguous()
            c = self.proj_c(c)
        return x, c
    
class MixAttention_v2(nn.Module):
    
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
        self.scale = scale or dim**(-0.5)

        self.use_flash_attn = has_flash_attn
        self.use_xformers = has_xformers and (dim // num_heads) % 32 == 0

        self.qv1 = nn.Linear(dim, 2 * dim)
        self.kv2 = nn.Linear(dim, 2 * dim)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj_x = nn.Linear(dim, dim)
        self.proj_c = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attn_viz = nn.Identity() 

    def forward(self, x, c):
        B, N, C = x.shape        
        B, M, _ = c.shape 
        scale_x = math.log(M, N) * self.scale
        scale_c = math.log(N, N) * self.scale
        
        if self.use_flash_attn:
            qv1 = self.qv1(x)
            qv1 = rearrange(qv1, "B N (x h d) -> B N x h d", x=2, h=self.num_heads).contiguous()
            kv2 = self.kv2(c)
            kv2 = rearrange(kv2, "B M (x h d) -> B M x h d", x=2, h=self.num_heads).contiguous()
            
            q, v1 = qv1[:,:,0], qv1[:,:,1]
            k, v2 = kv2[:,:,0], kv2[:,:,1]
            
            x = flash_attn_func(q, k, v2, softmax_scale=scale_x)
            x = rearrange(x, "B N h d -> B N (h d)").contiguous()
            x = self.proj_x(x)
            c = flash_attn_func(k, q, v1, softmax_scale=scale_c)
            c = rearrange(c, "B M h d -> B M (h d)").contiguous()
            c = self.proj_c(c)
        elif self.use_xformers:
            qv1 = self.qv1(x)
            qv1 = rearrange(qv1, "B N (x h d) -> x B h N d", x=2, h=self.num_heads).contiguous()
            kv2 = self.kv2(c)
            kv2 = rearrange(kv2, "B M (x h d) -> x B h M d", x=2, h=self.num_heads).contiguous()
            
            q, v1 = qv1[0], qv1[1]
            k, v2 = kv2[0], kv2[1]
            
            x = xops.memory_efficient_attention(q, k, v2, scale=scale_x)
            x = rearrange(x, "B h N d -> B N (h d)").contiguous()
            x = self.proj_x(x)
            c = xops.memory_efficient_attention(k, q, v1, scale=scale_c)
            c = rearrange(c, "B h M d -> B M (h d)").contiguous()
            c = self.proj_c(c)
        else:
            qv1 = self.qv1(x)
            qv1 = rearrange(qv1, "B N (x h d) -> x B h N d", x=2, h=self.num_heads).contiguous()
            kv2 = self.kv2(c)
            kv2 = rearrange(kv2, "B M (x h d) -> x B h M d", x=2, h=self.num_heads).contiguous()
            
            q, v1 = qv1[0], qv1[1]
            k, v2 = kv2[0], kv2[1]

            x = F.scaled_dot_product_attention(q, k, v2, scale=scale_x)
            x = rearrange(x, "B h N d -> B N (h d)").contiguous()
            x = self.proj_x(x)
            c = F.scaled_dot_product_attention(k, q, v1, scale=scale_c)
            c = rearrange(c, "B h M d -> B M (h d)").contiguous()
            c = self.proj_c(c)
        return x, c

class StemAttention(nn.Module):
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

        self.use_flash_attn = has_flash_attn
        self.use_xformers = has_xformers and (dim // num_heads) % 32 == 0

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.attn_drop = attn_drop

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attn_viz = nn.Identity() 


    def forward(self, x, c):
        B, N, C = x.shape        
        B, M, _ = c.shape 
        
        if self.use_flash_attn:
            q = self.q(c)
            kv = self.kv(x)
            q = rearrange(q, "B M (h d) -> B M h d", h=self.num_heads).contiguous()
            kv = rearrange(kv, "B N (x h d) -> B N x h d", x=2, h=self.num_heads).contiguous()
            
            c = flash_attn_kvpacked_func(q, kv)
            c = rearrange(c, "B M h d -> B M (h d)").contiguous()
            c = self.proj(c)
        elif self.use_xformers:
            q = self.q(c)
            kv = self.kv(x)
            q = rearrange(q, "B M (h d) -> B M h d", h=self.num_heads).contiguous()
            kv = rearrange(kv, "B N (x h d) -> x B N h d", x=2, h=self.num_heads).contiguous()
            k, v = kv[0], kv[1]
            
            c = xops.memory_efficient_attention(q, k, v)
            c = rearrange(c, "B M h d -> B M (h d)").contiguous()
            c = self.proj(c)
        else:
            q = self.q(c)
            kv = self.kv(x)
            q = rearrange(q, "B M (h d) -> B h M d", h=self.num_heads).contiguous()
            kv = rearrange(kv, "B N (x h d) -> x B h N d", x=2, h=self.num_heads).contiguous()
            k, v = kv[0], kv[1]
            
            c = F.scaled_dot_product_attention(q, k, v)
            c = rearrange(c, "B h M d -> B M (h d)").contiguous()
            c = self.proj(c)
        return c
    

class MixBlock(nn.Module):
    def __init__(self, dim, 
                 attn_drop, proj_drop, drop_path=0., attn_type=None,
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
        
        self.attn_type = attn_type
        if attn_type == "M":
            self.attn = MixAttention(dim=dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=proj_drop)
        elif attn_type == "M2":
            self.attn = MixAttention_v2(dim=dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=proj_drop)
        elif attn_type == "S" or attn_type == None:
            self.attn = StandardAttention(dim=dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=proj_drop)
        elif attn_type == "STEM":
            self.attn = StemAttention(dim=dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=proj_drop)
            
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
            self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones((1,1,dim)), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((1,1,dim)), requires_grad=True)
        else:
            self.use_layer_scale = False
        self.pre_norm = pre_norm
            
    def forward_with_xc(self, x, c):

        _, C, H, W = x.shape
        # conv pos embedding
        x = x + self.pos_embed(x)
        # permute to NHWC tensor for attention & mlp
        x = rearrange(x, "N C H W -> N (H W) C")
        # x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)

        # attention & mlp
        if self.pre_norm:
            if self.use_layer_scale:
                _x, _c = self.attn(self.norm1(x), self.norm1(c))
                x = x + self.drop_path(self.gamma1 * _x) # (N, H, W, C)
                x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x))) # (N, H, W, C)
                c = c + self.drop_path(self.gamma1 * _c) # (N, H, W, C)
                c = c + self.drop_path(self.gamma2 * self.mlp(self.norm2(c))) # (N, H, W, C)
            else:
                _x, _c = self.attn(self.norm1(x), self.norm1(c))
                x = x + self.drop_path(_x) # (N, H, W, C)
                x = x + self.drop_path(self.mlp(self.norm2(x))) # (N, H, W, C)
                c = c + self.drop_path(_c) # (N, H, W, C)
                c = c + self.drop_path(self.mlp(self.norm2(c))) # (N, H, W, C)
        else: # https://kexue.fm/archives/9009
            if self.use_layer_scale:
                _x, _c = self.attn(x,c)
                x = self.norm1(x + self.drop_path(self.gamma1 * _x)) # (N, H, W, C)
                x = self.norm2(x + self.drop_path(self.gamma2 * self.mlp(x))) # (N, H, W, C)
                c = self.norm1(c + self.drop_path(self.gamma1 * _c)) # (N, H, W, C)
                c = self.norm2(c + self.drop_path(self.gamma2 * self.mlp(c))) # (N, H, W, C)
            else:
                _x, _c = self.attn(x,c)
                x = self.norm1(x + self.drop_path(_x)) # (N, H, W, C)
                x = self.norm2(x + self.drop_path(self.mlp(x))) # (N, H, W, C)
                c = self.norm1(c + self.drop_path(_c)) # (N, H, W, C)
                c = self.norm2(c + self.drop_path(self.mlp(c))) # (N, H, W, C)
                
        x = rearrange(x, "N (H W) C -> N C H W",H=H,W=W)
        # permute back
        # x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        return x, c

    def forward_with_c(self, x, c):
        
        _, C, H, W = x.shape
        _x = x
        # conv pos embedding
        x = x + self.pos_embed(x)
        # permute to NHWC tensor for attention & mlp
        x = rearrange(x, "N C H W -> N (H W) C")
        # x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)

        # attention & mlp
        if self.pre_norm:
            if self.use_layer_scale:
                c = c + self.drop_path(self.gamma1 * self.attn(self.norm1(x), self.norm1(c))) # (N, H, W, C)
                c = c + self.drop_path(self.gamma2 * self.mlp(self.norm2(c))) # (N, H, W, C)
            else:
                c = c + self.drop_path(self.attn(self.norm1(x),self.norm1(c))) # (N, H, W, C)
                c = c + self.drop_path(self.mlp(self.norm2(c))) # (N, H, W, C)
        else: # https://kexue.fm/archives/9009
            if self.use_layer_scale:
                c = self.norm1(c + self.drop_path(self.gamma1 * self.attn(x,c))) # (N, H, W, C)
                c = self.norm2(c + self.drop_path(self.gamma2 * self.mlp(c))) # (N, H, W, C)
            else:
                c = self.norm1(c + self.drop_path(self.attn(x,c))) # (N, H, W, C)
                c = self.norm2(c + self.drop_path(self.mlp(c))) # (N, H, W, C)

        x = _x
        # permute back
        # x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        return x, c

    def forward_with_x(self, x, c):
        
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
        return x, c
    
    def forward(self, x, c):
        if self.attn_type == "M" or self.attn_type == "M2":
            return self.forward_with_xc(x,c)
        elif self.attn_type == "S":
            return self.forward_with_x(x,c)
        elif self.attn_type == "STEM":
            return self.forward_with_c(x,c)

@BACKBONES.register_module()
class MixFormer(BaseModule):
    def __init__(self, 
                 depth=[2, 3, 4, 8, 3], 
                 in_chans=3, 
                 num_classes=1000, 
                 embed_dim=[64, 64, 128, 320, 512], 
                 head_dim=64, 
                 mlp_ratios=[4, 4, 4, 4, 4], 
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop=0., 
                 drop_path_rate=0.,
                 # <<<------
                 attn_type=["STEM","M","M","S","S"],
                 queries_len=128,
                 qk_dims=None,
                 cpe_ks=3,
                 pre_norm=True,
                 mlp_dwconv=False,
                 representation_size=None,
                 layer_scale_init_value=-1,
                 use_checkpoint_stages=[],
                 pretrained=None,
                 init_cfg=None,
                 # ------>>>
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        qk_dims = qk_dims or embed_dim
        
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            init_cfg = init_cfg
        else:
            raise TypeError('pretrained must be a str or None')
        
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

        for i in range(4):
            if attn_type[i] == "STEM":
                downsample_layer = nn.Identity()
            else:
                downsample_layer = nn.Sequential(
                    nn.Conv2d(embed_dim[i], embed_dim[i+1], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                    nn.BatchNorm2d(embed_dim[i+1])
                )
            if use_checkpoint_stages:
                downsample_layer = checkpoint_wrapper(downsample_layer)
            self.downsample_layers.append(downsample_layer)
        ##########################################################################

        #TODO: maybe remove last LN
        self.queries_len = queries_len
        self.prototypes = nn.Parameter(torch.randn(self.queries_len ,embed_dim[0]), requires_grad=True) 
        
        self.prototype_downsample = nn.ModuleList()
        prototype_downsample = nn.Sequential(
            nn.Linear(embed_dim[0], embed_dim[0] * 4),
            nn.LayerNorm(embed_dim[0] * 4),
            nn.GELU(),
            nn.Linear(embed_dim[0] * 4, embed_dim[0]),
            nn.LayerNorm(embed_dim[0])
        )
        self.prototype_downsample.append(prototype_downsample)
        for i in range(4):
            prototype_downsample = nn.Sequential(
                nn.Linear(embed_dim[i], embed_dim[i] * 4),
                nn.LayerNorm(embed_dim[i] * 4),
                nn.GELU(),
                nn.Linear(embed_dim[i] * 4, embed_dim[i+1]),
                nn.LayerNorm(embed_dim[i+1])
            )
            self.prototype_downsample.append(prototype_downsample)

        # self.prototype_stem = nn.ModuleList()
        # for i in range(4):
        #     prototype_stem = StemAttention(embed_dim[0], num_heads=2, bias=True)
        #     self.prototype_stem.append(prototype_stem)
        
        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        nheads= [dim // head_dim for dim in qk_dims]
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))] 
        cur = 0
        for i in range(5):
            stage = nn.ModuleList(
                [MixBlock(dim=embed_dim[i], 
                           attn_drop=attn_drop, proj_drop=drop_rate,
                           drop_path=dp_rates[cur + j],
                           attn_type=attn_type[i],
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
        self.norm_c = nn.LayerNorm(embed_dim[-1])
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
        # self.head = nn.Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()
        # self.apply(self._init_weights)

    def forward_features(self, x, c):
        outs = []
        for i in range(5): 
            x = self.downsample_layers[i](x)
            c = self.prototype_downsample[i](c)
            for j, block in enumerate(self.stages[i]):
                x, c = block(x, c)
            if i > 0:
                out = x + c.transpose(-2,-1).contiguous().mean(-1,keepdim=True).unsqueeze(-1)
                outs.append(out)
        # x = self.norm(x)
        # x = self.pre_logits(x)
        
        # c = self.norm_c(c)
        # c = self.pre_logits(c)

        # x = x.flatten(2).mean(-1)
        # c = c.transpose(-2,-1).contiguous().mean(-1)
        # x = x + c

        return outs

    def forward(self, x):
        B, _, H, W = x.shape 
        c = self.prototypes.repeat(B,1,1)
        x = self.forward_features(x, c)
        # x = self.head(x)
        return x

    def init_weights(self):
        logger = get_root_logger()
        if self.init_cfg is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            ckpt = _load_checkpoint(
                self.init_cfg['checkpoint'], logger=logger, map_location='cpu')
                
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'state_dict_ema' in ckpt:
                _state_dict = ckpt['state_dict_ema']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = OrderedDict()
            for k, v in _state_dict.items():
                if k.startswith('backbone.'):
                    state_dict[k[9:]] = v
                else:
                    state_dict[k] = v

            # strip prefix of state_dict
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}


            self.load_state_dict(state_dict, False)
        
    def train(self, mode=True):
        freeze_bn = False
        super(MixFormer, self).train(mode)
        if freeze_bn:
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                if isinstance(m, nn.LayerNorm):
                    m.eval()


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
        depth=[2, 2, 2, 4, 2],
        embed_dim=[96, 96, 192, 320, 384], 
        head_dim=32,
        mlp_ratios=[4, 4, 4, 4, 4],
        attn_type=["STEM","M","M","S","S"],
        queries_len=16,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.,
        qk_dims=None,
        cpe_ks=3,
        pre_norm=True,
        mlp_dwconv=False,
        representation_size=None,
        layer_scale_init_value=-1,
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
def mixformer_tiny_v2(pretrained=False, pretrained_cfg=None,
                  pretrained_cfg_overlay=None, **kwargs):
    model = MixFormer(
        depth=[2, 2, 2, 4, 2],
        embed_dim=[96, 96, 192, 320, 384], 
        head_dim=32,
        mlp_ratios=[4, 4, 4, 4, 4],
        attn_type=["STEM","M2","M2","S","S"],
        queries_len=16,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.,
        qk_dims=None,
        cpe_ks=3,
        pre_norm=True,
        mlp_dwconv=False,
        representation_size=None,
        layer_scale_init_value=-1,
        use_checkpoint_stages=[],
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def mixformer_small(pretrained=False, pretrained_cfg=None,
                  pretrained_cfg_overlay=None, **kwargs):
    model = MixFormer(
        depth=[4, 4, 4, 16, 4],
        embed_dim=[64, 64, 128, 256, 512], 
        head_dim=32,
        mlp_ratios=[3, 3, 3, 3, 3],
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

