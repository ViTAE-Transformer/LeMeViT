import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
from timm.models.registry import register_model
import math
import itertools
from einops import rearrange
from typing import Tuple
import matplotlib.pyplot as plt
import copy

# ******************************************************************************************************basic operations
class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = torch.nn.BatchNorm2d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, out_dim, activation):
        super().__init__()

        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim = out_dim
        self.act = activation()
        self.conv1 = Conv2d_BN(dim, out_dim, 1, 1, 0)
        self.conv2 = Conv2d_BN(out_dim, out_dim, 3, 2, 1, groups=out_dim)
        self.conv3 = Conv2d_BN(out_dim, out_dim, 1, 1, 0)

    def forward(self, x):
        in_ndim = x.ndim
        if in_ndim == 3:
            H, W = self.input_resolution
            B = len(x)
            # (B, C, H, W)
            x = x.view(B, H, W, -1).permute(0, 3, 1, 2)

        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)

        if in_ndim == 3:
            x = x.flatten(2).transpose(1, 2)
        return x


class LayerNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.norm(x)

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, in_chans, embed_dim, resolution, activation):
        super().__init__()
        img_size: Tuple[int, int] = to_2tuple(resolution)
        self.patches_resolution = (img_size[0] // 4, img_size[1] // 4)
        self.num_patches = self.patches_resolution[0] * \
            self.patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        n = embed_dim
        self.seq = nn.Sequential(
            Conv2d_BN(in_chans, n // 2, 3, 2, 1),
            activation(),
            Conv2d_BN(n // 2, n, 3, 2, 1),
        )

    def forward(self, x):
        return self.seq(x)


#*******************************************************************************************************different blocks
# convnext block
class CxBlock(nn.Module):

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

# fca block
class FcaBlock(nn.Module):
    def __init__(self, dim, input_resolution, inner_depth, cab_max_d, s_up, num_heads, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., dp_rate=0.,
                 layer_scale_init_value = 1e-6, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        # paramters****************************************************************************************************
        self.dim = dim
        self.input_resolution = input_resolution
        H, W = input_resolution
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.inner_depth = inner_depth
        self.s_up = s_up

        self.gamma_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.gamma_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

        # skip tokens out
        if self.s_up:
            self.dw_conv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, stride=4, groups=dim, bias=False)
        s_token_num = math.ceil(input_resolution[0] / 4) * math.ceil(input_resolution[1] / 4)

        # depth bias
        if self.inner_depth>0:
            self.s_token_bias = nn.Parameter(torch.zeros(1, num_heads, 1, min(self.inner_depth, cab_max_d)*s_token_num))
            self.s_norm = norm_layer(dim)
            self.s_scale_factors = nn.Parameter(torch.ones(1, min(self.inner_depth,  cab_max_d)*s_token_num, 1))

        # token mixer / self attention*********************************************************************************
        self.norm1 = norm_layer(dim)
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        points = list(itertools.product(
            range(input_resolution[0]), range(input_resolution[1])))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N),
                             persistent=False)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)
        self.softmax = nn.Softmax(dim=-1)

        # local conv
        self.local_conv = Conv2d_BN(
            dim, dim, ks=3, stride=1, pad=1, groups=dim)

        # channel mixer / ffn*******************************************************************************************
        self.drop_path = DropPath(dp_rate) if dp_rate > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, s=None):
        # token mixer
        B, N, C = x.shape
        res_x = x

        if self.inner_depth > 0:
            s = s * self.s_scale_factors
            _, N_S, _ = s.shape
            x, s = self.norm1(x), self.s_norm(s)
            qkv_x = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            qkv_s = self.qkv(s).reshape(B, N_S, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q_x, k_x, v_x = qkv_x[0], qkv_x[1], qkv_x[2]
            k_s, v_s = qkv_s[1], qkv_s[2]

            q_x = q_x * self.scale
            attn_x = (q_x @ k_x.transpose(-2, -1))
            attn_s = (q_x @ k_s.transpose(-2, -1))
            attn_x = attn_x + self.attention_biases[:, self.attention_bias_idxs]
            attn_s = attn_s + self.s_token_bias
            attn = torch.cat((attn_x, attn_s), dim=-1)
            v = torch.cat((v_x, v_s), dim=-2)

            attn = self.softmax(attn)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            x = res_x + self.drop_path(self.gamma_1 * x)
        else:
            skip_x = x
            x = self.norm1(x)
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))

            attn = attn + self.attention_biases[:, self.attention_bias_idxs]

            attn = self.softmax(attn)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            x = skip_x + self.drop_path(self.gamma_1 * x)

        # local conv and skip token generate
        H, W = self.input_resolution
        x = x.transpose(1,2).reshape(B, C, H, W)
        if self.s_up:
            s_update = self.dw_conv(x)
            s_update = s_update.view(B, C, -1).transpose(1,2)
        x = self.local_conv(x)
        x = x.view(B, C, N).transpose(1,2)

        # channel mixer
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))

        if self.s_up:
            return x, s_update
        else:
            return x, []

#*******************************************************************************************************different layers
class CxLayer(nn.Module):
    def __init__(self, dim_in, dim_out, input_resolution, depth, dp_rates, layer_scale_init_value=1e-6, downsample=None):
        super().__init__()
        self.dim_in = dim_in
        self.input_resolutuion = input_resolution
        self.depth=depth
        self.downsample=downsample

        # blocks
        self.blocks = nn.Sequential(*[
            CxBlock(dim=dim_in, drop_path=dp_rates[j],layer_scale_init_value=layer_scale_init_value)
            for j in range(depth)
        ])

        # downsample
        if self.downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim_in, out_dim=dim_out, activation=nn.GELU)

    def forward(self, x):
        x = self.blocks(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x


class FcaLayer(nn.Module):
    def __init__(self, dim_in, dim_out, input_resolution, depth, num_heads,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 dp_rates=[], layer_scale_init_value = 1e-6,
                 norm_layer=nn.LayerNorm, downsample=None, cab_max_d=12):

        super().__init__()
        self.dim = dim_in
        self.input_resolution = input_resolution
        self.depth = depth
        self.downsample = downsample
        self.cab_max_d = cab_max_d

        self.blocks = nn.ModuleList([
            FcaBlock(dim=dim_in, input_resolution=input_resolution,
                     inner_depth=i,
                     cab_max_d=cab_max_d,
                     s_up=i < depth - 1,
                     num_heads=num_heads,
                     mlp_ratio=mlp_ratio,
                     qkv_bias=qkv_bias, qk_scale=qk_scale,
                     drop=drop, attn_drop=attn_drop,
                     dp_rate=dp_rates[i],
                     layer_scale_init_value=layer_scale_init_value,
                     norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if self.downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim_in, out_dim=dim_out, activation=nn.GELU)

    def forward(self, x):
        s_list = []
        for i, blk in enumerate(self.blocks):
            if i==0:
                x, s = blk(x)
                s_list.append(s)
            else:
                x, s = blk(x, torch.cat(s_list[-self.cab_max_d:], dim=1))
                s_list.append(s)

        if self.downsample is not None:
            x = self.downsample(x)
        return x

#***************************************************************************************************models
# FcaFormer
class FcaFormer(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=1000,
                 embed_dims=[48, 96, 192, 384], depths=[3, 3, 9, 2],
                 num_heads=[0, 0, 12, 24],
                 mlp_ratio=4.,
                 layer_scale_init_value=1e-6,
                 drop_path_rate=0.1,
                 head_init_scale=1.0,
                 cab_max_d = 12
                 ):
        super().__init__()

        self.num_classes = num_classes
        self.depths = depths
        self.num_layers = len(depths)
        self.mlp_ratio = mlp_ratio

        activation = nn.GELU
        self.patch_embed = PatchEmbed(in_chans=in_chans,
                                      embed_dim=embed_dims[0],
                                      resolution=img_size,
                                      activation=activation)

        patches_resolution = self.patch_embed.patches_resolution

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        layer_dpr = []
        cur_depth=0
        for depth in depths:
            layer_dpr.append(dpr[cur_depth:cur_depth+depth])
            cur_depth+=depth

        # two shallow layers:
        self.layer_0 = CxLayer(dim_in=embed_dims[0], dim_out=embed_dims[1],
                               input_resolution=(patches_resolution[0] // (2 ** 0), patches_resolution[1] // (2 ** 0)),
                               depth=depths[0],
                               dp_rates=layer_dpr[0],
                               layer_scale_init_value=1e-6,
                               downsample=PatchMerging,
                               )
        self.layer_1 = CxLayer(dim_in=embed_dims[1], dim_out=embed_dims[2],
                               input_resolution=(patches_resolution[0] // (2 ** 1), patches_resolution[1] // (2 ** 1)),
                               depth=depths[1],
                               dp_rates=layer_dpr[1],
                               layer_scale_init_value=1e-6,
                               downsample=PatchMerging,
                               )
        # two deep layers:
        self.layer_2 = FcaLayer(dim_in=embed_dims[2], dim_out=embed_dims[3],
                                input_resolution=(patches_resolution[0] // (2 ** 2), patches_resolution[1] // (2 ** 2)),
                                depth=depths[2],
                                num_heads=num_heads[2],
                                dp_rates=layer_dpr[2],
                                layer_scale_init_value=1e-6,
                                downsample=PatchMerging,
                                cab_max_d=cab_max_d,
                                )
        self.layer_3 = FcaLayer(dim_in=embed_dims[3], dim_out=embed_dims[3],
                                input_resolution=(patches_resolution[0] // (2 ** 3), patches_resolution[1] // (2 ** 3)),
                                depth=depths[3],
                                num_heads=num_heads[3],
                                dp_rates=layer_dpr[3],
                                layer_scale_init_value=1e-6,
                                downsample=None,
                                cab_max_d=cab_max_d
                                )

        # Classifier head
        self.norm_head = nn.LayerNorm(embed_dims[-1])
        self.head = nn.Linear(
            embed_dims[-1], num_classes) if num_classes > 0 else torch.nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.patch_embed(x) # x: (B, C, 56, 56)
        # shallow layers
        x = self.layer_0(x)
        x = self.layer_1(x)
        x = rearrange(x, 'b c h w -> b (h w) c') # x: (B, N, C)
        # deep layers
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = x.mean(1)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(self.norm_head(x))
        return x

    def get_param_num(self):
        total_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_num

# 6.2M params 1.37 GFlops 80.3 top1 acc
@register_model
def fcaformer_l1(pretrained=False, in_22k=False, **kwargs):
    model = FcaFormer(embed_dims=[64, 128, 192, 320], depths=[2, 2, 6, 2], num_heads=[0, 0, 6, 10], **kwargs)
    return model

# 16.3M params 3.6 GFlops 83.1 top1 acc
@register_model
def fcaformer_l2(pretrained=False, in_22k=False, **kwargs):
    model = FcaFormer(embed_dims=[96, 192, 320, 480], depths=[2, 2, 7, 2], num_heads=[0, 0, 10, 15], **kwargs)
    return model

# 27.9M params 6.7 GFlops 84.2 top1 acc
@register_model
def fcaformer_l3(pretrained=False, in_22k=False, **kwargs):
    model = FcaFormer(embed_dims=[96, 192, 320, 512], depths=[3, 6, 12, 3], num_heads=[0, 0, 10, 16], **kwargs)
    return model

# 65.6M params 14.5 GFlops 84.9 top1 acc
@register_model
def fcaformer_l4(pretrained=False, in_22k=False, **kwargs):
    model = FcaFormer(embed_dims=[128, 256, 512, 768], depths=[3, 6, 12, 3], num_heads=[0, 0, 16, 24], **kwargs)
    return model




