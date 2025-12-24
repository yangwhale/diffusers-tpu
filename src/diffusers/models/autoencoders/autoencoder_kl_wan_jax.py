# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
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

"""
Pure JAX implementation of Wan VAE (AutoencoderKLWan).

=============================================================================
FLAX NNX → PURE JAX 改造指南
=============================================================================

本文件从 autoencoder_kl_wan_flax.py 改造而来，主要改动点用 # PURE_JAX: 标记。

核心改动原则：
1. nnx.Module 类 → 纯函数 + Params 字典
2. self.xxx 属性 → params['xxx'] 参数传递
3. nnx.Conv → jax.lax.conv_general_dilated
4. nnx.Param → 普通 jnp.ndarray 在 params 字典中
5. self._feat_map 可变状态 → cache 显式传递和返回
6. nnx.jit → jax.jit（无需 pytree=False hack）

改造后的优势：
- 无 nnx.split/merge 开销
- 无 pytree=False 导致的 tracing 限制
- 纯函数式，更适合 JAX 的编译优化
- 更透明的参数管理

=============================================================================
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
# PURE_JAX: 移除 from flax import nnx
from jax import lax  # PURE_JAX: 用于底层卷积

from jax.sharding import PartitionSpec as P

# PURE_JAX: 类型别名，替代 nnx.Module
Params = Dict[str, Any]
Cache = Optional[jnp.ndarray]
CacheList = List[Cache]

# FLAX: Flax equivalent of interop.torch_view(jax.lax.with_sharding_constraint)
def mark_sharding(inputs, spec):
    try:
        return jax.lax.with_sharding_constraint(inputs, spec)
    except (ValueError, Exception):
        return inputs

CACHE_T = 2


# =============================================================================
# PURE_JAX: 卷积基础函数 (替代 nnx.Conv)
# =============================================================================

def conv3d(params: Params, x: jnp.ndarray, strides=(1, 1, 1), padding="VALID") -> jnp.ndarray:
    """
    PURE_JAX: 替代 nnx.Conv 的 3D 卷积
    
    params: {'kernel': (T, H, W, In, Out), 'bias': (Out,)}
    x: (B, T, H, W, C) - NTHWC 格式
    """
    out = lax.conv_general_dilated(
        lhs=x,
        rhs=params['kernel'],
        window_strides=strides,
        padding=padding,
        dimension_numbers=('NTHWC', 'THWIO', 'NTHWC'),
    )
    if 'bias' in params and params['bias'] is not None:
        out = out + params['bias']
    return out


def conv2d(params: Params, x: jnp.ndarray, strides=(1, 1), padding="VALID") -> jnp.ndarray:
    """
    PURE_JAX: 替代 nnx.Conv 的 2D 卷积
    
    params: {'kernel': (H, W, In, Out), 'bias': (Out,)}
    x: (B, H, W, C) - NHWC 格式
    """
    out = lax.conv_general_dilated(
        lhs=x,
        rhs=params['kernel'],
        window_strides=strides,
        padding=padding,
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    if 'bias' in params and params['bias'] is not None:
        out = out + params['bias']
    return out


# =============================================================================
# AvgDown3D: nnx.Module → 纯函数
# =============================================================================
# PURE_JAX: 原 class AvgDown3D(nnx.Module) 改为纯函数
# PURE_JAX: __init__ 中的 self.xxx = xxx 不再需要，参数直接传入函数

def avg_down_3d(
    x: jnp.ndarray,
    in_channels: int,
    out_channels: int,
    factor_t: int,
    factor_s: int = 1,
) -> jnp.ndarray:
    """
    PURE_JAX: 原 AvgDown3D.__call__ 方法
    无需 params，因为此函数无可学习参数
    """
    # FLAX: x is (B, T, H, W, C) in NTHWC format, not (B, C, T, H, W)
    B, T, H, W, C = x.shape
    factor = factor_t * factor_s * factor_s
    group_size = in_channels * factor // out_channels

    pad_t = (factor_t - T % factor_t) % factor_t
    if pad_t > 0:
        # FLAX: jnp.pad instead of F.pad, different axis order
        pad_width = [(0, 0), (pad_t, 0), (0, 0), (0, 0), (0, 0)]
        x = jnp.pad(x, pad_width, mode='constant', constant_values=0)
        T = x.shape[1]

    # FLAX: reshape for NTHWC format
    x = x.reshape(B, T//factor_t, factor_t, H//factor_s, factor_s, W//factor_s, factor_s, C)
    x = x.transpose(0, 1, 3, 5, 7, 2, 4, 6)
    x = x.reshape(B, T//factor_t, H//factor_s, W//factor_s, C * factor)
    x = x.reshape(B, T//factor_t, H//factor_s, W//factor_s, out_channels, group_size)
    x = jnp.mean(x, axis=-1)
    return x


# =============================================================================
# DupUp3D: nnx.Module → 纯函数
# =============================================================================

def dup_up_3d(
    x: jnp.ndarray,
    in_channels: int,
    out_channels: int,
    factor_t: int,
    factor_s: int = 1,
    first_chunk: bool = False,
) -> jnp.ndarray:
    """PURE_JAX: 原 DupUp3D.__call__ 方法"""
    # FLAX: x is (B, T, H, W, C)
    B, T, H, W, C = x.shape
    factor = factor_t * factor_s * factor_s
    repeats = out_channels * factor // in_channels

    # FLAX: jnp.repeat instead of repeat_interleave
    x = jnp.repeat(x[:, :, :, :, :, None], repeats, axis=5).reshape(B, T, H, W, C * repeats)
    x = x.reshape(B, T, H, W, out_channels, factor_t, factor_s, factor_s)
    x = x.transpose(0, 1, 5, 2, 6, 3, 7, 4)
    x = x.reshape(B, T * factor_t, H * factor_s, W * factor_s, out_channels)

    if first_chunk:
        x = x[:, factor_t - 1:, :, :, :]
    return x


# =============================================================================
# WanCausalConv3d: nnx.Module → 纯函数
# =============================================================================
# PURE_JAX: 原 class WanCausalConv3d(nnx.Module)
# PURE_JAX: __init__ 中 self.conv = nnx.Conv(...) → params['conv'] 传入
# PURE_JAX: __call__ → wan_causal_conv3d 函数

def get_causal_padding(kernel_size, stride, padding):
    """PURE_JAX: 计算因果卷积的 padding 配置"""
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,) * 3
    if isinstance(stride, int):
        stride = (stride,) * 3
    if isinstance(padding, int):
        padding = (padding,) * 3
    # (W_left, W_right, H_left, H_right, T_left, T_right)
    return (padding[2], padding[2], padding[1], padding[1], 2 * padding[0], 0)


def wan_causal_conv3d(
    params: Params,
    x: jnp.ndarray,
    cache_x: Cache = None,
    padding_config: Tuple[int, ...] = None,
) -> jnp.ndarray:
    """
    PURE_JAX: 原 WanCausalConv3d.__call__ 方法
    
    params: {'kernel': ..., 'bias': ...}  # PURE_JAX: 原 self.conv 的参数
    padding_config: 从 get_causal_padding 获取
    """
    # FLAX: x is (B, T, H, W, C)
    padding = list(padding_config)
    if cache_x is not None and padding[4] > 0:
        x = jnp.concatenate([cache_x, x], axis=1)
        padding[4] -= cache_x.shape[1]

    # FLAX: jnp.pad for NTHWC format
    pad_width = [
        (0, 0),  # Batch
        (padding[4], padding[5]),  # Time
        (padding[2], padding[3]),  # Height
        (padding[0], padding[1]),  # Width
        (0, 0),  # Channels
    ]
    x = jnp.pad(x, pad_width, mode='constant', constant_values=0)

    # Sharding along width (matches TorchAx mark_sharding on height in NCTHW)
    success = False
    try:
        x = mark_sharding(x, P(None, None, None, ("dp", "tp"), None))
        success = True
    except ValueError:
        pass
    if not success:
        try:
            x = mark_sharding(x, P(None, None, None, ("tp",), None))
            success = True
        except ValueError:
            pass
    if not success:
        try:
            x = mark_sharding(x, P(None, None, None, ("dp",), None))
        except ValueError:
            pass

    # PURE_JAX: self.conv(x) → conv3d(params, x)
    return conv3d(params, x, strides=(1, 1, 1), padding="VALID")


# =============================================================================
# WanRMS_norm: nnx.Module → 纯函数
# =============================================================================
# PURE_JAX: 原 self.gamma = nnx.Param(...) → params['gamma']

def rms_norm(params: Params, x: jnp.ndarray) -> jnp.ndarray:
    """
    PURE_JAX: 原 WanRMS_norm.__call__ 方法
    
    params: {'gamma': (dim,), 'bias': (dim,) 可选}
    """
    # FLAX: F.normalize -> manual L2 normalize along channel dim
    scale = jnp.sqrt(x.shape[-1]).astype(x.dtype)  # PURE_JAX: 动态计算 scale
    norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
    norm = jnp.maximum(norm, 1e-12)
    x_normalized = x / norm
    out = x_normalized * scale * params['gamma']  # PURE_JAX: params['gamma'] 替代 self.gamma.value
    if 'bias' in params and params['bias'] is not None:
        out = out + params['bias']
    return out


# =============================================================================
# WanResample: nnx.Module → 纯函数
# =============================================================================
# PURE_JAX: 最复杂的改造之一
# PURE_JAX: 原 self.resample_conv, self.time_conv → params['resample_conv'], params['time_conv']
# PURE_JAX: 根据 mode 拆分为多个函数

def resample_upsample_3d(
    params: Params,
    x: jnp.ndarray,
    dim: int,
    feat_cache: CacheList = None,
    feat_idx: int = 0,
) -> Tuple[jnp.ndarray, int, CacheList]:
    """PURE_JAX: 原 WanResample.__call__ 的 upsample3d 分支"""
    B, T, H, W, C = x.shape

    if feat_cache is not None:
        idx = feat_idx
        if feat_cache[idx] is None:
            feat_cache[idx] = (None,)
            feat_idx += 1
        else:
            cache_x = x[:, -CACHE_T:, :, :, :]
            if cache_x.shape[1] < 2 and feat_cache[idx] is not None and feat_cache[idx][0] is not None:
                cache_x = jnp.concatenate([feat_cache[idx][0][:, -1:, :, :, :], cache_x], axis=1)
            if cache_x.shape[1] < 2 and feat_cache[idx] is not None and feat_cache[idx][0] is None:
                cache_x = jnp.concatenate([jnp.zeros_like(cache_x), cache_x], axis=1)
            
            # PURE_JAX: self.time_conv(x, ...) → wan_causal_conv3d(params['time_conv'], x, ...)
            time_conv_padding = get_causal_padding((3, 1, 1), 1, (1, 0, 0))
            if feat_cache[idx][0] is None:
                x = wan_causal_conv3d(params['time_conv'], x, None, time_conv_padding)
            else:
                x = wan_causal_conv3d(params['time_conv'], x, feat_cache[idx][0], time_conv_padding)
            feat_cache[idx] = (cache_x,)
            feat_idx += 1

            # FLAX: reshape for NTHWC
            x = x.reshape(B, T, H, W, 2, dim)
            x = x.transpose(0, 1, 4, 2, 3, 5).reshape(B, T * 2, H, W, dim)

    T_curr = x.shape[1]
    C_curr = x.shape[4]
    x = x.reshape(B * T_curr, H, W, C_curr)

    # FLAX: jax.image.resize instead of nn.Upsample
    x = jax.image.resize(x, (B * T_curr, H * 2, W * 2, C_curr), method='nearest')
    
    # PURE_JAX: self.resample_conv(x) → conv2d(params['resample_conv'], x, ...)
    x = conv2d(params['resample_conv'], x, strides=(1, 1), padding=((1, 1), (1, 1)))

    H_new, W_new = x.shape[1], x.shape[2]
    x = x.reshape(B, T_curr, H_new, W_new, x.shape[-1])

    return x, feat_idx, feat_cache


def resample_upsample_2d(params: Params, x: jnp.ndarray) -> jnp.ndarray:
    """PURE_JAX: 原 WanResample.__call__ 的 upsample2d 分支"""
    B, T, H, W, C = x.shape
    x = x.reshape(B * T, H, W, C)
    
    x = jax.image.resize(x, (B * T, H * 2, W * 2, C), method='nearest')
    x = conv2d(params['resample_conv'], x, strides=(1, 1), padding=((1, 1), (1, 1)))
    
    H_new, W_new = x.shape[1], x.shape[2]
    return x.reshape(B, T, H_new, W_new, x.shape[-1])


def resample_downsample_3d(
    params: Params,
    x: jnp.ndarray,
    feat_cache: CacheList = None,
    feat_idx: int = 0,
) -> Tuple[jnp.ndarray, int, CacheList]:
    """PURE_JAX: 原 WanResample.__call__ 的 downsample3d 分支"""
    B, T, H, W, C = x.shape
    x = x.reshape(B * T, H, W, C)
    
    # PURE_JAX: self.resample_conv(x) → conv2d
    x = conv2d(params['resample_conv'], x, strides=(2, 2), padding=((0, 1), (0, 1)))
    
    H_new, W_new = x.shape[1], x.shape[2]
    x = x.reshape(B, T, H_new, W_new, x.shape[-1])
    
    if feat_cache is not None:
        idx = feat_idx
        if feat_cache[idx] is None:
            feat_cache[idx] = x
            feat_idx += 1
        else:
            cache_x = x[:, -1:, :, :, :]
            time_conv_padding = get_causal_padding((3, 1, 1), (2, 1, 1), (0, 0, 0))
            x = wan_causal_conv3d(
                params['time_conv'],
                jnp.concatenate([feat_cache[idx][:, -1:, :, :, :], x], axis=1),
                None,
                time_conv_padding,
            )
            feat_cache[idx] = cache_x
            feat_idx += 1

    return x, feat_idx, feat_cache


def resample_downsample_2d(params: Params, x: jnp.ndarray) -> jnp.ndarray:
    """PURE_JAX: 原 WanResample.__call__ 的 downsample2d 分支"""
    B, T, H, W, C = x.shape
    x = x.reshape(B * T, H, W, C)
    x = conv2d(params['resample_conv'], x, strides=(2, 2), padding=((0, 1), (0, 1)))
    H_new, W_new = x.shape[1], x.shape[2]
    return x.reshape(B, T, H_new, W_new, x.shape[-1])


# =============================================================================
# WanResidualBlock: nnx.Module → 纯函数
# =============================================================================
# PURE_JAX: 原 self.norm1, self.conv1, self.norm2, self.conv2, self.conv_shortcut
# PURE_JAX: → params['norm1'], params['conv1'], params['norm2'], params['conv2'], params['conv_shortcut']

def residual_block(
    params: Params,
    x: jnp.ndarray,
    feat_cache: CacheList = None,
    feat_idx: int = 0,
) -> Tuple[jnp.ndarray, int, CacheList]:
    """PURE_JAX: 原 WanResidualBlock.__call__ 方法"""
    # Apply shortcut connection
    # PURE_JAX: self.conv_shortcut(x) → wan_causal_conv3d(params['conv_shortcut'], x, ...)
    if 'conv_shortcut' in params and params['conv_shortcut'] is not None:
        shortcut_padding = get_causal_padding(1, 1, 0)
        h = wan_causal_conv3d(params['conv_shortcut'], x, None, shortcut_padding)
    else:
        h = x

    # First normalization and activation
    x = rms_norm(params['norm1'], x)  # PURE_JAX: self.norm1(x) → rms_norm(params['norm1'], x)
    x = jax.nn.silu(x)

    conv_padding = get_causal_padding(3, 1, 1)
    if feat_cache is not None:
        idx = feat_idx
        cache_x = x[:, -CACHE_T:, :, :, :]
        if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
            cache_x = jnp.concatenate([feat_cache[idx][:, -1:, :, :, :], cache_x], axis=1)

        # PURE_JAX: self.conv1(x, feat_cache[idx]) → wan_causal_conv3d(params['conv1'], x, feat_cache[idx], ...)
        x = wan_causal_conv3d(params['conv1'], x, feat_cache[idx], conv_padding)
        feat_cache[idx] = cache_x
        feat_idx += 1
    else:
        x = wan_causal_conv3d(params['conv1'], x, None, conv_padding)

    # Second normalization and activation
    x = rms_norm(params['norm2'], x)
    x = jax.nn.silu(x)

    # Dropout (skip in deterministic mode)

    if feat_cache is not None:
        idx = feat_idx
        cache_x = x[:, -CACHE_T:, :, :, :]
        if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
            cache_x = jnp.concatenate([feat_cache[idx][:, -1:, :, :, :], cache_x], axis=1)

        x = wan_causal_conv3d(params['conv2'], x, feat_cache[idx], conv_padding)
        feat_cache[idx] = cache_x
        feat_idx += 1
    else:
        x = wan_causal_conv3d(params['conv2'], x, None, conv_padding)

    # Add residual connection
    return x + h, feat_idx, feat_cache


# =============================================================================
# WanAttentionBlock: nnx.Module → 纯函数
# =============================================================================

def attention_block(params: Params, x: jnp.ndarray) -> jnp.ndarray:
    """PURE_JAX: 原 WanAttentionBlock.__call__ 方法"""
    identity = x
    # FLAX: x is (B, T, H, W, C)
    B, T, H, W, C = x.shape

    x = x.reshape(B * T, H, W, C)
    x = rms_norm(params['norm'], x)  # PURE_JAX: self.norm(x) → rms_norm(params['norm'], x)

    # compute query, key, value
    # PURE_JAX: self.to_qkv(x) → conv2d(params['to_qkv'], x, ...)
    qkv = conv2d(params['to_qkv'], x, strides=(1, 1), padding="SAME")
    qkv = qkv.reshape(B * T, H * W, 3 * C)
    q, k, v = jnp.split(qkv, 3, axis=-1)

    # add head dimension (1 head)
    q = q[:, None, :, :]
    k = k[:, None, :, :]
    v = v[:, None, :, :]

    # apply attention
    # FLAX: manual scaled dot product attention
    scale = 1.0 / jnp.sqrt(jnp.array(C, dtype=x.dtype))
    attn_weights = jnp.matmul(q, k.swapaxes(-1, -2)) * scale
    attn_weights = jax.nn.softmax(attn_weights, axis=-1)
    x = jnp.matmul(attn_weights, v)

    x = x.squeeze(1).reshape(B * T, H, W, C)

    # output projection
    x = conv2d(params['proj'], x, strides=(1, 1), padding="SAME")

    # Reshape back
    x = x.reshape(B, T, H, W, C)

    return x + identity


# =============================================================================
# WanMidBlock: nnx.Module → 纯函数
# =============================================================================
# PURE_JAX: 原 self.resnets = nnx.List([...]) → params['resnets'] = {0: ..., 1: ...}
# PURE_JAX: 注意 nnx.List 变为整数键的字典

def mid_block(
    params: Params,
    x: jnp.ndarray,
    feat_cache: CacheList = None,
    feat_idx: int = 0,
) -> Tuple[jnp.ndarray, int, CacheList]:
    """PURE_JAX: 原 WanMidBlock.__call__ 方法"""
    resnets = params['resnets']
    attentions = params['attentions']
    
    # PURE_JAX: 遍历字典需要按键排序
    resnet_keys = sorted([k for k in resnets.keys() if isinstance(k, int)])
    attn_keys = sorted([k for k in attentions.keys() if isinstance(k, int)])
    
    # First residual block
    x, feat_idx, feat_cache = residual_block(resnets[resnet_keys[0]], x, feat_cache, feat_idx)

    # Process through attention and residual blocks
    # PURE_JAX: 原 for attn, resnet in zip(self.attentions, self.resnets[1:])
    for i, attn_key in enumerate(attn_keys):
        resnet_key = resnet_keys[i + 1]
        if attentions[attn_key] is not None:
            x = attention_block(attentions[attn_key], x)
        x, feat_idx, feat_cache = residual_block(resnets[resnet_key], x, feat_cache, feat_idx)

    return x, feat_idx, feat_cache


# =============================================================================
# WanUpBlock: nnx.Module → 纯函数
# =============================================================================

def up_block(
    params: Params,
    x: jnp.ndarray,
    out_dim: int,
    upsample_mode: Optional[str] = None,
    feat_cache: CacheList = None,
    feat_idx: int = 0,
) -> Tuple[jnp.ndarray, int, CacheList]:
    """PURE_JAX: 原 WanUpBlock.__call__ 方法"""
    resnets = params['resnets']
    resnet_keys = sorted([k for k in resnets.keys() if isinstance(k, int)])
    
    # PURE_JAX: 原 for resnet in self.resnets
    for key in resnet_keys:
        x, feat_idx, feat_cache = residual_block(resnets[key], x, feat_cache, feat_idx)

    # PURE_JAX: 原 if len(self.upsamplers) > 0
    if 'upsamplers' in params and 0 in params['upsamplers']:
        upsampler_params = params['upsamplers'][0]
        if upsample_mode == "upsample3d":
            x, feat_idx, feat_cache = resample_upsample_3d(
                upsampler_params, x, out_dim, feat_cache, feat_idx
            )
        elif upsample_mode == "upsample2d":
            x = resample_upsample_2d(upsampler_params, x)
    
    return x, feat_idx, feat_cache


# =============================================================================
# WanDecoder3d: nnx.Module → 纯函数
# =============================================================================
# PURE_JAX: 这是最关键的改造 - 整个 Decoder 类变成一个函数

def decoder_forward(
    params: Params,
    x: jnp.ndarray,
    config: "AutoencoderKLWanConfig",
    feat_cache: CacheList = None,
    first_chunk: bool = False,
) -> Tuple[jnp.ndarray, int, CacheList]:
    """
    PURE_JAX: 原 WanDecoder3d.__call__ 方法
    
    params: {
        'conv_in': {...},
        'mid_block': {...},
        'up_blocks': {0: {...}, 1: {...}, ...},
        'norm_out': {...},
        'conv_out': {...},
    }
    """
    feat_idx = 0
    conv_padding = get_causal_padding(3, 1, 1)
    
    ## conv_in
    if feat_cache is not None:
        idx = feat_idx
        cache_x = x[:, -CACHE_T:, :, :, :]
        if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
            cache_x = jnp.concatenate([feat_cache[idx][:, -1:, :, :, :], cache_x], axis=1)
        # PURE_JAX: self.conv_in(x, feat_cache[idx]) → wan_causal_conv3d(params['conv_in'], x, ...)
        x = wan_causal_conv3d(params['conv_in'], x, feat_cache[idx], conv_padding)
        feat_cache[idx] = cache_x
        feat_idx += 1
    else:
        x = wan_causal_conv3d(params['conv_in'], x, None, conv_padding)

    ## middle
    x, feat_idx, feat_cache = mid_block(params['mid_block'], x, feat_cache, feat_idx)

    ## upsamples
    # PURE_JAX: 原 for up_block in self.up_blocks
    up_blocks = params['up_blocks']
    up_block_keys = sorted([k for k in up_blocks.keys() if isinstance(k, int)])
    
    # 计算 upsample mode (与 WanDecoder3d.__init__ 逻辑一致)
    dim_mult = list(config.dim_mult)
    temperal_upsample = list(reversed(config.temperal_downsample))
    decoder_base_dim = config.decoder_base_dim or config.base_dim
    dims = [decoder_base_dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
    
    for i, key in enumerate(up_block_keys):
        up_flag = i != len(dim_mult) - 1
        temp_up = temperal_upsample[i] if up_flag else False
        upsample_mode = None
        if up_flag:
            upsample_mode = "upsample3d" if temp_up else "upsample2d"
        
        # 计算 out_dim
        if i > 0 and not config.is_residual:
            in_dim = dims[i] // 2
        else:
            in_dim = dims[i]
        out_dim = dims[i + 1]
        
        x, feat_idx, feat_cache = up_block(
            up_blocks[key], x, out_dim, upsample_mode, feat_cache, feat_idx
        )

    ## head
    x = rms_norm(params['norm_out'], x)
    x = jax.nn.silu(x)
    if feat_cache is not None:
        idx = feat_idx
        cache_x = x[:, -CACHE_T:, :, :, :]
        if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
            cache_x = jnp.concatenate([feat_cache[idx][:, -1:, :, :, :], cache_x], axis=1)
        x = wan_causal_conv3d(params['conv_out'], x, feat_cache[idx], conv_padding)
        feat_cache[idx] = cache_x
        feat_idx += 1
    else:
        x = wan_causal_conv3d(params['conv_out'], x, None, conv_padding)

    # Replicate back to every devices
    x = mark_sharding(x, P())
    return x, feat_idx, feat_cache


# =============================================================================
# patchify / unpatchify: 保持不变（原本就是纯函数）
# =============================================================================

def patchify(x, patch_size):
    if patch_size == 1:
        return x

    if x.ndim != 5:
        raise ValueError(f"Invalid input shape: {x.shape}")
    # FLAX: x shape: [B, T, H, W, C] instead of [B, C, T, H, W]
    B, T, H, W, C = x.shape

    if H % patch_size != 0 or W % patch_size != 0:
        raise ValueError(f"Height ({H}) and width ({W}) must be divisible by patch_size ({patch_size})")

    x = x.reshape(B, T, H // patch_size, patch_size, W // patch_size, patch_size, C)
    x = x.transpose(0, 1, 2, 4, 3, 5, 6)
    x = x.reshape(B, T, H // patch_size, W // patch_size, C * patch_size * patch_size)

    return x


def unpatchify(x, patch_size):
    if patch_size == 1:
        return x

    if x.ndim != 5:
        raise ValueError(f"Invalid input shape: {x.shape}")
    # FLAX: x shape: (B, T, H, W, C_patches)
    B, T, H, W, C_patches = x.shape
    C = C_patches // (patch_size * patch_size)

    x = x.reshape(B, T, H, W, patch_size, patch_size, C)
    x = x.transpose(0, 1, 2, 4, 3, 5, 6)
    x = x.reshape(B, T, H * patch_size, W * patch_size, C)

    return x


# =============================================================================
# AutoencoderKLWanConfig: 保持不变
# =============================================================================

@dataclass
class AutoencoderKLWanConfig:
    """
    Configuration class for AutoencoderKLWan.
    PURE_JAX: 配置类保持不变
    """
    config_name: str = "config.json"

    base_dim: int = 96
    decoder_base_dim: Optional[int] = None
    z_dim: int = 16
    dim_mult: Tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    attn_scales: Tuple[float, ...] = ()
    temperal_downsample: Tuple[bool, ...] = (False, True, True)
    dropout: float = 0.0
    latents_mean: Tuple[float, ...] = (
        -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
        0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
    )
    latents_std: Tuple[float, ...] = (
        2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
        3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
    )
    is_residual: bool = False
    in_channels: int = 3
    out_channels: int = 3
    patch_size: Optional[int] = None
    scale_factor_temporal: Optional[int] = 4
    scale_factor_spatial: Optional[int] = 8

    @classmethod
    def from_dict(cls, config_dict: Dict):
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(cls)}
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}
        # Handle list/tuple conversion
        for key in ["dim_mult", "attn_scales", "temperal_downsample", "latents_mean", "latents_std"]:
            if key in filtered_dict:
                filtered_dict[key] = tuple(filtered_dict[key])
        return cls(**filtered_dict)


# =============================================================================
# AutoencoderKLWan: nnx.Module → 普通类 + 纯函数
# =============================================================================
# PURE_JAX: 原 class AutoencoderKLWan(nnx.Module, pytree=False)
# PURE_JAX: 改为普通类，不继承 nnx.Module，无需 pytree=False hack

class AutoencoderKLWan:
    """
    PURE_JAX: 原 AutoencoderKLWan(nnx.Module, pytree=False) 改为普通类
    
    主要改动：
    1. 不继承 nnx.Module
    2. self.encoder, self.decoder 等 → self.params['encoder'], self.params['decoder']
    3. self._feat_map 可变状态 → 显式传递的 cache
    4. 无需 pytree=False，因为不再是 nnx.Module
    """

    config_class = AutoencoderKLWanConfig

    def __init__(self, params: Params, config: AutoencoderKLWanConfig):
        """
        PURE_JAX: 原 __init__ 创建子模块，现在直接接收 params
        """
        self.params = params
        self.config = config

    def _count_decoder_convs(self) -> int:
        """PURE_JAX: 计算 decoder 中需要 cache 的卷积数量"""
        count = 1  # conv_in
        count += 4  # mid_block: 2 resnets * 2 convs each
        
        dim_mult = list(self.config.dim_mult)
        num_res_blocks = self.config.num_res_blocks
        temperal_upsample = list(reversed(self.config.temperal_downsample))
        
        for i in range(len(dim_mult)):
            count += (num_res_blocks + 1) * 2  # resnets
            up_flag = i != len(dim_mult) - 1
            if up_flag and temperal_upsample[i]:
                count += 1  # time_conv in upsample3d
        
        count += 1  # conv_out
        return count

    def _decode(self, z: jnp.ndarray) -> jnp.ndarray:
        """
        PURE_JAX: 原 _decode 方法
        
        改动：
        - self.clear_cache() → 创建新的 cache list
        - self.post_quant_conv(z) → conv3d(self.params['post_quant_conv'], z)
        - self.decoder(..., feat_cache=self._feat_map) → decoder_forward(..., feat_cache=cache)
        """
        B, T, H, W, C = z.shape

        # PURE_JAX: 原 self.clear_cache() → 创建新 cache
        num_convs = self._count_decoder_convs()
        cache = [None] * num_convs

        # PURE_JAX: self.post_quant_conv(z) → conv3d
        x = conv3d(self.params['post_quant_conv'], z, padding="SAME")
        
        out = None
        for i in range(T):
            if i == 0:
                # PURE_JAX: self.decoder(...) → decoder_forward(self.params['decoder'], ...)
                out, _, cache = decoder_forward(
                    self.params['decoder'],
                    x[:, i : i + 1, :, :, :],
                    self.config,
                    feat_cache=cache,
                    first_chunk=True,
                )
            else:
                out_, _, cache = decoder_forward(
                    self.params['decoder'],
                    x[:, i : i + 1, :, :, :],
                    self.config,
                    feat_cache=cache,
                    first_chunk=False,
                )
                out = jnp.concatenate([out, out_], axis=1)

        if self.config.patch_size is not None:
            out = unpatchify(out, patch_size=self.config.patch_size)

        out = jnp.clip(out, -1.0, 1.0)
        return out

    def decode(self, z: jnp.ndarray) -> jnp.ndarray:
        """PURE_JAX: 原 decode 方法，基本不变"""
        # 注意：use_slicing 逻辑可按需保留
        return self._decode(z)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        subfolder: Optional[str] = "vae",
        dtype: jnp.dtype = jnp.float32,
        **kwargs,
    ):
        """
        PURE_JAX: 原 from_pretrained 方法
        
        主要改动：
        1. 不再创建 nnx 模块，直接构建 params 字典
        2. 权重转换逻辑保留，但输出格式改变
        3. 无需 nnx.split/nnx.merge
        """
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open
        import json
        import re
        from collections import defaultdict

        # 1. Config
        config_path = hf_hub_download(
            pretrained_model_name_or_path,
            subfolder=subfolder,
            filename="config.json"
        )
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config_dict.update(kwargs)
        config = cls.config_class.from_dict(config_dict)

        # 2. Weights
        ckpt_path = hf_hub_download(
            pretrained_model_name_or_path,
            subfolder=subfolder,
            filename="diffusion_pytorch_model.safetensors"
        )

        pytorch_weights = {}
        with safe_open(ckpt_path, framework="np") as f:
            for k in f.keys():
                pytorch_weights[k] = f.get_tensor(k)

        # 3. Convert weights (PyTorch -> JAX)
        # PURE_JAX: 构建嵌套字典而非 nnx 模块
        def make_nested_dict():
            return defaultdict(make_nested_dict)
        
        params = make_nested_dict()

        def rename_key(key):
            """
            PURE_JAX: 键名转换
            
            与 Flax NNX 版本的区别：
            - conv_in.weight → conv_in.kernel (不是 conv_in.conv.kernel)
            - 因为我们直接用 conv3d 函数，不包装在 WanCausalConv3d 类中
            """
            # CausalConv3d: 原版有 .conv. 中间层，纯 JAX 版去掉
            # PURE_JAX: conv_in.conv.kernel → conv_in.kernel
            key = key.replace("conv_in.bias", "conv_in.bias")
            key = key.replace("conv_in.weight", "conv_in.kernel")
            key = key.replace("conv_out.bias", "conv_out.bias")
            key = key.replace("conv_out.weight", "conv_out.kernel")

            key = re.sub(r"conv(\d+)\.weight", r"conv\1.kernel", key)
            key = re.sub(r"conv(\d+)\.bias", r"conv\1.bias", key)

            key = key.replace("time_conv.weight", "time_conv.kernel")
            key = key.replace("time_conv.bias", "time_conv.bias")

            key = key.replace("quant_conv.weight", "quant_conv.kernel")
            key = key.replace("quant_conv.bias", "quant_conv.bias")
            key = key.replace("post_quant_conv.weight", "post_quant_conv.kernel")
            key = key.replace("post_quant_conv.bias", "post_quant_conv.bias")

            key = key.replace("conv_shortcut.weight", "conv_shortcut.kernel")
            key = key.replace("conv_shortcut.bias", "conv_shortcut.bias")

            # Resample layers: resample.1 是 resample_conv
            key = key.replace("resample.1.weight", "resample_conv.kernel")
            key = key.replace("resample.1.bias", "resample_conv.bias")

            # Attention
            key = key.replace("to_qkv.weight", "to_qkv.kernel")
            key = key.replace("to_qkv.bias", "to_qkv.bias")
            key = key.replace("proj.weight", "proj.kernel")
            key = key.replace("proj.bias", "proj.bias")

            # Norm: weight -> gamma
            key = key.replace(".weight", ".gamma")

            return key

        def set_nested(d, keys, value):
            for key in keys[:-1]:
                d = d[key]
            d[keys[-1]] = value

        def convert_key_to_path(key: str) -> List:
            """将点分隔的键转换为路径列表"""
            # 处理数字索引
            key = re.sub(r'\.(\d+)\.', r'[\1].', key)
            key = re.sub(r'\.(\d+)$', r'[\1]', key)
            
            parts = []
            current = ""
            for char in key:
                if char == '.':
                    if current:
                        parts.append(current)
                        current = ""
                elif char == '[':
                    if current:
                        parts.append(current)
                        current = ""
                    current = "["
                elif char == ']':
                    current += char
                    parts.append(current)
                    current = ""
                else:
                    current += char
            if current:
                parts.append(current)
            
            result = []
            for part in parts:
                if part.startswith('[') and part.endswith(']'):
                    result.append(int(part[1:-1]))
                else:
                    result.append(part)
            
            return result

        for pt_key, pt_tensor in pytorch_weights.items():
            jax_key = rename_key(pt_key)

            # Convert tensor shape
            if "kernel" in jax_key:
                if pt_tensor.ndim == 5:
                    # Conv3d: (Out, In, T, H, W) -> (T, H, W, In, Out)
                    pt_tensor = pt_tensor.transpose(2, 3, 4, 1, 0)
                elif pt_tensor.ndim == 4:
                    # Conv2d: (Out, In, H, W) -> (H, W, In, Out)
                    pt_tensor = pt_tensor.transpose(2, 3, 1, 0)

            if "gamma" in jax_key:
                pt_tensor = pt_tensor.squeeze()

            tensor = jnp.array(pt_tensor, dtype=dtype)
            
            keys = convert_key_to_path(jax_key)
            set_nested(params, keys, tensor)

        # Convert defaultdict to regular dict
        def to_regular_dict(d):
            if isinstance(d, defaultdict):
                d = {k: to_regular_dict(v) for k, v in d.items()}
            return d

        params = to_regular_dict(params)

        # PURE_JAX: 直接返回实例，无需 nnx.merge
        return cls(params, config)
