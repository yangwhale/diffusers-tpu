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

This version removes all Flax NNX dependencies and uses:
- Pure functions with explicit pytree parameters
- jax.lax.conv_general_dilated instead of nnx.Conv
- jax.lax.scan for frame-by-frame decoding
- Explicit cache passing (no mutable state)

Performance improvements over NNX version:
- No pytree=False hack (full JAX tracing optimization)
- No nnx.split/merge overhead
- No module instantiation overhead
- Faster compilation and cache loading
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import PartitionSpec as P

# Type aliases for pytree parameters
Array = jnp.ndarray
Params = Dict[str, Any]
Cache = Optional[Array]
CacheList = List[Cache]

CACHE_T = 2


# =============================================================================
# Sharding Utilities
# =============================================================================

def mark_sharding(inputs: Array, spec: P) -> Array:
    """Apply sharding constraint, silently ignore if not applicable."""
    try:
        return lax.with_sharding_constraint(inputs, spec)
    except (ValueError, Exception):
        return inputs


def try_shard_on_width(x: Array) -> Array:
    """Try to shard on width dimension (NTHWC format)."""
    for spec in [P(None, None, None, ("dp", "tp"), None),
                 P(None, None, None, ("tp",), None),
                 P(None, None, None, ("dp",), None)]:
        try:
            return mark_sharding(x, spec)
        except ValueError:
            continue
    return x


# =============================================================================
# Pure Function Implementations
# =============================================================================

def conv3d(
    params: Params,
    x: Array,
    strides: Tuple[int, int, int] = (1, 1, 1),
    padding: str = "VALID",
) -> Array:
    """
    Pure 3D convolution using lax.conv_general_dilated.
    
    Args:
        params: {'kernel': (T, H, W, In, Out), 'bias': (Out,)}
        x: Input tensor (B, T, H, W, C)
        strides: Convolution strides
        padding: Padding mode
    
    Returns:
        Output tensor (B, T', H', W', Out)
    """
    kernel = params['kernel']
    out = lax.conv_general_dilated(
        lhs=x,
        rhs=kernel,
        window_strides=strides,
        padding=padding,
        dimension_numbers=('NTHWC', 'THWIO', 'NTHWC'),
    )
    if 'bias' in params and params['bias'] is not None:
        out = out + params['bias']
    return out


def conv2d(
    params: Params,
    x: Array,
    strides: Tuple[int, int] = (1, 1),
    padding: Union[str, Tuple] = "VALID",
) -> Array:
    """
    Pure 2D convolution using lax.conv_general_dilated.
    
    Args:
        params: {'kernel': (H, W, In, Out), 'bias': (Out,)}
        x: Input tensor (B, H, W, C)
        strides: Convolution strides
        padding: Padding mode or explicit padding
    
    Returns:
        Output tensor (B, H', W', Out)
    """
    kernel = params['kernel']
    out = lax.conv_general_dilated(
        lhs=x,
        rhs=kernel,
        window_strides=strides,
        padding=padding,
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    if 'bias' in params and params['bias'] is not None:
        out = out + params['bias']
    return out


def conv3d_causal(
    params: Params,
    x: Array,
    cache: Cache,
    padding_config: Tuple[int, ...],
) -> Tuple[Array, Cache]:
    """
    Causal 3D convolution with cache support.
    
    Args:
        params: {'kernel': (T, H, W, In, Out), 'bias': (Out,)}
        x: Input tensor (B, T, H, W, C)
        cache: Previous frames cache (B, cache_T, H, W, C) or None
        padding_config: (W_left, W_right, H_left, H_right, T_left, T_right)
    
    Returns:
        output: Convolution result
        new_cache: Updated cache for next call
    """
    padding = list(padding_config)
    
    # Concatenate cache if available
    if cache is not None and padding[4] > 0:
        x = jnp.concatenate([cache, x], axis=1)
        padding[4] -= cache.shape[1]
    
    # Build new cache before padding
    new_cache = x[:, -CACHE_T:, :, :, :]
    
    # Apply padding (T, H, W dimensions)
    pad_width = [
        (0, 0),                    # Batch
        (padding[4], padding[5]),  # Time
        (padding[2], padding[3]),  # Height
        (padding[0], padding[1]),  # Width
        (0, 0),                    # Channels
    ]
    x = jnp.pad(x, pad_width, mode='constant', constant_values=0)
    
    # Apply sharding
    x = try_shard_on_width(x)
    
    # Convolution
    out = conv3d(params, x, strides=(1, 1, 1), padding="VALID")
    
    return out, new_cache


def rms_norm(params: Params, x: Array) -> Array:
    """
    RMS normalization.
    
    Args:
        params: {'gamma': (dim,), 'bias': (dim,) optional}
        x: Input tensor (..., dim)
    
    Returns:
        Normalized tensor
    """
    scale = jnp.sqrt(x.shape[-1]).astype(x.dtype)
    norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
    norm = jnp.maximum(norm, 1e-12)
    x_normalized = x / norm
    out = x_normalized * scale * params['gamma']
    if 'bias' in params and params['bias'] is not None:
        out = out + params['bias']
    return out


# =============================================================================
# AvgDown3D / DupUp3D Pure Functions
# =============================================================================

def avg_down_3d(
    x: Array,
    in_channels: int,
    out_channels: int,
    factor_t: int,
    factor_s: int = 1,
) -> Array:
    """
    Average pooling based downsampling for 3D data.
    
    Args:
        x: Input (B, T, H, W, C)
        in_channels, out_channels: Channel dimensions
        factor_t, factor_s: Temporal and spatial downsampling factors
    """
    B, T, H, W, C = x.shape
    factor = factor_t * factor_s * factor_s
    group_size = in_channels * factor // out_channels
    
    pad_t = (factor_t - T % factor_t) % factor_t
    if pad_t > 0:
        pad_width = [(0, 0), (pad_t, 0), (0, 0), (0, 0), (0, 0)]
        x = jnp.pad(x, pad_width, mode='constant', constant_values=0)
        T = x.shape[1]
    
    x = x.reshape(B, T//factor_t, factor_t, H//factor_s, factor_s, W//factor_s, factor_s, C)
    x = x.transpose(0, 1, 3, 5, 7, 2, 4, 6)
    x = x.reshape(B, T//factor_t, H//factor_s, W//factor_s, C * factor)
    x = x.reshape(B, T//factor_t, H//factor_s, W//factor_s, out_channels, group_size)
    x = jnp.mean(x, axis=-1)
    return x


def dup_up_3d(
    x: Array,
    in_channels: int,
    out_channels: int,
    factor_t: int,
    factor_s: int = 1,
    first_chunk: bool = False,
) -> Array:
    """
    Duplication based upsampling for 3D data.
    
    Args:
        x: Input (B, T, H, W, C)
        in_channels, out_channels: Channel dimensions
        factor_t, factor_s: Temporal and spatial upsampling factors
        first_chunk: Whether this is the first chunk (affects output slicing)
    """
    B, T, H, W, C = x.shape
    factor = factor_t * factor_s * factor_s
    repeats = out_channels * factor // in_channels
    
    x = jnp.repeat(x[:, :, :, :, :, None], repeats, axis=5).reshape(B, T, H, W, C * repeats)
    x = x.reshape(B, T, H, W, out_channels, factor_t, factor_s, factor_s)
    x = x.transpose(0, 1, 5, 2, 6, 3, 7, 4)
    x = x.reshape(B, T * factor_t, H * factor_s, W * factor_s, out_channels)
    
    if first_chunk:
        x = x[:, factor_t - 1:, :, :, :]
    return x


# =============================================================================
# Resample Pure Functions
# =============================================================================

def resample_upsample_3d(
    params: Params,
    x: Array,
    cache: CacheList,
    cache_idx: int,
    dim: int,
) -> Tuple[Array, int, CacheList]:
    """
    Upsample3D: time_conv (temporal) + spatial upsample + resample_conv.
    """
    B, T, H, W, C = x.shape
    
    # Time convolution
    if cache[cache_idx] is None:
        cache[cache_idx] = (None,)
        cache_idx += 1
    else:
        cache_x = x[:, -CACHE_T:, :, :, :]
        if cache_x.shape[1] < 2 and cache[cache_idx] is not None and cache[cache_idx][0] is not None:
            cache_x = jnp.concatenate([cache[cache_idx][0][:, -1:, :, :, :], cache_x], axis=1)
        if cache_x.shape[1] < 2 and cache[cache_idx] is not None and cache[cache_idx][0] is None:
            cache_x = jnp.concatenate([jnp.zeros_like(cache_x), cache_x], axis=1)
        
        time_conv_params = params['time_conv']
        padding_config = (0, 0, 0, 0, 2, 0)  # (1, 0, 0) padding in causal conv
        
        if cache[cache_idx][0] is None:
            x, _ = conv3d_causal(time_conv_params, x, None, padding_config)
        else:
            x, _ = conv3d_causal(time_conv_params, x, cache[cache_idx][0], padding_config)
        
        cache[cache_idx] = (cache_x,)
        cache_idx += 1
        
        # Reshape for temporal upsample
        x = x.reshape(B, T, H, W, 2, dim)
        x = x.transpose(0, 1, 4, 2, 3, 5).reshape(B, T * 2, H, W, dim)
    
    T_curr = x.shape[1]
    C_curr = x.shape[4]
    x = x.reshape(B * T_curr, H, W, C_curr)
    
    # Spatial upsample (nearest neighbor 2x)
    x = jax.image.resize(x, (B * T_curr, H * 2, W * 2, C_curr), method='nearest')
    
    # Resample conv (2D)
    x = conv2d(params['resample_conv'], x, strides=(1, 1), padding=((1, 1), (1, 1)))
    
    H_new, W_new = x.shape[1], x.shape[2]
    x = x.reshape(B, T_curr, H_new, W_new, x.shape[-1])
    
    return x, cache_idx, cache


def resample_upsample_2d(
    params: Params,
    x: Array,
) -> Array:
    """
    Upsample2D: spatial upsample + resample_conv.
    """
    B, T, H, W, C = x.shape
    x = x.reshape(B * T, H, W, C)
    
    # Spatial upsample (nearest neighbor 2x)
    x = jax.image.resize(x, (B * T, H * 2, W * 2, C), method='nearest')
    
    # Resample conv (2D)
    x = conv2d(params['resample_conv'], x, strides=(1, 1), padding=((1, 1), (1, 1)))
    
    H_new, W_new = x.shape[1], x.shape[2]
    x = x.reshape(B, T, H_new, W_new, x.shape[-1])
    
    return x


def resample_downsample_3d(
    params: Params,
    x: Array,
    cache: CacheList,
    cache_idx: int,
) -> Tuple[Array, int, CacheList]:
    """
    Downsample3D: resample_conv (spatial) + time_conv (temporal).
    """
    B, T, H, W, C = x.shape
    x = x.reshape(B * T, H, W, C)
    
    # Resample conv (2D) with stride 2
    x = conv2d(params['resample_conv'], x, strides=(2, 2), padding=((0, 1), (0, 1)))
    
    H_new, W_new = x.shape[1], x.shape[2]
    x = x.reshape(B, T, H_new, W_new, x.shape[-1])
    
    # Time convolution
    if cache[cache_idx] is None:
        cache[cache_idx] = x
        cache_idx += 1
    else:
        cache_x = x[:, -1:, :, :, :]
        time_conv_params = params['time_conv']
        padding_config = (0, 0, 0, 0, 0, 0)  # No padding, stride 2 in time
        
        x_concat = jnp.concatenate([cache[cache_idx][:, -1:, :, :, :], x], axis=1)
        x, _ = conv3d_causal(time_conv_params, x_concat, None, padding_config)
        
        cache[cache_idx] = cache_x
        cache_idx += 1
    
    return x, cache_idx, cache


def resample_downsample_2d(
    params: Params,
    x: Array,
) -> Array:
    """
    Downsample2D: resample_conv with stride 2.
    """
    B, T, H, W, C = x.shape
    x = x.reshape(B * T, H, W, C)
    
    # Resample conv (2D) with stride 2
    x = conv2d(params['resample_conv'], x, strides=(2, 2), padding=((0, 1), (0, 1)))
    
    H_new, W_new = x.shape[1], x.shape[2]
    x = x.reshape(B, T, H_new, W_new, x.shape[-1])
    
    return x


# =============================================================================
# Residual Block Pure Function
# =============================================================================

def residual_block(
    params: Params,
    x: Array,
    cache: CacheList,
    cache_idx: int,
) -> Tuple[Array, int, CacheList]:
    """
    Residual block with two causal convolutions.
    
    Args:
        params: {
            'norm1': RMS norm params,
            'conv1': Conv3d params,
            'norm2': RMS norm params,
            'conv2': Conv3d params,
            'conv_shortcut': Conv3d params (optional),
        }
        x: Input (B, T, H, W, C)
        cache: Cache list
        cache_idx: Current cache index
    
    Returns:
        output, new_cache_idx, updated_cache
    """
    # Shortcut
    if 'conv_shortcut' in params and params['conv_shortcut'] is not None:
        h, _ = conv3d_causal(params['conv_shortcut'], x, None, (0, 0, 0, 0, 0, 0))
    else:
        h = x
    
    # First norm + activation + conv
    x = rms_norm(params['norm1'], x)
    x = jax.nn.silu(x)
    
    cache_x = x[:, -CACHE_T:, :, :, :]
    if cache_x.shape[1] < 2 and cache[cache_idx] is not None:
        cache_x = jnp.concatenate([cache[cache_idx][:, -1:, :, :, :], cache_x], axis=1)
    
    padding_config = (1, 1, 1, 1, 2, 0)  # padding=1 for 3x3x3 kernel
    x, _ = conv3d_causal(params['conv1'], x, cache[cache_idx], padding_config)
    cache[cache_idx] = cache_x
    cache_idx += 1
    
    # Second norm + activation + conv
    x = rms_norm(params['norm2'], x)
    x = jax.nn.silu(x)
    
    cache_x = x[:, -CACHE_T:, :, :, :]
    if cache_x.shape[1] < 2 and cache[cache_idx] is not None:
        cache_x = jnp.concatenate([cache[cache_idx][:, -1:, :, :, :], cache_x], axis=1)
    
    x, _ = conv3d_causal(params['conv2'], x, cache[cache_idx], padding_config)
    cache[cache_idx] = cache_x
    cache_idx += 1
    
    return x + h, cache_idx, cache


# =============================================================================
# Attention Block Pure Function
# =============================================================================

def attention_block(
    params: Params,
    x: Array,
) -> Array:
    """
    Causal self-attention with a single head.
    
    Args:
        params: {
            'norm': RMS norm params,
            'to_qkv': Conv2d params (1x1),
            'proj': Conv2d params (1x1),
        }
        x: Input (B, T, H, W, C)
    
    Returns:
        Output (B, T, H, W, C)
    """
    identity = x
    B, T, H, W, C = x.shape
    
    x = x.reshape(B * T, H, W, C)
    x = rms_norm(params['norm'], x)
    
    # Compute Q, K, V
    qkv = conv2d(params['to_qkv'], x, strides=(1, 1), padding="SAME")
    qkv = qkv.reshape(B * T, H * W, 3 * C)
    q, k, v = jnp.split(qkv, 3, axis=-1)
    
    # Add head dimension (1 head)
    q = q[:, None, :, :]
    k = k[:, None, :, :]
    v = v[:, None, :, :]
    
    # Scaled dot-product attention
    scale = 1.0 / jnp.sqrt(jnp.array(C, dtype=x.dtype))
    attn_weights = jnp.matmul(q, k.swapaxes(-1, -2)) * scale
    attn_weights = jax.nn.softmax(attn_weights, axis=-1)
    x = jnp.matmul(attn_weights, v)
    
    x = x.squeeze(1).reshape(B * T, H, W, C)
    
    # Output projection
    x = conv2d(params['proj'], x, strides=(1, 1), padding="SAME")
    x = x.reshape(B, T, H, W, C)
    
    return x + identity


# =============================================================================
# Mid Block Pure Function
# =============================================================================

def mid_block(
    params: Params,
    x: Array,
    cache: CacheList,
    cache_idx: int,
) -> Tuple[Array, int, CacheList]:
    """
    Middle block: resnet -> (attention -> resnet) * num_layers.
    
    Args:
        params: {
            'resnets': dict with integer keys (0, 1, ...) for residual block params,
            'attentions': dict with integer keys for attention block params,
        }
    """
    resnets = params['resnets']
    attentions = params['attentions']
    
    # Get number of resnets (handle both dict with int keys and list)
    if isinstance(resnets, dict):
        resnet_keys = sorted([k for k in resnets.keys() if isinstance(k, int)])
        num_resnets = len(resnet_keys)
    else:
        num_resnets = len(resnets)
        resnet_keys = list(range(num_resnets))
    
    if isinstance(attentions, dict):
        attn_keys = sorted([k for k in attentions.keys() if isinstance(k, int)])
    else:
        attn_keys = list(range(len(attentions)))
    
    # First resnet
    x, cache_idx, cache = residual_block(resnets[resnet_keys[0]], x, cache, cache_idx)
    
    # Attention + resnet pairs
    for i, attn_key in enumerate(attn_keys):
        resnet_key = resnet_keys[i + 1]
        attn_params = attentions[attn_key]
        resnet_params = resnets[resnet_key]
        
        if attn_params is not None:
            x = attention_block(attn_params, x)
        x, cache_idx, cache = residual_block(resnet_params, x, cache, cache_idx)
    
    return x, cache_idx, cache


# =============================================================================
# Encoder Pure Functions
# =============================================================================

def encoder_down_block_residual(
    params: Params,
    x: Array,
    cache: CacheList,
    cache_idx: int,
    config: Dict,
) -> Tuple[Array, int, CacheList]:
    """
    Residual downsampling block for encoder.
    """
    x_copy = x
    
    # Residual blocks
    for resnet_params in iter_dict_values(params['resnets']):
        x, cache_idx, cache = residual_block(resnet_params, x, cache, cache_idx)
    
    # Downsample if needed
    if 'downsampler' in params and params['downsampler'] is not None:
        mode = config['mode']
        if mode == 'downsample3d':
            x, cache_idx, cache = resample_downsample_3d(
                params['downsampler'], x, cache, cache_idx
            )
        else:  # downsample2d
            x = resample_downsample_2d(params['downsampler'], x)
    
    # Shortcut with average pooling
    x_short = avg_down_3d(
        x_copy,
        config['in_dim'],
        config['out_dim'],
        config['factor_t'],
        config['factor_s'],
    )
    
    return x + x_short, cache_idx, cache


def encoder_forward(
    params: Params,
    x: Array,
    cache: CacheList,
    config: Dict,
) -> Tuple[Array, int, CacheList]:
    """
    Encoder forward pass.
    
    Args:
        params: Encoder parameters
        x: Input (B, T, H, W, C)
        cache: Feature cache list
        config: Encoder configuration
    
    Returns:
        output, cache_idx, updated_cache
    """
    cache_idx = 0
    
    # Conv in
    cache_x = x[:, -CACHE_T:, :, :, :]
    if cache_x.shape[1] < 2 and cache[cache_idx] is not None:
        cache_x = jnp.concatenate([cache[cache_idx][:, -1:, :, :, :], cache_x], axis=1)
    
    padding_config = (1, 1, 1, 1, 2, 0)
    x, _ = conv3d_causal(params['conv_in'], x, cache[cache_idx], padding_config)
    cache[cache_idx] = cache_x
    cache_idx += 1
    
    # Down blocks
    down_blocks = params['down_blocks']
    if isinstance(down_blocks, dict):
        down_block_keys = sorted([k for k in down_blocks.keys() if isinstance(k, int)])
    else:
        down_block_keys = list(range(len(down_blocks)))
    
    for i, key in enumerate(down_block_keys):
        down_block_params = down_blocks[key]
        block_config = config['down_blocks'][i]
        if block_config['type'] == 'residual':
            x, cache_idx, cache = encoder_down_block_residual(
                down_block_params, x, cache, cache_idx, block_config
            )
        else:
            # Non-residual blocks (residual blocks + optional attention + resample)
            layers = down_block_params['layers']
            if isinstance(layers, dict):
                layer_keys = sorted([k for k in layers.keys() if isinstance(k, int)])
            else:
                layer_keys = list(range(len(layers)))
            
            for j, layer_key in enumerate(layer_keys):
                layer_params = layers[layer_key]
                layer_type = block_config['layers'][j]['type']
                if layer_type == 'residual':
                    x, cache_idx, cache = residual_block(layer_params, x, cache, cache_idx)
                elif layer_type == 'attention':
                    x = attention_block(layer_params, x)
                elif layer_type == 'downsample3d':
                    x, cache_idx, cache = resample_downsample_3d(layer_params, x, cache, cache_idx)
                elif layer_type == 'downsample2d':
                    x = resample_downsample_2d(layer_params, x)
    
    # Mid block
    x, cache_idx, cache = mid_block(params['mid_block'], x, cache, cache_idx)
    
    # Output
    x = rms_norm(params['norm_out'], x)
    x = jax.nn.silu(x)
    
    cache_x = x[:, -CACHE_T:, :, :, :]
    if cache_x.shape[1] < 2 and cache[cache_idx] is not None:
        cache_x = jnp.concatenate([cache[cache_idx][:, -1:, :, :, :], cache_x], axis=1)
    
    x, _ = conv3d_causal(params['conv_out'], x, cache[cache_idx], padding_config)
    cache[cache_idx] = cache_x
    cache_idx += 1
    
    return x, cache_idx, cache


# =============================================================================
# Decoder Pure Functions
# =============================================================================

def iter_dict_values(d):
    """Iterate over dict values in sorted key order (for int keys)."""
    if isinstance(d, dict):
        keys = sorted([k for k in d.keys() if isinstance(k, int)])
        for k in keys:
            yield d[k]
    else:
        for v in d:
            yield v


def decoder_up_block_residual(
    params: Params,
    x: Array,
    cache: CacheList,
    cache_idx: int,
    config: Dict,
    first_chunk: bool,
) -> Tuple[Array, int, CacheList]:
    """
    Residual upsampling block for decoder.
    """
    x_copy = x
    
    # Residual blocks
    for resnet_params in iter_dict_values(params['resnets']):
        x, cache_idx, cache = residual_block(resnet_params, x, cache, cache_idx)
    
    # Upsample if needed
    if 'upsampler' in params and params['upsampler'] is not None:
        mode = config['mode']
        if mode == 'upsample3d':
            x, cache_idx, cache = resample_upsample_3d(
                params['upsampler'], x, cache, cache_idx, config['dim']
            )
        else:  # upsample2d
            x = resample_upsample_2d(params['upsampler'], x)
    
    # Shortcut with duplication upsampling
    if config.get('up_flag', False):
        x_short = dup_up_3d(
            x_copy,
            config['in_dim'],
            config['out_dim'],
            config['factor_t'],
            config['factor_s'],
            first_chunk=first_chunk,
        )
        return x + x_short, cache_idx, cache
    
    return x, cache_idx, cache


def decoder_up_block(
    params: Params,
    x: Array,
    cache: CacheList,
    cache_idx: int,
    config: Dict,
) -> Tuple[Array, int, CacheList]:
    """
    Standard upsampling block for decoder.
    """
    # Residual blocks
    for resnet_params in iter_dict_values(params['resnets']):
        x, cache_idx, cache = residual_block(resnet_params, x, cache, cache_idx)
    
    # Upsample if needed
    if 'upsamplers' in params and len(params['upsamplers']) > 0:
        mode = config['mode']
        if mode == 'upsample3d':
            x, cache_idx, cache = resample_upsample_3d(
                params['upsamplers'][0], x, cache, cache_idx, config['dim']
            )
        elif mode == 'upsample2d':
            x = resample_upsample_2d(params['upsamplers'][0], x)
    
    return x, cache_idx, cache


def decoder_forward(
    params: Params,
    x: Array,
    cache: CacheList,
    config: Dict,
    first_chunk: bool = False,
) -> Tuple[Array, int, CacheList]:
    """
    Decoder forward pass for a single frame or chunk.
    
    Args:
        params: Decoder parameters
        x: Input latent (B, 1, H, W, C) - single frame
        cache: Feature cache list
        config: Decoder configuration
        first_chunk: Whether this is the first chunk
    
    Returns:
        output, cache_idx, updated_cache
    """
    cache_idx = 0
    
    # Conv in
    cache_x = x[:, -CACHE_T:, :, :, :]
    if cache_x.shape[1] < 2 and cache[cache_idx] is not None:
        cache_x = jnp.concatenate([cache[cache_idx][:, -1:, :, :, :], cache_x], axis=1)
    
    padding_config = (1, 1, 1, 1, 2, 0)
    x, _ = conv3d_causal(params['conv_in'], x, cache[cache_idx], padding_config)
    cache[cache_idx] = cache_x
    cache_idx += 1
    
    # Mid block
    x, cache_idx, cache = mid_block(params['mid_block'], x, cache, cache_idx)
    
    # Up blocks
    up_blocks = params['up_blocks']
    if isinstance(up_blocks, dict):
        up_block_keys = sorted([k for k in up_blocks.keys() if isinstance(k, int)])
    else:
        up_block_keys = list(range(len(up_blocks)))
    
    for i, key in enumerate(up_block_keys):
        up_block_params = up_blocks[key]
        block_config = config['up_blocks'][i]
        if block_config['type'] == 'residual':
            x, cache_idx, cache = decoder_up_block_residual(
                up_block_params, x, cache, cache_idx, block_config, first_chunk
            )
        else:
            x, cache_idx, cache = decoder_up_block(
                up_block_params, x, cache, cache_idx, block_config
            )
    
    # Output
    x = rms_norm(params['norm_out'], x)
    x = jax.nn.silu(x)
    
    cache_x = x[:, -CACHE_T:, :, :, :]
    if cache_x.shape[1] < 2 and cache[cache_idx] is not None:
        cache_x = jnp.concatenate([cache[cache_idx][:, -1:, :, :, :], cache_x], axis=1)
    
    x, _ = conv3d_causal(params['conv_out'], x, cache[cache_idx], padding_config)
    cache[cache_idx] = cache_x
    cache_idx += 1
    
    # Replicate to all devices
    x = mark_sharding(x, P())
    
    return x, cache_idx, cache


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class AutoencoderKLWanConfig:
    """Configuration class for AutoencoderKLWan (Pure JAX version)."""
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
        for key in ["dim_mult", "attn_scales", "temperal_downsample", "latents_mean", "latents_std"]:
            if key in filtered_dict:
                filtered_dict[key] = tuple(filtered_dict[key])
        return cls(**filtered_dict)


# =============================================================================
# Patchify / Unpatchify
# =============================================================================

def patchify(x: Array, patch_size: int) -> Array:
    """Convert image to patches."""
    if patch_size == 1:
        return x
    
    B, T, H, W, C = x.shape
    if H % patch_size != 0 or W % patch_size != 0:
        raise ValueError(f"Height ({H}) and width ({W}) must be divisible by patch_size ({patch_size})")
    
    x = x.reshape(B, T, H // patch_size, patch_size, W // patch_size, patch_size, C)
    x = x.transpose(0, 1, 2, 4, 3, 5, 6)
    x = x.reshape(B, T, H // patch_size, W // patch_size, C * patch_size * patch_size)
    return x


def unpatchify(x: Array, patch_size: int) -> Array:
    """Convert patches back to image."""
    if patch_size == 1:
        return x
    
    B, T, H, W, C_patches = x.shape
    C = C_patches // (patch_size * patch_size)
    
    x = x.reshape(B, T, H, W, patch_size, patch_size, C)
    x = x.transpose(0, 1, 2, 4, 3, 5, 6)
    x = x.reshape(B, T, H * patch_size, W * patch_size, C)
    return x


# =============================================================================
# Main VAE Functions
# =============================================================================

def count_decoder_convs(config: AutoencoderKLWanConfig) -> int:
    """Count the number of causal conv3d layers in decoder."""
    count = 1  # conv_in
    
    # Mid block: 1 + num_layers resnets, each resnet has 2 convs
    count += (1 + 1) * 2  # 2 resnets for num_layers=1
    
    # Up blocks
    dim_mult = list(config.dim_mult)
    num_res_blocks = config.num_res_blocks
    temperal_upsample = list(reversed(config.temperal_downsample))
    
    for i in range(len(dim_mult)):
        # Each up block has (num_res_blocks + 1) resnets
        count += (num_res_blocks + 1) * 2
        
        # Upsample has time_conv (1 conv) if 3d
        up_flag = i != len(dim_mult) - 1
        if up_flag and temperal_upsample[i]:
            count += 1  # time_conv in upsample3d
    
    count += 1  # conv_out
    
    return count


def count_encoder_convs(config: AutoencoderKLWanConfig) -> int:
    """Count the number of causal conv3d layers in encoder."""
    count = 1  # conv_in
    
    dim_mult = list(config.dim_mult)
    num_res_blocks = config.num_res_blocks
    temperal_downsample = list(config.temperal_downsample)
    
    for i in range(len(dim_mult)):
        if config.is_residual:
            # Residual down block has num_res_blocks resnets
            count += num_res_blocks * 2
            # Downsample has time_conv if 3d
            if i != len(dim_mult) - 1 and temperal_downsample[i]:
                count += 1
        else:
            # Non-residual: num_res_blocks resnets
            count += num_res_blocks * 2
            # Downsample
            if i != len(dim_mult) - 1 and temperal_downsample[i]:
                count += 1
    
    # Mid block
    count += (1 + 1) * 2
    
    count += 1  # conv_out
    
    return count


def init_cache(num_convs: int) -> CacheList:
    """Initialize cache list with None values."""
    return [None] * num_convs


def vae_encode(
    params: Params,
    x: Array,
    config: AutoencoderKLWanConfig,
) -> Tuple[Array, Array]:
    """
    Encode input to latent space.
    
    Args:
        params: VAE parameters
        x: Input video (B, T, H, W, C) in NTHWC format
        config: VAE configuration
    
    Returns:
        mean, logvar of latent distribution
    """
    B, T, H, W, C = x.shape
    
    num_enc_convs = count_encoder_convs(config)
    cache = init_cache(num_enc_convs)
    
    if config.patch_size is not None:
        x = patchify(x, config.patch_size)
    
    # Build encoder config
    enc_config = build_encoder_config(config)
    
    # Frame-by-frame encoding
    iter_ = 1 + (T - 1) // 4
    out = None
    
    for i in range(iter_):
        if i == 0:
            out, _, cache = encoder_forward(
                params['encoder'], x[:, :1, :, :, :], cache, enc_config
            )
        else:
            start_idx = 1 + 4 * (i - 1)
            end_idx = 1 + 4 * i
            out_, _, cache = encoder_forward(
                params['encoder'], x[:, start_idx:end_idx, :, :, :], cache, enc_config
            )
            out = jnp.concatenate([out, out_], axis=1)
    
    # Quant conv
    enc = conv3d(params['quant_conv'], out, padding="SAME")
    
    # Split to mean and logvar
    mean, logvar = jnp.split(enc, 2, axis=-1)
    
    return mean, logvar


def vae_decode(
    params: Params,
    z: Array,
    config: AutoencoderKLWanConfig,
) -> Array:
    """
    Decode latent to video.
    
    Args:
        params: VAE parameters
        z: Latent tensor (B, T, H, W, C) in NTHWC format
        config: VAE configuration
    
    Returns:
        Decoded video (B, T_out, H_out, W_out, C_out)
    """
    B, T, H, W, C = z.shape
    
    num_dec_convs = count_decoder_convs(config)
    cache = init_cache(num_dec_convs)
    
    # Post quant conv
    x = conv3d(params['post_quant_conv'], z, padding="SAME")
    
    # Build decoder config
    dec_config = build_decoder_config(config)
    
    # Frame-by-frame decoding
    out = None
    for i in range(T):
        if i == 0:
            out, _, cache = decoder_forward(
                params['decoder'],
                x[:, i:i+1, :, :, :],
                cache,
                dec_config,
                first_chunk=True,
            )
        else:
            out_, _, cache = decoder_forward(
                params['decoder'],
                x[:, i:i+1, :, :, :],
                cache,
                dec_config,
                first_chunk=False,
            )
            out = jnp.concatenate([out, out_], axis=1)
    
    # Unpatchify if needed
    if config.patch_size is not None:
        out = unpatchify(out, config.patch_size)
    
    return jnp.clip(out, -1.0, 1.0)


# =============================================================================
# Configuration Builders
# =============================================================================

def build_encoder_config(config: AutoencoderKLWanConfig) -> Dict:
    """Build encoder block configuration."""
    dim_mult = list(config.dim_mult)
    dims = [config.base_dim * u for u in [1] + dim_mult]
    temperal_downsample = list(config.temperal_downsample)
    
    down_blocks = []
    for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
        down_flag = i != len(dim_mult) - 1
        temp_down = temperal_downsample[i] if down_flag else False
        
        if config.is_residual:
            down_blocks.append({
                'type': 'residual',
                'in_dim': in_dim,
                'out_dim': out_dim,
                'factor_t': 2 if temp_down else 1,
                'factor_s': 2 if down_flag else 1,
                'mode': 'downsample3d' if temp_down else ('downsample2d' if down_flag else None),
            })
        else:
            layers = []
            for _ in range(config.num_res_blocks):
                layers.append({'type': 'residual', 'in_dim': in_dim, 'out_dim': out_dim})
                in_dim = out_dim
            if down_flag:
                mode = 'downsample3d' if temp_down else 'downsample2d'
                layers.append({'type': mode})
            down_blocks.append({'type': 'standard', 'layers': layers})
    
    return {'down_blocks': down_blocks}


def build_decoder_config(config: AutoencoderKLWanConfig) -> Dict:
    """Build decoder block configuration."""
    decoder_base_dim = config.decoder_base_dim or config.base_dim
    dim_mult = list(config.dim_mult)
    dims = [decoder_base_dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
    temperal_upsample = list(reversed(config.temperal_downsample))
    
    up_blocks = []
    for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
        if i > 0 and not config.is_residual:
            in_dim = in_dim // 2
        
        up_flag = i != len(dim_mult) - 1
        temp_up = temperal_upsample[i] if up_flag else False
        
        if config.is_residual:
            up_blocks.append({
                'type': 'residual',
                'in_dim': in_dim,
                'out_dim': out_dim,
                'factor_t': 2 if temp_up else 1,
                'factor_s': 2 if up_flag else 1,
                'up_flag': up_flag,
                'dim': out_dim,
                'mode': 'upsample3d' if temp_up else ('upsample2d' if up_flag else None),
            })
        else:
            mode = None
            if up_flag:
                mode = 'upsample3d' if temp_up else 'upsample2d'
            up_blocks.append({
                'type': 'standard',
                'dim': out_dim,
                'mode': mode,
            })
    
    return {'up_blocks': up_blocks}


# =============================================================================
# Weight Loading
# =============================================================================

def load_vae_params(
    pretrained_model_name_or_path: str,
    subfolder: Optional[str] = "vae",
    dtype: jnp.dtype = jnp.float32,
) -> Tuple[Params, AutoencoderKLWanConfig]:
    """
    Load VAE parameters from pretrained checkpoint.
    
    Args:
        pretrained_model_name_or_path: HuggingFace model ID or local path
        subfolder: Subfolder containing VAE files
        dtype: Target dtype for parameters
    
    Returns:
        params: Nested dict of parameters
        config: VAE configuration
    """
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    import json
    import re
    
    # 1. Load config
    config_path = hf_hub_download(
        pretrained_model_name_or_path,
        subfolder=subfolder,
        filename="config.json"
    )
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    config = AutoencoderKLWanConfig.from_dict(config_dict)
    
    # 2. Load weights
    ckpt_path = hf_hub_download(
        pretrained_model_name_or_path,
        subfolder=subfolder,
        filename="diffusion_pytorch_model.safetensors"
    )
    
    pytorch_weights = {}
    with safe_open(ckpt_path, framework="np") as f:
        for k in f.keys():
            pytorch_weights[k] = f.get_tensor(k)
    
    # 3. Convert PyTorch weights to JAX format
    params = convert_pytorch_to_jax_params(pytorch_weights, config, dtype)
    
    return params, config


def convert_pytorch_to_jax_params(
    pytorch_weights: Dict[str, Any],
    config: AutoencoderKLWanConfig,
    dtype: jnp.dtype,
) -> Params:
    """
    Convert PyTorch weight dict to JAX parameter pytree.
    
    Handles:
    - Weight key renaming (following Flax NNX conventions)
    - Tensor transposition for different conventions
    - Dtype conversion
    """
    import re
    from collections import defaultdict
    
    def make_nested_dict():
        return defaultdict(make_nested_dict)
    
    params = make_nested_dict()
    
    def set_nested(d, keys, value):
        for key in keys[:-1]:
            d = d[key]
        d[keys[-1]] = value
    
    def rename_key(key: str) -> str:
        """
        Rename PyTorch key to JAX conventions.
        Based on Flax NNX version's rename_key function.
        """
        # CausalConv3d internal conv
        key = key.replace("conv_in.bias", "conv_in.bias")
        key = key.replace("conv_in.weight", "conv_in.kernel")
        key = key.replace("conv_out.bias", "conv_out.bias")
        key = key.replace("conv_out.weight", "conv_out.kernel")
        
        # Numbered conv layers (conv1, conv2, etc.)
        key = re.sub(r"conv(\d+)\.weight", r"conv\1.kernel", key)
        key = re.sub(r"conv(\d+)\.bias", r"conv\1.bias", key)
        
        # Time conv
        key = key.replace("time_conv.weight", "time_conv.kernel")
        key = key.replace("time_conv.bias", "time_conv.bias")
        
        # Quant conv
        key = key.replace("quant_conv.weight", "quant_conv.kernel")
        key = key.replace("quant_conv.bias", "quant_conv.bias")
        key = key.replace("post_quant_conv.weight", "post_quant_conv.kernel")
        key = key.replace("post_quant_conv.bias", "post_quant_conv.bias")
        
        # Conv shortcut
        key = key.replace("conv_shortcut.weight", "conv_shortcut.kernel")
        key = key.replace("conv_shortcut.bias", "conv_shortcut.bias")
        
        # Resample layers - IMPORTANT: resample.1 -> resample_conv
        key = key.replace("resample.1.weight", "resample_conv.kernel")
        key = key.replace("resample.1.bias", "resample_conv.bias")
        
        # Attention layers
        key = key.replace("to_qkv.weight", "to_qkv.kernel")
        key = key.replace("to_qkv.bias", "to_qkv.bias")
        key = key.replace("proj.weight", "proj.kernel")
        key = key.replace("proj.bias", "proj.bias")
        
        # Norm layers: weight -> gamma
        key = key.replace(".weight", ".gamma")
        
        return key
    
    def convert_key(key: str) -> List[str]:
        """Convert renamed key to list of nested keys."""
        # Replace module indices with list indices
        key = re.sub(r'\.(\d+)\.', r'[\1].', key)
        key = re.sub(r'\.(\d+)$', r'[\1]', key)
        
        # Split by dots
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
        
        # Convert to proper keys
        result = []
        for part in parts:
            if part.startswith('[') and part.endswith(']'):
                result.append(int(part[1:-1]))
            else:
                result.append(part)
        
        return result
    
    for pt_key, pt_tensor in pytorch_weights.items():
        # First apply key renaming
        jax_key = rename_key(pt_key)
        
        # Convert tensor shape
        tensor = pt_tensor
        if 'kernel' in jax_key:
            if tensor.ndim == 5:
                # Conv3d: (Out, In, T, H, W) -> (T, H, W, In, Out)
                tensor = tensor.transpose(2, 3, 4, 1, 0)
            elif tensor.ndim == 4:
                # Conv2d: (Out, In, H, W) -> (H, W, In, Out)
                tensor = tensor.transpose(2, 3, 1, 0)
        
        if 'gamma' in jax_key:
            tensor = tensor.squeeze()
        
        # Convert to JAX array
        tensor = jnp.array(tensor, dtype=dtype)
        
        # Set in nested dict
        keys = convert_key(jax_key)
        set_nested(params, keys, tensor)
    
    # Convert defaultdict to regular dict
    def to_regular_dict(d):
        if isinstance(d, defaultdict):
            d = {k: to_regular_dict(v) for k, v in d.items()}
        return d
    
    return to_regular_dict(params)


# =============================================================================
# JIT-Compiled Decode Function
# =============================================================================

@functools.partial(jax.jit, static_argnums=(2,))
def decode_jit(
    params: Params,
    z: Array,
    config_hash: int,  # Use hash for static arg (config is not hashable)
    config_dict: Dict,  # Pass config as dict
) -> Array:
    """
    JIT-compiled decode function.
    
    Note: config is passed as both a hash (for cache key) and dict (for values).
    """
    config = AutoencoderKLWanConfig.from_dict(config_dict)
    return vae_decode(params, z, config)


# =============================================================================
# High-Level API (compatible with original interface)
# =============================================================================

class AutoencoderKLWan:
    """
    Pure JAX VAE with an object-oriented interface for compatibility.
    
    Unlike the NNX version, this is just a thin wrapper around pure functions.
    All actual computation uses the pure functions above.
    """
    
    def __init__(self, params: Params, config: AutoencoderKLWanConfig):
        self.params = params
        self.config = config
        self._config_dict = {
            'base_dim': config.base_dim,
            'decoder_base_dim': config.decoder_base_dim,
            'z_dim': config.z_dim,
            'dim_mult': list(config.dim_mult),
            'num_res_blocks': config.num_res_blocks,
            'attn_scales': list(config.attn_scales),
            'temperal_downsample': list(config.temperal_downsample),
            'dropout': config.dropout,
            'latents_mean': list(config.latents_mean),
            'latents_std': list(config.latents_std),
            'is_residual': config.is_residual,
            'in_channels': config.in_channels,
            'out_channels': config.out_channels,
            'patch_size': config.patch_size,
            'scale_factor_temporal': config.scale_factor_temporal,
            'scale_factor_spatial': config.scale_factor_spatial,
        }
        self._config_hash = hash(tuple(sorted(self._config_dict.items())))
    
    def encode(self, x: Array) -> Tuple[Array, Array]:
        """Encode input to latent distribution."""
        return vae_encode(self.params, x, self.config)
    
    def decode(self, z: Array) -> Array:
        """Decode latent to output."""
        return vae_decode(self.params, z, self.config)
    
    def decode_jit(self, z: Array) -> Array:
        """JIT-compiled decode."""
        return decode_jit(self.params, z, self._config_hash, self._config_dict)
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        subfolder: Optional[str] = "vae",
        dtype: jnp.dtype = jnp.float32,
        **kwargs,
    ) -> "AutoencoderKLWan":
        """Load pretrained model."""
        params, config = load_vae_params(
            pretrained_model_name_or_path,
            subfolder=subfolder,
            dtype=dtype,
        )
        return cls(params, config)
