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
Flax/JAX implementation of Wan VAE with full feature parity to PyTorch version.

This implementation includes:
- CausalConv3d with feat_cache support
- Tiling for memory efficiency
- Frame batch processing
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union, List

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from flax import nnx

from ...configuration_utils import ConfigMixin


def _apply_sharding_constraint(inputs, is_nthwc=True):
    """
    Apply sharding constraint to distribute data across TPUs.
    
    This function mirrors the TorchAx mark_sharding behavior.
    
    Args:
        inputs: JAX array to apply sharding to
        is_nthwc: If True, use NTHWC format (Flax), otherwise NCTHW (TorchAx)
        
    Returns:
        inputs with sharding constraint applied (or unchanged if not in mesh context)
    """
    # PartitionSpec for Flax NTHWC format:
    # - Batch (N): None (not sharded)
    # - Time (T): None (not sharded)
    # - Height (H): None (not sharded)
    # - Width (W): ("dp", "tp") or ("tp") or ("dp") - sharded across devices
    # - Channels (C): None (not sharded)
    
    # TorchAx NCTHW would have W at index 4, but Flax NTHWC has W at index 3
    if is_nthwc:
        # Flax format: (B, T, H, W, C) - shard on W (index 3)
        specs = [
            P(None, None, None, ("dp", "tp"), None),  # Try dp+tp first
            P(None, None, None, ("tp",), None),       # Try tp only
            P(None, None, None, ("dp",), None),       # Try dp only
        ]
    else:
        # TorchAx format: (B, C, T, H, W) - shard on W (index 4)
        specs = [
            P(None, None, None, None, ("dp", "tp")),
            P(None, None, None, None, ("tp",)),
            P(None, None, None, None, ("dp",)),
        ]
    
    for spec in specs:
        try:
            return jax.lax.with_sharding_constraint(inputs, spec)
        except (ValueError, Exception):
            # This spec didn't work, try next one
            continue
    
    # No sharding worked (likely not in a mesh context), return unchanged
    return inputs


@dataclass
class FlaxAutoencoderKLWanConfig:
    """
    Configuration class for FlaxAutoencoderKLWan.
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
        """从字典创建配置"""
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(cls)}
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}
        # Handle list/tuple conversion
        if "dim_mult" in filtered_dict:
            filtered_dict["dim_mult"] = tuple(filtered_dict["dim_mult"])
        if "attn_scales" in filtered_dict:
            filtered_dict["attn_scales"] = tuple(filtered_dict["attn_scales"])
        if "temperal_downsample" in filtered_dict:
            filtered_dict["temperal_downsample"] = tuple(filtered_dict["temperal_downsample"])
        if "latents_mean" in filtered_dict:
            filtered_dict["latents_mean"] = tuple(filtered_dict["latents_mean"])
        if "latents_std" in filtered_dict:
            filtered_dict["latents_std"] = tuple(filtered_dict["latents_std"])
        return cls(**filtered_dict)


class FlaxConv3d(nnx.Module):
    """Basic 3D convolution wrapper for Flax NNX."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]] = (3, 3, 3),
        stride: Union[int, Tuple[int, int, int]] = (1, 1, 1),
        padding: Union[int, Tuple[int, int, int], str] = 0,
        rngs: nnx.Rngs = None,
    ):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3
            
        # Handle padding
        if isinstance(padding, int):
            if padding == 0:
                padding_mode = ((0, 0), (0, 0), (0, 0))
            else:
                padding_mode = ((padding, padding), (padding, padding), (padding, padding))
        elif isinstance(padding, tuple) and len(padding) == 3:
            padding_mode = tuple((p, p) for p in padding)
        else:
            padding_mode = padding
            
        self.conv = nnx.Conv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            strides=stride,
            padding=padding_mode,
            rngs=rngs,
        )

    def __call__(self, x):
        return self.conv(x)


class FlaxConv2d(nnx.Module):
    """Basic 2D convolution wrapper for Flax NNX."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = (3, 3),
        stride: Union[int, Tuple[int, int]] = (1, 1),
        padding: Union[int, str] = 0,
        rngs: nnx.Rngs = None,
    ):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
            
        # Handle padding
        if isinstance(padding, int):
            if padding == 0:
                padding_mode = ((0, 0), (0, 0))
            else:
                padding_mode = ((padding, padding), (padding, padding))
        else:
            padding_mode = padding
            
        self.conv = nnx.Conv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            strides=stride,
            padding=padding_mode,
            rngs=rngs,
        )

    def __call__(self, x):
        return self.conv(x)


class FlaxWanCausalConv3d(nnx.Module):
    """
    A 3D causal convolution layer with feat_cache support for Wan VAE.
    """
    
    CACHE_T = 2
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3
        if isinstance(padding, int):
            padding = (padding,) * 3
            
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding_val = padding
        
        # Calculate causal padding
        # padding is (pad_t, pad_h, pad_w) in PyTorch if passed as int/tuple
        # But Wan implementation: padding[2] is width pad, padding[1] is height pad, padding[0] is time pad
        # WanCausalConv3d:
        # self._padding = (self.padding[2], self.padding[2], self.padding[1], self.padding[1], 2 * self.padding[0], 0)
        # padding arg passed to Conv3d is (0, 0, 0)
        
        self.time_pad = 2 * padding[0]
        self.height_pad = padding[1]
        self.width_pad = padding[2]
        
        # JAX padding format: ((before_1, after_1), (before_2, after_2), ...)
        # NTHWC: (N, T, H, W, C)
        # Constant padding for spatial dimensions
        self.const_padding_spatial = (
            (self.height_pad, self.height_pad),
            (self.width_pad, self.width_pad)
        )
        
        # Use FlaxConv3d for the underlying convolution
        # Pass 0 padding as we will handle it manually
        self.conv = FlaxConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            rngs=rngs,
        )
    
    def __call__(
        self,
        inputs: jnp.ndarray,
        feat_cache: Optional[list] = None,
        feat_idx: int = 0,
    ):
        """
        Args:
            inputs: (B, T, H, W, C)
            feat_cache: list of cache tensors
            feat_idx: current cache index
        """
        # Determine padding for time dimension
        time_pad_val = self.time_pad
        
        # Handle cache
        if feat_cache is not None and time_pad_val > 0:
            # Get cache from list
            if feat_cache[feat_idx] is not None:
                cache_x = feat_cache[feat_idx]
                # In PyTorch: x = torch.cat([cache_x, x], dim=2)  (NCTHW)
                # In JAX: concatenate on Time dimension (axis 1)
                inputs = jnp.concatenate([cache_x, inputs], axis=1)
                
                # Update time padding
                # padding[4] -= cache_x.shape[2] (NCTHW, T is dim 2)
                time_pad_val -= cache_x.shape[1]
                if time_pad_val < 0:
                    time_pad_val = 0
            
            # Note: We don't update cache here, we just use it for padding reduction
            # Cache update happens OUTSIDE this call in the calling layer usually,
            # BUT WanCausalConv3d implementation doesn't seem to update cache inside itself?
            # Wait, WanCausalConv3d DOES NOT update cache. It just consumes it.
            # The cache update logic is in WanResidualBlock/WanResample/WanEncoder3d/WanDecoder3d.
            # Correct.
        
        # Apply padding
        # NTHWC format
        pad_width = [
            (0, 0),  # Batch
            (time_pad_val, 0),  # Time (causal: pad before)
            (self.height_pad, self.height_pad),  # Height
            (self.width_pad, self.width_pad),    # Width
            (0, 0),  # Channels
        ]
        
        inputs = jnp.pad(inputs, pad_width, mode='constant', constant_values=0)
        
        # Apply sharding constraint
        inputs = _apply_sharding_constraint(inputs, is_nthwc=True)
        
        return self.conv(inputs)


class FlaxWanRMS_norm(nnx.Module):
    """
    A custom RMS normalization layer.
    MATCHES: WanRMS_norm
    """
    def __init__(
        self,
        dim: int,
        channel_first: bool = True,
        images: bool = True,
        bias: bool = False,
        rngs: nnx.Rngs = None,
    ):
        self.dim = dim
        self.channel_first = channel_first  # Kept for compatibility, but JAX is always channel-last usually
        self.images = images
        self.bias_enabled = bias
        
        self.scale_const = dim**0.5
        
        # Parameter shapes
        # PyTorch:
        # broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        # shape = (dim, *broadcastable_dims) if channel_first else (dim,)
        # For NTHWC (channel last), we want params to broadcast correctly.
        # If channel_first=True in PyTorch (NCTHW), gamma is (C, 1, 1, 1)
        # In JAX NTHWC, gamma should be (C,) to broadcast over (N, T, H, W, C)
        
        self.gamma = nnx.Param(jnp.ones((dim,)))
        if bias:
            self.bias = nnx.Param(jnp.zeros((dim,)))
        else:
            self.bias = None

    def __call__(self, x):
        # x is (B, T, H, W, C)
        
        # F.normalize(x, dim=(1 if self.channel_first else -1))
        # PyTorch normalize computes x / max(norm(x), eps)
        # Wait, F.normalize uses L2 norm by default.
        # norm = x.norm(p=2, dim=dim, keepdim=True).clamp(min=eps).expand_as(x)
        # return x / norm
        
        # JAX implementation of F.normalize(dim=-1)
        # Calculate L2 norm along channel dimension
        norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
        
        # Avoid division by zero
        eps = 1e-12
        norm = jnp.maximum(norm, eps)
        
        x_normalized = x / norm
        
        # * self.scale * self.gamma
        out = x_normalized * self.scale_const * self.gamma.value
        
        if self.bias is not None:
            out = out + self.bias.value
            
        return out


class FlaxAvgDown3D(nnx.Module):
    """
    Matches AvgDown3D
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor_t: int,
        factor_s: int = 1,
        rngs: nnx.Rngs = None,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s
        
        assert in_channels * self.factor % out_channels == 0
        self.group_size = in_channels * self.factor // out_channels
        
    def __call__(self, x):
        # x: (B, T, H, W, C)  (JAX NTHWC)
        B, T, H, W, C = x.shape
        
        # Pad time dimension if needed
        # pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t (NCTHW)
        pad_t = (self.factor_t - T % self.factor_t) % self.factor_t
        
        if pad_t > 0:
            # PyTorch: F.pad(x, (0,0,0,0,pad_t,0)) -> pads last dim first. (left, right, top, bottom, front, back)
            # So pads time front? No, PyTorch pad is (last_dim_left, last_dim_right, 2nd_last_left...)
            # NCTHW: W, H, T, C, N
            # (0,0, 0,0, pad_t, 0) -> W(0,0), H(0,0), T(pad_t, 0) -> Pad T front?
            # Wait, PyTorch docs: "Padding ... starting from the last dimension and moving forward."
            # If 3D input (N,C,D,H,W), pad arg has 6 values.
            # (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
            # pad_front is index 4.
            # So it pads T dimension BEFORE.
            
            # JAX: (N, T, H, W, C)
            pad_width = [(0, 0), (pad_t, 0), (0, 0), (0, 0), (0, 0)]
            x = jnp.pad(x, pad_width, mode='constant', constant_values=0)
            # Update T
            T = x.shape[1]

        # Reshape logic conversion
        # PyTorch:
        # x = x.view(B, C, T//ft, ft, H//fs, fs, W//fs, fs)
        # x = x.permute(0, 1, 3, 5, 7, 2, 4, 6) -> (B, C, ft, fs, fs, T//ft, H//fs, W//fs)
        # x = x.view(B, C*factor, T//ft, H//fs, W//fs)
        # x = x.view(B, out_c, group_size, T//ft, H//fs, W//fs)
        # x = x.mean(dim=2) -> (B, out_c, T//ft, H//fs, W//fs)
        
        # JAX (NTHWC):
        # Goal: mimic the channel mixing and spatial/temporal downsampling
        
        # 1. Reshape to split dimensions
        x = x.reshape(B, T//self.factor_t, self.factor_t, H//self.factor_s, self.factor_s, W//self.factor_s, self.factor_s, C)
        
        # 2. Permute to bring factors together with C
        # Want: (B, T//ft, H//fs, W//fs, C, ft, fs, fs)
        x = x.transpose(0, 1, 3, 5, 7, 2, 4, 6)
        
        # 3. Flatten factors into C
        # (B, T//ft, H//fs, W//fs, C * ft * fs * fs)
        x = x.reshape(B, T//self.factor_t, H//self.factor_s, W//self.factor_s, C * self.factor)
        
        # 4. Reshape for grouping
        # (B, T//ft, H//fs, W//fs, out_channels, group_size)
        x = x.reshape(B, T//self.factor_t, H//self.factor_s, W//self.factor_s, self.out_channels, self.group_size)
        
        # 5. Mean over group_size
        x = jnp.mean(x, axis=-1)
        
        return x


class FlaxDupUp3D(nnx.Module):
    """
    Matches DupUp3D
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor_t: int,
        factor_s: int = 1,
        rngs: nnx.Rngs = None,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s
        
        assert out_channels * self.factor % in_channels == 0
        self.repeats = out_channels * self.factor // in_channels
        
    def __call__(self, x, first_chunk=False):
        # x: (B, T, H, W, C)
        B, T, H, W, C = x.shape
        
        # PyTorch:
        # x = x.repeat_interleave(self.repeats, dim=1) -> (B, C*repeats, T, H, W)
        # x = x.view(B, out_c, ft, fs, fs, T, H, W)
        # x = x.permute(0, 1, 5, 2, 6, 3, 7, 4) -> (B, out_c, T, ft, H, fs, W, fs)
        # x = x.view(B, out_c, T*ft, H*fs, W*fs)
        
        # JAX (NTHWC):
        # 1. Repeat channels
        # (B, T, H, W, C) -> (B, T, H, W, C, repeats) -> (B, T, H, W, C*repeats)
        x = jnp.repeat(x[:, :, :, :, :, None], self.repeats, axis=5).reshape(B, T, H, W, C * self.repeats)
        
        # 2. Reshape to separate factors
        # C*repeats = out_channels * factor
        # (B, T, H, W, out_c, ft, fs, fs)
        x = x.reshape(B, T, H, W, self.out_channels, self.factor_t, self.factor_s, self.factor_s)
        
        # 3. Permute to interleave
        # Want (B, T, ft, H, fs, W, fs, out_c)
        x = x.transpose(0, 1, 5, 2, 6, 3, 7, 4)
        
        # 4. Flatten to final shape
        # (B, T*ft, H*fs, W*fs, out_c)
        x = x.reshape(B, T * self.factor_t, H * self.factor_s, W * self.factor_s, self.out_channels)
        
        if first_chunk:
            # x = x[:, :, self.factor_t - 1 :, :, :] (NCTHW)
            # JAX NTHWC: index 1 is Time
            x = x[:, self.factor_t - 1:, :, :, :]
            
        return x


class FlaxWanResample(nnx.Module):
    """
    Matches WanResample
    """
    def __init__(
        self,
        dim: int,
        mode: str,
        upsample_out_dim: int = None,
        rngs: nnx.Rngs = None,
    ):
        self.dim = dim
        self.mode = mode
        
        # default to dim // 2
        if upsample_out_dim is None:
            upsample_out_dim = dim // 2
        
        self.resample_identity = False
        
        if mode == "upsample2d":
            self.resample_conv = FlaxConv2d(dim, upsample_out_dim, 3, padding=1, rngs=rngs)
            self.time_conv = None
            
        elif mode == "upsample3d":
            self.resample_conv = FlaxConv2d(dim, upsample_out_dim, 3, padding=1, rngs=rngs)
            self.time_conv = FlaxWanCausalConv3d(
                dim, dim * 2, kernel_size=(3, 1, 1), padding=(1, 0, 0), rngs=rngs
            )
            
        elif mode == "downsample2d":
            # Padding: ZeroPad2d((0, 1, 0, 1)) -> (left, right, top, bottom)
            # Conv2d stride=(2, 2)
            # We can use padding in Conv2d or manual padding.
            # JAX padding: ((top, bottom), (left, right)) for 2D spatial
            # ZeroPad2d((0, 1, 0, 1)) means pad right=1, bottom=1.
            self.resample_conv = FlaxConv2d(
                dim, dim, 3, stride=(2, 2), padding=((0, 1), (0, 1)), rngs=rngs
            )
            self.time_conv = None
            
        elif mode == "downsample3d":
            self.resample_conv = FlaxConv2d(
                dim, dim, 3, stride=(2, 2), padding=((0, 1), (0, 1)), rngs=rngs
            )
            self.time_conv = FlaxWanCausalConv3d(
                dim, dim, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0), rngs=rngs
            )
        else:
            self.resample_identity = True
            self.resample_conv = None
            self.time_conv = None

    def __call__(self, x, feat_cache=None, feat_idx=0, deterministic: bool = True):
        # x: (B, T, H, W, C)
        B, T, H, W, C = x.shape
        
        if self.resample_identity:
            return x, feat_idx, feat_cache
            
        # Handle time_conv for 3D modes
        if self.mode == "upsample3d":
            if feat_cache is not None:
                CACHE_T = 2
                cache_x = x[:, -CACHE_T:, :, :, :]
                
                prev_cache = feat_cache[feat_idx]
                
                # Check if we need to pad with previous chunk's last frame
                if cache_x.shape[1] < 2 and prev_cache is not None:
                    last_frame = prev_cache[:, -1:, :, :, :]
                    cache_x = jnp.concatenate([last_frame, cache_x], axis=1)
                
                if cache_x.shape[1] < 2 and prev_cache is None:
                    zeros = jnp.zeros_like(cache_x)
                    cache_x = jnp.concatenate([zeros, cache_x], axis=1)
                    
                if prev_cache is None:
                    # First time: no cache to use
                    x = self.time_conv(x, feat_cache=None)
                else:
                    temp_cache = [prev_cache]
                    x = self.time_conv(x, feat_cache=temp_cache, feat_idx=0)
                
                feat_cache[feat_idx] = cache_x
                feat_idx += 1
                
                # Reshape: (B, T, H, W, 2*C) -> (B, T*2, H, W, C)
                x = x.reshape(B, T, H, W, 2, self.dim)
                x = x.transpose(0, 1, 4, 2, 3, 5).reshape(B, T * 2, H, W, self.dim)
        
        # Prepare for 2D spatial resampling
        # x: (B, T, H, W, C)
        T_curr = x.shape[1]
        x = x.reshape(B * T_curr, H, W, x.shape[-1])
        
        # Apply spatial resampling
        if self.mode in ["upsample2d", "upsample3d"]:
            # Nearest exact upsampling
            x = jax.image.resize(x, (B * T_curr, H * 2, W * 2, x.shape[-1]), method='nearest')
            x = self.resample_conv(x)
        elif self.mode in ["downsample2d", "downsample3d"]:
            x = self.resample_conv(x)
        
        # Reshape back to (B, T, H, W, C)
        H_new, W_new = x.shape[1], x.shape[2]
        x = x.reshape(B, T_curr, H_new, W_new, x.shape[-1])
        
        # Downsample3d specific logic
        if self.mode == "downsample3d":
            if feat_cache is not None:
                prev_cache = feat_cache[feat_idx]
                cache_x = x[:, -1:, :, :, :]
                
                if prev_cache is None:
                    feat_cache[feat_idx] = x
                    feat_idx += 1
                else:
                    last_frame_prev = prev_cache[:, -1:, :, :, :]
                    x_in = jnp.concatenate([last_frame_prev, x], axis=1)
                    
                    x = self.time_conv(x_in, feat_cache=None)
                    
                    feat_cache[feat_idx] = cache_x
                    feat_idx += 1
                    
        return x, feat_idx, feat_cache


class FlaxWanAttentionBlock(nnx.Module):
    """
    Causal self-attention with a single head.
    Matches WanAttentionBlock.
    """
    def __init__(
        self,
        dim: int,
        rngs: nnx.Rngs = None,
    ):
        self.dim = dim
        
        self.norm = FlaxWanRMS_norm(dim, rngs=rngs)
        self.to_qkv = FlaxConv2d(dim, dim * 3, 1, rngs=rngs)
        self.proj = FlaxConv2d(dim, dim, 1, rngs=rngs)
        
    def __call__(self, x):
        # x: (B, T, H, W, C)
        B, T, H, W, C = x.shape
        identity = x
        
        # Reshape to (B*T, H, W, C) for 2D ops
        x = x.reshape(B * T, H, W, C)
        x = self.norm(x)
        
        # Compute qkv
        qkv = self.to_qkv(x)  # (B*T, H, W, 3*C)
        
        # Reshape for attention
        # (B*T, H, W, 3*C) -> (B*T, H*W, 3*C)
        qkv = qkv.reshape(B * T, H * W, 3 * C)
        
        # Split
        q, k, v = jnp.split(qkv, 3, axis=-1)  # (B*T, H*W, C)
        
        # Add head dimension (1 head)
        # (B*T, 1, H*W, C)
        q = q[:, None, :, :]
        k = k[:, None, :, :]
        v = v[:, None, :, :]
        
        # Scaled Dot Product Attention
        scale = 1.0 / jnp.sqrt(C)
        attn_weights = jnp.matmul(q, k.swapaxes(-1, -2)) * scale
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        
        out = jnp.matmul(attn_weights, v)  # (B*T, 1, H*W, C)
        
        # Remove head dim
        out = out.squeeze(1)  # (B*T, H*W, C)
        
        # Reshape back
        out = out.reshape(B * T, H, W, C)
        
        # Output projection
        out = self.proj(out)
        
        # Reshape to 5D
        out = out.reshape(B, T, H, W, C)
        
        return out + identity


class FlaxWanResidualBlock(nnx.Module):
    """
    Matches WanResidualBlock.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.0,
        non_linearity: str = "silu",
        rngs: nnx.Rngs = None,
    ):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout_rate = dropout
        
        # Norms
        self.norm1 = FlaxWanRMS_norm(in_dim, images=False, rngs=rngs)
        self.norm2 = FlaxWanRMS_norm(out_dim, images=False, rngs=rngs)
        
        # Convs
        self.conv1 = FlaxWanCausalConv3d(in_dim, out_dim, 3, padding=1, rngs=rngs)
        self.conv2 = FlaxWanCausalConv3d(out_dim, out_dim, 3, padding=1, rngs=rngs)
        
        # Shortcut
        if in_dim != out_dim:
            self.conv_shortcut = FlaxWanCausalConv3d(in_dim, out_dim, 1, rngs=rngs)
        else:
            self.conv_shortcut = None
            
    def __call__(
        self,
        x,
        feat_cache=None,
        feat_idx=0,
        deterministic: bool = True,
    ):
        # Shortcut
        if self.conv_shortcut is not None:
            # Shortcut conv usually 1x1x1, so no cache needed or simple pass-through?
            # WanCausalConv3d with kernel 1 has padding 0.
            # PyTorch: self.conv_shortcut(x) (no cache passed)
            h = self.conv_shortcut(x, feat_cache=None)
        else:
            h = x
            
        # First block
        x = self.norm1(x)
        x = jax.nn.silu(x)
        
        # Conv1 with cache
        if feat_cache is not None:
            CACHE_T = 2
            cache_x = x[:, -CACHE_T:, :, :, :]
            
            prev_cache = feat_cache[feat_idx]
            
            temp_cache = [prev_cache]
            x = self.conv1(x, feat_cache=temp_cache, feat_idx=0)
            
            if cache_x.shape[1] < 2 and prev_cache is not None:
                last_frame = prev_cache[:, -1:, :, :, :]
                cache_x = jnp.concatenate([last_frame, cache_x], axis=1)
                
            feat_cache[feat_idx] = cache_x
            feat_idx += 1
        else:
            x = self.conv1(x)
            
        # Second block
        x = self.norm2(x)
        x = jax.nn.silu(x)
        
        if self.dropout_rate > 0 and not deterministic:
            x = nnx.Dropout(self.dropout_rate)(x)
            
        # Conv2 with cache
        if feat_cache is not None:
            CACHE_T = 2
            cache_x = x[:, -CACHE_T:, :, :, :]
            
            prev_cache = feat_cache[feat_idx]
            
            temp_cache = [prev_cache]
            x = self.conv2(x, feat_cache=temp_cache, feat_idx=0)
            
            if cache_x.shape[1] < 2 and prev_cache is not None:
                last_frame = prev_cache[:, -1:, :, :, :]
                cache_x = jnp.concatenate([last_frame, cache_x], axis=1)
                
            feat_cache[feat_idx] = cache_x
            feat_idx += 1
        else:
            x = self.conv2(x)
        
        # CRITICAL: Add residual connection (skip connection)
        # This was missing and caused severe quality degradation
        return x + h, feat_idx, feat_cache


def patchify(x, patch_size):
    if patch_size == 1:
        return x
        
    if x.ndim != 5:
        raise ValueError(f"Invalid input shape: {x.shape}")
    
    # x shape: [B, T, H, W, C]
    B, T, H, W, C = x.shape
    
    if H % patch_size != 0 or W % patch_size != 0:
        raise ValueError(f"Height ({H}) and width ({W}) must be divisible by patch_size ({patch_size})")
    
    # Reshape
    # (B, T, H//p, p, W//p, p, C)
    x = x.reshape(B, T, H // patch_size, patch_size, W // patch_size, patch_size, C)
    
    # Rearrange
    # Want (B, T, H//p, W//p, C*p*p)
    # Transpose: (B, T, H//p, W//p, p, p, C) -> (0, 1, 2, 4, 6, 3, 5) ?
    # PyTorch: x.permute(0, 1, 6, 4, 2, 3, 5) from (B, C, T, H//p, p, W//p, p)
    
    # Let's derive from scratch for NTHWC
    # (B, T, H, W, C) -> (B, T, H//p, p, W//p, p, C)
    # We want to merge p, p, C into one dimension at the end.
    # Transpose to (B, T, H//p, W//p, p, p, C)
    x = x.transpose(0, 1, 2, 4, 3, 5, 6)
    
    # Flatten last 3
    x = x.reshape(B, T, H // patch_size, W // patch_size, C * patch_size * patch_size)
    
    return x


def unpatchify(x, patch_size):
    if patch_size == 1:
        return x
        
    if x.ndim != 5:
        raise ValueError(f"Invalid input shape: {x.shape}")
        
    # x shape: (B, T, H, W, C_patches)
    B, T, H, W, C_patches = x.shape
    C = C_patches // (patch_size * patch_size)
    
    # Reshape
    # (B, T, H, W, p, p, C)
    x = x.reshape(B, T, H, W, patch_size, patch_size, C)
    
    # Transpose back
    # (B, T, H, p, W, p, C)
    # From (0, 1, 2, 3, 4, 5, 6) to (0, 1, 2, 4, 3, 5, 6)
    x = x.transpose(0, 1, 2, 4, 3, 5, 6)
    
    # Flatten
    # (B, T, H*p, W*p, C)
    x = x.reshape(B, T, H * patch_size, W * patch_size, C)
    
    return x


class FlaxAutoencoderKLWan(nnx.Module):
    """
    A VAE model with KL loss for encoding videos into latents and decoding latent representations into videos.
    """
    config_class = FlaxAutoencoderKLWanConfig
    
    def __init__(
        self,
        config: FlaxAutoencoderKLWanConfig,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
        mesh = None,
    ):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.mesh = mesh
        
        self.z_dim = config.z_dim
        
        decoder_base_dim = config.decoder_base_dim
        if decoder_base_dim is None:
            decoder_base_dim = config.base_dim
            
        self.encoder = FlaxWanEncoder3d(
            in_channels=config.in_channels,
            dim=config.base_dim,
            z_dim=config.z_dim * 2,
            dim_mult=config.dim_mult,
            num_res_blocks=config.num_res_blocks,
            attn_scales=config.attn_scales,
            temperal_downsample=config.temperal_downsample,
            dropout=config.dropout,
            is_residual=config.is_residual,
            rngs=rngs,
        )
        
        self.quant_conv = FlaxWanCausalConv3d(config.z_dim * 2, config.z_dim * 2, 1, rngs=rngs)
        self.post_quant_conv = FlaxWanCausalConv3d(config.z_dim, config.z_dim, 1, rngs=rngs)
        
        # Reverse temporal upsample
        temperal_upsample = tuple(reversed(config.temperal_downsample))
        
        self.decoder = FlaxWanDecoder3d(
            dim=decoder_base_dim,
            z_dim=config.z_dim,
            dim_mult=config.dim_mult,
            num_res_blocks=config.num_res_blocks,
            attn_scales=config.attn_scales,
            temperal_upsample=temperal_upsample,
            dropout=config.dropout,
            out_channels=config.out_channels,
            is_residual=config.is_residual,
            rngs=rngs,
        )
        
        self.spatial_compression_ratio = config.scale_factor_spatial
        
        # Tiling params
        self.use_tiling = False
        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256
        self.tile_sample_stride_height = 192
        self.tile_sample_stride_width = 192
        
        self.use_slicing = False
        
        # Precompute conv counts
        self._cached_conv_counts = {
            "decoder": self._count_causal_conv3d(self.decoder) if self.decoder is not None else 0,
            "encoder": self._count_causal_conv3d(self.encoder) if self.encoder is not None else 0,
        }
        
        # Mean/Std for Latents are accessed from config when needed
        # (not stored as model attributes to avoid weight loading issues)

    def _count_causal_conv3d(self, module):
        count = 0
        node_types = nnx.graph.iter_graph([module])
        for _, value in node_types:
            if isinstance(value, FlaxWanCausalConv3d):
                count += 1
        return count
        
    def _init_feat_cache(self, mode="decoder"):
        count = self._cached_conv_counts.get(mode, 0)
        return [None] * count
        
    def _encode(self, x: jnp.ndarray, deterministic: bool = True):
        # x: (B, T, H, W, C)
        B, T, H, W, C = x.shape
        
        # Patchify
        if self.config.patch_size is not None:
            x = patchify(x, patch_size=self.config.patch_size)
            
        # Tiling check (omitted for brevity, can be added later if needed)
        
        # Process in chunks of 4 frames
        # iter_ = 1 + (num_frame - 1) // 4
        
        enc_feat_map = self._init_feat_cache("encoder")
        
        chunk_size = 4
        num_chunks = 1 + (T - 1) // chunk_size
        
        outs = []
        for i in range(num_chunks):
            if i == 0:
                # First frame only for first chunk?
                # PyTorch: x[:, :, :1, :, :]
                # JAX: x[:, :1, :, :, :]
                chunk_in = x[:, :1, :, :, :]
                out, _, enc_feat_map = self.encoder(chunk_in, feat_cache=enc_feat_map, deterministic=deterministic)
            else:
                # x[:, :, 1 + 4*(i-1) : 1 + 4*i, :, :]
                start = 1 + chunk_size * (i - 1)
                end = 1 + chunk_size * i
                chunk_in = x[:, start:end, :, :, :]
                
                out_chunk, _, enc_feat_map = self.encoder(chunk_in, feat_cache=enc_feat_map, deterministic=deterministic)
                # In PyTorch, out accumulates by concatenation
                # out = torch.cat([out, out_], 2)
                # But here we are in loop.
                # We can't update 'out' in place in JAX loop effectively without scan.
                # But since this is Python loop (unrolled), we can just append to list.
                # But the FIRST chunk produces 'out'.
                outs.append(out_chunk)
                
        # Concatenate: first out (from i=0) + rest
        # i=0 produces 'out'. i>0 produces 'out_chunk'.
        # Wait, the loop structure in PyTorch:
        # if i==0: out = encoder(...)
        # else: out_ = encoder(...); out = cat([out, out_])
        # So 'out' grows.
        
        # In JAX Python loop:
        if num_chunks > 0:
            final_out = outs[0] if outs else None # But wait, i=0 logic is inside loop
            
            # Let's rewrite cleaner
            out_accum = None
            for i in range(num_chunks):
                if i == 0:
                    chunk_in = x[:, :1, :, :, :]
                    out_accum, _, enc_feat_map = self.encoder(chunk_in, feat_cache=enc_feat_map, deterministic=deterministic)
                else:
                    start = 1 + chunk_size * (i - 1)
                    end = 1 + chunk_size * i
                    chunk_in = x[:, start:end, :, :, :]
                    out_chunk, _, enc_feat_map = self.encoder(chunk_in, feat_cache=enc_feat_map, deterministic=deterministic)
                    out_accum = jnp.concatenate([out_accum, out_chunk], axis=1)
            
            enc = self.quant_conv(out_accum)
            return enc
        else:
            return None # Should not happen

    def encode(self, x: jnp.ndarray, return_dict: bool = True, deterministic: bool = True):
        if self.use_slicing and x.shape[0] > 1:
            # Split batch
            # encoded_slices = [self._encode(x[i:i+1]) for i in range(x.shape[0])]
            # h = jnp.concatenate(encoded_slices, axis=0)
            pass # TODO: Slicing
            
        h = self._encode(x, deterministic=deterministic)
        
        # DiagonalGaussianDistribution
        mean, logvar = jnp.split(h, 2, axis=-1)
        # logvar = jnp.clip(logvar, -30.0, 20.0)
        # std = jnp.exp(0.5 * logvar)
        # var = jnp.exp(logvar)
        
        # We return mean and logvar typically
        return mean, logvar

    def _decode(self, z: jnp.ndarray, deterministic: bool = True):
        B, T, H, W, C = z.shape
        
        feat_map = self._init_feat_cache("decoder")
        
        # Post quant conv
        x = self.post_quant_conv(z)
        
        # Loop over frames
        outs = []
        for i in range(T):
            chunk_in = x[:, i:i+1, :, :, :]
            
            first_chunk = (i == 0)
            out_chunk, _, feat_map = self.decoder(
                chunk_in,
                feat_cache=feat_map,
                first_chunk=first_chunk,
                deterministic=deterministic
            )
            outs.append(out_chunk)
            
        out = jnp.concatenate(outs, axis=1)
        
        if self.config.patch_size is not None:
            out = unpatchify(out, patch_size=self.config.patch_size)
            
        out = jnp.clip(out, -1.0, 1.0)
        
        return out

    def decode(self, z: jnp.ndarray, deterministic: bool = True):
        decoded = self._decode(z, deterministic=deterministic)
        return decoded

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        subfolder: Optional[str] = "vae",
        dtype: jnp.dtype = jnp.float32,
        mesh = None,
        **kwargs,
    ):
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open
        import json
        import re
        from flax.traverse_util import unflatten_dict
        
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
                
        # 3. Convert weights
        jax_weights = {}
        
        # Helper to rename keys
        def rename_key(key):
            key = key.replace("up_blocks.", "up_blocks_") # Temp to avoid split
            key = key.replace("mid_block.", "mid_block_")
            key = key.replace("down_blocks.", "down_blocks_")
            
            # Map module names
            # FlaxWanCausalConv3d has nested structure: .conv (FlaxConv3d) -> .conv (nnx.Conv)
            # So conv_in.weight -> conv_in.conv.conv.kernel
            key = key.replace("conv_in.bias", "conv_in.conv.conv.bias")
            key = key.replace("conv_in.weight", "conv_in.conv.conv.kernel")
            key = key.replace("conv_out.bias", "conv_out.conv.conv.bias")
            key = key.replace("conv_out.weight", "conv_out.conv.conv.kernel")
            
            # Inside blocks
            key = key.replace("attentions.", "attentions_")
            key = key.replace("resnets.", "resnets_")
            key = key.replace("upsamplers.", "upsamplers_")
            # key = key.replace("resample.", "resample_") # careful
            
            # Conv layers in ResidualBlock (FlaxWanCausalConv3d)
            # conv1.weight -> conv1.conv.conv.kernel (FlaxWanCausalConv3d -> FlaxConv3d -> nnx.Conv)
            key = re.sub(r"conv(\d+)\.weight", r"conv\1.conv.conv.kernel", key)
            key = re.sub(r"conv(\d+)\.bias", r"conv\1.conv.conv.bias", key)
            
            # time_conv (FlaxWanCausalConv3d)
            key = key.replace("time_conv.weight", "time_conv.conv.conv.kernel")
            key = key.replace("time_conv.bias", "time_conv.conv.conv.bias")
            
            # quant_conv / post_quant_conv (FlaxWanCausalConv3d)
            key = key.replace("quant_conv.weight", "quant_conv.conv.conv.kernel")
            key = key.replace("quant_conv.bias", "quant_conv.conv.conv.bias")
            
            # conv_shortcut (FlaxWanCausalConv3d)
            key = key.replace("conv_shortcut.weight", "conv_shortcut.conv.conv.kernel")
            key = key.replace("conv_shortcut.bias", "conv_shortcut.conv.conv.bias")
            
            # Resample layers
            # In PyTorch: resample.1 is the conv (Sequential)
            # In Flax: FlaxWanResample uses FlaxConv2d (not FlaxWanCausalConv3d)
            # FlaxConv2d has: .conv (nnx.Conv), so only one level
            # Encoder downsample
            if "down_blocks" in key and "resample" in key:
                # down_blocks.X.resample.1.weight -> down_blocks.X.resample_conv.conv.kernel
                key = key.replace("resample.1.weight", "resample_conv.conv.kernel")
                key = key.replace("resample.1.bias", "resample_conv.conv.bias")
            
            # Decoder upsample
            if "up_blocks" in key and "upsamplers" in key:
                # upsamplers.0.resample.1.weight -> upsamplers.0.resample_conv.conv.kernel
                key = key.replace("resample.1.weight", "resample_conv.conv.kernel")
                key = key.replace("resample.1.bias", "resample_conv.conv.bias")
                
            # Attention (FlaxConv2d - only one .conv level)
            key = key.replace("to_qkv.weight", "to_qkv.conv.kernel")
            key = key.replace("to_qkv.bias", "to_qkv.conv.bias")
            key = key.replace("proj.weight", "proj.conv.kernel")
            key = key.replace("proj.bias", "proj.conv.bias")
            
            # Norm
            key = key.replace("norm.weight", "norm.gamma")
            key = key.replace("norm.bias", "norm.bias")
            key = key.replace("norm1.weight", "norm1.gamma")
            key = key.replace("norm1.bias", "norm1.bias")
            key = key.replace("norm2.weight", "norm2.gamma")
            key = key.replace("norm2.bias", "norm2.bias")
            key = key.replace("norm_out.weight", "norm_out.gamma")
            key = key.replace("norm_out.bias", "norm_out.bias")
            
            # Revert structural replacements
            key = key.replace("up_blocks_", "up_blocks.")
            key = key.replace("mid_block_", "mid_block.")
            key = key.replace("down_blocks_", "down_blocks.")
            key = key.replace("attentions_", "attentions.")
            key = key.replace("resnets_", "resnets.")
            key = key.replace("upsamplers_", "upsamplers.")
            
            return key

        for pt_key, pt_tensor in pytorch_weights.items():
            # Rename key
            flax_key = rename_key(pt_key)
            
            # Convert tensor shape (Conv weights: O, I, ...)
            # Flax Conv: (K..., I, O)
            # Dense/Linear: (I, O)
            
            if "conv" in flax_key and "kernel" in flax_key:
                # PyTorch Conv3d: (Out, In, T, H, W)
                # JAX Conv3d: (T, H, W, In, Out)
                if pt_tensor.ndim == 5:
                    pt_tensor = pt_tensor.transpose(2, 3, 4, 1, 0)
                # PyTorch Conv2d: (Out, In, H, W)
                # JAX Conv2d: (H, W, In, Out)
                elif pt_tensor.ndim == 4:
                    pt_tensor = pt_tensor.transpose(2, 3, 1, 0)
            
            # Norm gamma: PyTorch (C, 1, 1, 1) or (C, 1, 1) -> Flax (C,)
            if "gamma" in flax_key:
                pt_tensor = pt_tensor.squeeze()
            
            # Cast to dtype
            jax_weights[flax_key] = jnp.array(pt_tensor, dtype=dtype)
            
        # 4. Initialize
        key = jax.random.key(0)
        rngs = nnx.Rngs(key)
        model = cls(config=config, rngs=rngs, dtype=dtype, mesh=mesh)
        
        # Load weights
        nested_weights = unflatten_dict(jax_weights, sep=".")
        graphdef, _ = nnx.split(model)
        model = nnx.merge(graphdef, nested_weights)
        
        return model



class FlaxWanMidBlock(nnx.Module):
    """
    Middle block for WanVAE encoder and decoder.
    Matches WanMidBlock.
    """
    def __init__(
        self,
        dim: int,
        dropout: float = 0.0,
        non_linearity: str = "silu",
        num_layers: int = 1,
        rngs: nnx.Rngs = None,
    ):
        self.dim = dim
        
        resnets = [FlaxWanResidualBlock(dim, dim, dropout, non_linearity, rngs=rngs)]
        attentions = []
        
        for _ in range(num_layers):
            attentions.append(FlaxWanAttentionBlock(dim, rngs=rngs))
            resnets.append(FlaxWanResidualBlock(dim, dim, dropout, non_linearity, rngs=rngs))
            
        self.attentions = nnx.List(attentions)
        self.resnets = nnx.List(resnets)
        
    def __call__(self, x, feat_cache=None, feat_idx=0, deterministic: bool = True):
        # First residual block
        x, feat_idx, feat_cache = self.resnets[0](x, feat_cache, feat_idx, deterministic)
        
        # Process through attention and residual blocks
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                x = attn(x)
                
            x, feat_idx, feat_cache = resnet(x, feat_cache, feat_idx, deterministic)
            
        return x, feat_idx, feat_cache


class FlaxWanResidualDownBlock(nnx.Module):
    """
    Matches WanResidualDownBlock
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float,
        num_res_blocks: int,
        temperal_downsample: bool = False,
        down_flag: bool = False,
        rngs: nnx.Rngs = None,
    ):
        self.avg_shortcut = FlaxAvgDown3D(
            in_dim,
            out_dim,
            factor_t=2 if temperal_downsample else 1,
            factor_s=2 if down_flag else 1,
            rngs=rngs
        )
        
        resnets = []
        curr_dim = in_dim
        for _ in range(num_res_blocks):
            resnets.append(FlaxWanResidualBlock(curr_dim, out_dim, dropout, rngs=rngs))
            curr_dim = out_dim
        self.resnets = nnx.List(resnets)
        
        if down_flag:
            mode = "downsample3d" if temperal_downsample else "downsample2d"
            self.downsampler = FlaxWanResample(out_dim, mode=mode, rngs=rngs)
        else:
            self.downsampler = None
            
    def __call__(self, x, feat_cache=None, feat_idx=0, deterministic: bool = True):
        # x_copy = x.clone() (in PyTorch)
        shortcut = self.avg_shortcut(x)
        
        for resnet in self.resnets:
            x, feat_idx, feat_cache = resnet(x, feat_cache, feat_idx, deterministic)
            
        if self.downsampler is not None:
            x, feat_idx, feat_cache = self.downsampler(x, feat_cache, feat_idx)
            
        return x + shortcut, feat_idx, feat_cache


class FlaxWanResidualUpBlock(nnx.Module):
    """
    Matches WanResidualUpBlock
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        dropout: float = 0.0,
        temperal_upsample: bool = False,
        up_flag: bool = False,
        non_linearity: str = "silu",
        rngs: nnx.Rngs = None,
    ):
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        if up_flag:
            self.avg_shortcut = FlaxDupUp3D(
                in_dim,
                out_dim,
                factor_t=2 if temperal_upsample else 1,
                factor_s=2,
                rngs=rngs
            )
        else:
            self.avg_shortcut = None
            
        resnets = []
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(FlaxWanResidualBlock(current_dim, out_dim, dropout, non_linearity, rngs=rngs))
            current_dim = out_dim
        self.resnets = nnx.List(resnets)
        
        if up_flag:
            upsample_mode = "upsample3d" if temperal_upsample else "upsample2d"
            # upsample_out_dim=out_dim
            self.upsampler = FlaxWanResample(out_dim, mode=upsample_mode, upsample_out_dim=out_dim, rngs=rngs)
        else:
            self.upsampler = None
            
    def __call__(self, x, feat_cache=None, feat_idx=0, first_chunk=False, deterministic: bool = True):
        # x_copy = x.clone()
        x_orig = x
        
        for resnet in self.resnets:
            x, feat_idx, feat_cache = resnet(x, feat_cache, feat_idx, deterministic)
            
        if self.upsampler is not None:
            x, feat_idx, feat_cache = self.upsampler(x, feat_cache, feat_idx)
            
        if self.avg_shortcut is not None:
            shortcut = self.avg_shortcut(x_orig, first_chunk=first_chunk)
            x = x + shortcut
            
        return x, feat_idx, feat_cache


class FlaxWanUpBlock(nnx.Module):
    """
    Matches WanUpBlock
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        dropout: float = 0.0,
        upsample_mode: Optional[str] = None,
        non_linearity: str = "silu",
        rngs: nnx.Rngs = None,
    ):
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        resnets = []
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(FlaxWanResidualBlock(current_dim, out_dim, dropout, non_linearity, rngs=rngs))
            current_dim = out_dim
        self.resnets = nnx.List(resnets)
        
        if upsample_mode is not None:
            # ModuleList containing one Resample
            self.upsamplers = nnx.List([FlaxWanResample(out_dim, mode=upsample_mode, rngs=rngs)])
        else:
            self.upsamplers = None
            
    def __call__(self, x, feat_cache=None, feat_idx=0, first_chunk=None, deterministic: bool = True):
        for resnet in self.resnets:
            x, feat_idx, feat_cache = resnet(x, feat_cache, feat_idx, deterministic)
            
        if self.upsamplers is not None:
            x, feat_idx, feat_cache = self.upsamplers[0](x, feat_cache, feat_idx)
            
        return x, feat_idx, feat_cache


class FlaxWanEncoder3d(nnx.Module):
    """
    Matches WanEncoder3d
    """
    def __init__(
        self,
        in_channels: int = 3,
        dim: int = 128,
        z_dim: int = 4,
        dim_mult: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attn_scales: Tuple[float, ...] = (),
        temperal_downsample: Tuple[bool, ...] = (True, True, False),
        dropout: float = 0.0,
        non_linearity: str = "silu",
        is_residual: bool = False,
        rngs: nnx.Rngs = None,
    ):
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        
        # dims
        dims = [dim * u for u in [1] + list(dim_mult)]
        scale = 1.0
        
        # init block
        self.conv_in = FlaxWanCausalConv3d(in_channels, dims[0], 3, padding=1, rngs=rngs)
        
        # downsample blocks
        down_blocks = []
        
        # zip(dims[:-1], dims[1:])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if is_residual:
                is_last = (i == len(dim_mult) - 1)
                t_down = temperal_downsample[i] if not is_last else False
                
                block = FlaxWanResidualDownBlock(
                    in_dim,
                    out_dim,
                    dropout,
                    num_res_blocks,
                    temperal_downsample=t_down,
                    down_flag=not is_last,
                    rngs=rngs
                )
                down_blocks.append(block)
            else:
                # Standard blocks: resnets + optional attn + optional downsample
                # We wrap this in a custom Module or just use nnx.List?
                # PyTorch uses ModuleList flatly. We can do the same but grouped by stage might be cleaner.
                # However, to match PyTorch structure strictly for weight loading, we should use a flat list if possible
                # BUT, here we will group them into a "Block" to make forward pass cleaner.
                # Wait, PyTorch implementation puts them all in `self.down_blocks` list directly.
                # So `down_blocks` is a mix of ResidualBlock, AttentionBlock, Resample.
                
                # To simplify forward pass and weight loading mapping, let's create a "StageBlock" wrapper
                # that holds the list of modules for this stage.
                # OR, we can just use a list of layers like PyTorch.
                
                stage_layers = []
                curr_dim_inner = in_dim
                
                for _ in range(num_res_blocks):
                    stage_layers.append(FlaxWanResidualBlock(curr_dim_inner, out_dim, dropout, rngs=rngs))
                    if scale in attn_scales:
                        stage_layers.append(FlaxWanAttentionBlock(out_dim, rngs=rngs))
                    curr_dim_inner = out_dim
                    
                if i != len(dim_mult) - 1:
                    mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
                    stage_layers.append(FlaxWanResample(out_dim, mode=mode, rngs=rngs))
                    scale /= 2.0
                    
                # We append a nnx.List of layers as one item in down_blocks? 
                # Or flat list? 
                # PyTorch WanEncoder3d has `self.down_blocks = nn.ModuleList([])` and appends everything there.
                # So it's one giant flat list.
                # Let's do the same.
                down_blocks.extend(stage_layers)
                
        self.down_blocks = nnx.List(down_blocks)
        
        # middle block
        self.mid_block = FlaxWanMidBlock(dims[-1], dropout, non_linearity, num_layers=1, rngs=rngs)
        
        # output blocks
        self.norm_out = FlaxWanRMS_norm(dims[-1], images=False, rngs=rngs)
        self.conv_out = FlaxWanCausalConv3d(dims[-1], z_dim, 3, padding=1, rngs=rngs)
        
    def __call__(self, x, feat_cache=None, feat_idx=0, deterministic: bool = True):
        # Conv in
        # Manual cache logic for conv_in (same as ResidualBlock convs)
        if feat_cache is not None:
            CACHE_T = 2
            cache_x = x[:, -CACHE_T:, :, :, :]
            prev_cache = feat_cache[feat_idx]
            
            temp_cache = [prev_cache]
            x = self.conv_in(x, feat_cache=temp_cache, feat_idx=0)
            
            if cache_x.shape[1] < 2 and prev_cache is not None:
                last_frame = prev_cache[:, -1:, :, :, :]
                cache_x = jnp.concatenate([last_frame, cache_x], axis=1)
                
            feat_cache[feat_idx] = cache_x
            feat_idx += 1
        else:
            x = self.conv_in(x)
            
        # Down blocks
        for layer in self.down_blocks:
            if isinstance(layer, (FlaxWanResidualBlock, FlaxWanResidualDownBlock, FlaxWanResample, FlaxWanMidBlock)):
                x, feat_idx, feat_cache = layer(x, feat_cache, feat_idx, deterministic)
            elif isinstance(layer, FlaxWanAttentionBlock):
                x = layer(x)
            else:
                # Should not happen based on init
                pass
                
        # Mid block
        x, feat_idx, feat_cache = self.mid_block(x, feat_cache, feat_idx, deterministic)
        
        # Head
        x = self.norm_out(x)
        x = jax.nn.silu(x)
        
        if feat_cache is not None:
            CACHE_T = 2
            cache_x = x[:, -CACHE_T:, :, :, :]
            prev_cache = feat_cache[feat_idx]
            
            temp_cache = [prev_cache]
            x = self.conv_out(x, feat_cache=temp_cache, feat_idx=0)
            
            if cache_x.shape[1] < 2 and prev_cache is not None:
                last_frame = prev_cache[:, -1:, :, :, :]
                cache_x = jnp.concatenate([last_frame, cache_x], axis=1)
                
            feat_cache[feat_idx] = cache_x
            feat_idx += 1
        else:
            x = self.conv_out(x)
            
        return x, feat_idx, feat_cache


class FlaxWanDecoder3d(nnx.Module):
    """
    Matches WanDecoder3d
    """
    def __init__(
        self,
        dim: int = 128,
        z_dim: int = 4,
        dim_mult: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attn_scales: Tuple[float, ...] = (),
        temperal_upsample: Tuple[bool, ...] = (False, True, True),
        dropout: float = 0.0,
        non_linearity: str = "silu",
        out_channels: int = 3,
        is_residual: bool = False,
        rngs: nnx.Rngs = None,
    ):
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample
        
        # dims
        dims = [dim * u for u in [dim_mult[-1]] + list(dim_mult[::-1])]
        
        # init block
        self.conv_in = FlaxWanCausalConv3d(z_dim, dims[0], 3, padding=1, rngs=rngs)
        
        # middle block
        self.mid_block = FlaxWanMidBlock(dims[0], dropout, non_linearity, num_layers=1, rngs=rngs)
        
        # upsample blocks
        up_blocks = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i > 0 and not is_residual:
                in_dim = in_dim // 2
                
            up_flag = (i != len(dim_mult) - 1)
            upsample_mode = None
            if up_flag and temperal_upsample[i]:
                upsample_mode = "upsample3d"
            elif up_flag:
                upsample_mode = "upsample2d"
                
            if is_residual:
                block = FlaxWanResidualUpBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_res_blocks=num_res_blocks,
                    dropout=dropout,
                    temperal_upsample=temperal_upsample[i] if up_flag else False,
                    up_flag=up_flag,
                    non_linearity=non_linearity,
                    rngs=rngs
                )
            else:
                block = FlaxWanUpBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_res_blocks=num_res_blocks,
                    dropout=dropout,
                    upsample_mode=upsample_mode,
                    non_linearity=non_linearity,
                    rngs=rngs
                )
            up_blocks.append(block)
            
        self.up_blocks = nnx.List(up_blocks)
        
        # output blocks
        self.norm_out = FlaxWanRMS_norm(out_dim, images=False, rngs=rngs)
        self.conv_out = FlaxWanCausalConv3d(out_dim, out_channels, 3, padding=1, rngs=rngs)
        
    def __call__(self, x, feat_cache=None, feat_idx=0, first_chunk=False, deterministic: bool = True):
        # Conv in
        if feat_cache is not None:
            CACHE_T = 2
            cache_x = x[:, -CACHE_T:, :, :, :]
            prev_cache = feat_cache[feat_idx]
            
            temp_cache = [prev_cache]
            x = self.conv_in(x, feat_cache=temp_cache, feat_idx=0)
            
            if cache_x.shape[1] < 2 and prev_cache is not None:
                last_frame = prev_cache[:, -1:, :, :, :]
                cache_x = jnp.concatenate([last_frame, cache_x], axis=1)
                
            feat_cache[feat_idx] = cache_x
            feat_idx += 1
        else:
            x = self.conv_in(x)
            
        # Middle
        x, feat_idx, feat_cache = self.mid_block(x, feat_cache, feat_idx, deterministic)
        
        # Upsamples
        for up_block in self.up_blocks:
            x, feat_idx, feat_cache = up_block(x, feat_cache, feat_idx, first_chunk=first_chunk, deterministic=deterministic)
            
        # Head
        x = self.norm_out(x)
        x = jax.nn.silu(x)
        
        if feat_cache is not None:
            CACHE_T = 2
            cache_x = x[:, -CACHE_T:, :, :, :]
            prev_cache = feat_cache[feat_idx]
            
            temp_cache = [prev_cache]
            x = self.conv_out(x, feat_cache=temp_cache, feat_idx=0)
            
            if cache_x.shape[1] < 2 and prev_cache is not None:
                last_frame = prev_cache[:, -1:, :, :, :]
                cache_x = jnp.concatenate([last_frame, cache_x], axis=1)
                
            feat_cache[feat_idx] = cache_x
            feat_idx += 1
        else:
            x = self.conv_out(x)
            
        # Replicate back to every device (matching PyTorch mark_sharding(x, P()))
        # Flax version: jax.lax.with_sharding_constraint(x, P())
        try:
            x = jax.lax.with_sharding_constraint(x, P())
        except (ValueError, RuntimeError):
            pass
        
        return x, feat_idx, feat_cache



