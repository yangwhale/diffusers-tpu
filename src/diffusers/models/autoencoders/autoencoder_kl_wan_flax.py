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

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

# FLAX: jax/flax instead of torch
import jax
import jax.numpy as jnp
from flax import nnx

from jax.sharding import PartitionSpec as P

# FLAX: Flax equivalent of interop.torch_view(jax.lax.with_sharding_constraint)
def mark_sharding(inputs, spec):
    try:
        return jax.lax.with_sharding_constraint(inputs, spec)
    except (ValueError, Exception):
        return inputs

CACHE_T = 2


class AvgDown3D(nnx.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        factor_t,
        factor_s=1,
        rngs: nnx.Rngs = None,  # FLAX: rngs for Flax NNX
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s

        assert in_channels * self.factor % out_channels == 0
        self.group_size = in_channels * self.factor // out_channels

    def __call__(self, x):  # FLAX: __call__ instead of forward
        # FLAX: x is (B, T, H, W, C) in NTHWC format, not (B, C, T, H, W)
        B, T, H, W, C = x.shape
        pad_t = (self.factor_t - T % self.factor_t) % self.factor_t

        if pad_t > 0:
            # FLAX: jnp.pad instead of F.pad, different axis order
            pad_width = [(0, 0), (pad_t, 0), (0, 0), (0, 0), (0, 0)]
            x = jnp.pad(x, pad_width, mode='constant', constant_values=0)
            T = x.shape[1]

        # FLAX: reshape for NTHWC format
        x = x.reshape(B, T//self.factor_t, self.factor_t, H//self.factor_s, self.factor_s, W//self.factor_s, self.factor_s, C)
        x = x.transpose(0, 1, 3, 5, 7, 2, 4, 6)
        x = x.reshape(B, T//self.factor_t, H//self.factor_s, W//self.factor_s, C * self.factor)
        x = x.reshape(B, T//self.factor_t, H//self.factor_s, W//self.factor_s, self.out_channels, self.group_size)
        x = jnp.mean(x, axis=-1)
        return x


class DupUp3D(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor_t,
        factor_s=1,
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
        # FLAX: x is (B, T, H, W, C)
        B, T, H, W, C = x.shape
        # FLAX: jnp.repeat instead of repeat_interleave
        x = jnp.repeat(x[:, :, :, :, :, None], self.repeats, axis=5).reshape(B, T, H, W, C * self.repeats)
        x = x.reshape(B, T, H, W, self.out_channels, self.factor_t, self.factor_s, self.factor_s)
        x = x.transpose(0, 1, 5, 2, 6, 3, 7, 4)
        x = x.reshape(B, T * self.factor_t, H * self.factor_s, W * self.factor_s, self.out_channels)

        if first_chunk:
            x = x[:, self.factor_t - 1:, :, :, :]
        return x


class WanCausalConv3d(nnx.Module):
    r"""
    A custom 3D causal convolution layer with feature caching support.
    """

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

        # Set up causal padding (matches TorchAx)
        self._padding = (padding[2], padding[2], padding[1], padding[1], 2 * padding[0], 0)

        # FLAX: nnx.Conv instead of nn.Conv3d
        self.conv = nnx.Conv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            strides=stride,
            padding=((0, 0), (0, 0), (0, 0)),
            rngs=rngs,
        )

    def __call__(self, x, cache_x=None):
        # FLAX: x is (B, T, H, W, C)
        padding = list(self._padding)
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

        return self.conv(x)


class WanRMS_norm(nnx.Module):
    r"""
    A custom RMS normalization layer.
    """

    def __init__(self, dim: int, channel_first: bool = True, images: bool = True, bias: bool = False, rngs: nnx.Rngs = None):
        self.channel_first = channel_first
        self.scale = dim**0.5
        # FLAX: nnx.Param instead of nn.Parameter, shape (dim,) for channel-last
        self.gamma = nnx.Param(jnp.ones((dim,)))
        self.bias = nnx.Param(jnp.zeros((dim,))) if bias else None

    def __call__(self, x):
        # FLAX: F.normalize -> manual L2 normalize along channel dim
        norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
        norm = jnp.maximum(norm, 1e-12)
        x_normalized = x / norm
        out = x_normalized * self.scale * self.gamma.value
        if self.bias is not None:
            out = out + self.bias.value
        return out


class WanResample(nnx.Module):
    r"""
    A custom resampling module for 2D and 3D data.
    """

    def __init__(self, dim: int, mode: str, upsample_out_dim: int = None, rngs: nnx.Rngs = None):
        self.dim = dim
        self.mode = mode

        # default to dim //2
        if upsample_out_dim is None:
            upsample_out_dim = dim // 2

        # layers
        if mode == "upsample2d":
            # FLAX: nnx.Conv instead of nn.Conv2d
            self.resample_conv = nnx.Conv(dim, upsample_out_dim, kernel_size=(3, 3), padding=((1, 1), (1, 1)), rngs=rngs)
            self.time_conv = None
        elif mode == "upsample3d":
            self.resample_conv = nnx.Conv(dim, upsample_out_dim, kernel_size=(3, 3), padding=((1, 1), (1, 1)), rngs=rngs)
            self.time_conv = WanCausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0), rngs=rngs)
        elif mode == "downsample2d":
            # FLAX: padding ((0, 1), (0, 1)) for ZeroPad2d((0, 1, 0, 1))
            self.resample_conv = nnx.Conv(dim, dim, kernel_size=(3, 3), strides=(2, 2), padding=((0, 1), (0, 1)), rngs=rngs)
            self.time_conv = None
        elif mode == "downsample3d":
            self.resample_conv = nnx.Conv(dim, dim, kernel_size=(3, 3), strides=(2, 2), padding=((0, 1), (0, 1)), rngs=rngs)
            self.time_conv = WanCausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0), rngs=rngs)
        else:
            self.resample_conv = None
            self.time_conv = None

    def __call__(self, x, feat_cache=None, feat_idx=0):
        # FLAX: x is (B, T, H, W, C)
        B, T, H, W, C = x.shape

        if self.mode == "upsample3d":
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
                    if feat_cache[idx][0] is None:
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx][0])
                    feat_cache[idx] = (cache_x,)
                    feat_idx += 1

                    # FLAX: reshape for NTHWC
                    x = x.reshape(B, T, H, W, 2, self.dim)
                    x = x.transpose(0, 1, 4, 2, 3, 5).reshape(B, T * 2, H, W, self.dim)

        T_curr = x.shape[1]
        x = x.reshape(B * T_curr, H, W, C if self.mode not in ["upsample3d"] or feat_cache is None or feat_cache[feat_idx-1] == (None,) else self.dim)

        if self.mode in ["upsample2d", "upsample3d"]:
            # FLAX: jax.image.resize instead of nn.Upsample
            x = jax.image.resize(x, (B * T_curr, H * 2, W * 2, x.shape[-1]), method='nearest')
            x = self.resample_conv(x)
        elif self.mode in ["downsample2d", "downsample3d"]:
            x = self.resample_conv(x)
        # else: identity, x unchanged

        H_new, W_new = x.shape[1], x.shape[2]
        x = x.reshape(B, T_curr, H_new, W_new, x.shape[-1])

        if self.mode == "downsample3d":
            if feat_cache is not None:
                idx = feat_idx
                if feat_cache[idx] is None:
                    feat_cache[idx] = x
                    feat_idx += 1
                else:
                    cache_x = x[:, -1:, :, :, :]
                    x = self.time_conv(jnp.concatenate([feat_cache[idx][:, -1:, :, :, :], x], axis=1))
                    feat_cache[idx] = cache_x
                    feat_idx += 1

        return x, feat_idx, feat_cache


class WanResidualBlock(nnx.Module):
    r"""
    A custom residual block module.
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

        # layers
        self.norm1 = WanRMS_norm(in_dim, images=False, rngs=rngs)
        self.conv1 = WanCausalConv3d(in_dim, out_dim, 3, padding=1, rngs=rngs)
        self.norm2 = WanRMS_norm(out_dim, images=False, rngs=rngs)
        self.dropout = dropout
        self.conv2 = WanCausalConv3d(out_dim, out_dim, 3, padding=1, rngs=rngs)
        self.conv_shortcut = WanCausalConv3d(in_dim, out_dim, 1, rngs=rngs) if in_dim != out_dim else None

    def __call__(self, x, feat_cache=None, feat_idx=0, deterministic: bool = True):
        # Apply shortcut connection
        h = self.conv_shortcut(x) if self.conv_shortcut is not None else x

        # First normalization and activation
        x = self.norm1(x)
        x = jax.nn.silu(x)

        if feat_cache is not None:
            idx = feat_idx
            cache_x = x[:, -CACHE_T:, :, :, :]
            if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
                cache_x = jnp.concatenate([feat_cache[idx][:, -1:, :, :, :], cache_x], axis=1)

            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx += 1
        else:
            x = self.conv1(x)

        # Second normalization and activation
        x = self.norm2(x)
        x = jax.nn.silu(x)

        # Dropout (skip in deterministic mode)

        if feat_cache is not None:
            idx = feat_idx
            cache_x = x[:, -CACHE_T:, :, :, :]
            if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
                cache_x = jnp.concatenate([feat_cache[idx][:, -1:, :, :, :], cache_x], axis=1)

            x = self.conv2(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx += 1
        else:
            x = self.conv2(x)

        # Add residual connection
        return x + h, feat_idx, feat_cache


class WanAttentionBlock(nnx.Module):
    r"""
    Causal self-attention with a single head.
    """

    def __init__(self, dim, rngs: nnx.Rngs = None):
        self.dim = dim

        # layers
        self.norm = WanRMS_norm(dim, rngs=rngs)
        # FLAX: nnx.Conv for 2D conv
        self.to_qkv = nnx.Conv(dim, dim * 3, kernel_size=(1, 1), rngs=rngs)
        self.proj = nnx.Conv(dim, dim, kernel_size=(1, 1), rngs=rngs)

    def __call__(self, x):
        identity = x
        # FLAX: x is (B, T, H, W, C)
        B, T, H, W, C = x.shape

        x = x.reshape(B * T, H, W, C)
        x = self.norm(x)

        # compute query, key, value
        qkv = self.to_qkv(x)
        qkv = qkv.reshape(B * T, H * W, 3 * C)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        # add head dimension (1 head)
        q = q[:, None, :, :]
        k = k[:, None, :, :]
        v = v[:, None, :, :]

        # apply attention
        # FLAX: manual scaled dot product attention
        scale = 1.0 / jnp.sqrt(C)
        attn_weights = jnp.matmul(q, k.swapaxes(-1, -2)) * scale
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        x = jnp.matmul(attn_weights, v)

        x = x.squeeze(1).reshape(B * T, H, W, C)

        # output projection
        x = self.proj(x)

        # Reshape back
        x = x.reshape(B, T, H, W, C)

        return x + identity


class WanMidBlock(nnx.Module):
    """
    Middle block for WanVAE encoder and decoder.
    """

    def __init__(self, dim: int, dropout: float = 0.0, non_linearity: str = "silu", num_layers: int = 1, rngs: nnx.Rngs = None):
        self.dim = dim

        # Create the components
        resnets = [WanResidualBlock(dim, dim, dropout, non_linearity, rngs=rngs)]
        attentions = []
        for _ in range(num_layers):
            attentions.append(WanAttentionBlock(dim, rngs=rngs))
            resnets.append(WanResidualBlock(dim, dim, dropout, non_linearity, rngs=rngs))
        # FLAX: nnx.List instead of nn.ModuleList
        self.attentions = nnx.List(attentions)
        self.resnets = nnx.List(resnets)

    def __call__(self, x, feat_cache=None, feat_idx=0, deterministic: bool = True):
        # First residual block
        x, feat_idx, feat_cache = self.resnets[0](x, feat_cache, feat_idx)

        # Process through attention and residual blocks
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                x = attn(x)
            x, feat_idx, feat_cache = resnet(x, feat_cache, feat_idx)

        return x, feat_idx, feat_cache


class WanResidualDownBlock(nnx.Module):
    def __init__(self, in_dim, out_dim, dropout, num_res_blocks, temperal_downsample=False, down_flag=False, rngs: nnx.Rngs = None):
        # Shortcut path with downsample
        self.avg_shortcut = AvgDown3D(
            in_dim,
            out_dim,
            factor_t=2 if temperal_downsample else 1,
            factor_s=2 if down_flag else 1,
            rngs=rngs,
        )

        # Main path with residual blocks and downsample
        resnets = []
        for _ in range(num_res_blocks):
            resnets.append(WanResidualBlock(in_dim, out_dim, dropout, rngs=rngs))
            in_dim = out_dim
        self.resnets = nnx.List(resnets)

        # Add the final downsample block
        if down_flag:
            mode = "downsample3d" if temperal_downsample else "downsample2d"
            self.downsampler = WanResample(out_dim, mode=mode, rngs=rngs)
        else:
            self.downsampler = None

    def __call__(self, x, feat_cache=None, feat_idx=0, deterministic: bool = True):
        x_copy = x
        for resnet in self.resnets:
            x, feat_idx, feat_cache = resnet(x, feat_cache, feat_idx)
        if self.downsampler is not None:
            x, feat_idx, feat_cache = self.downsampler(x, feat_cache, feat_idx)

        return x + self.avg_shortcut(x_copy), feat_idx, feat_cache


class WanEncoder3d(nnx.Module):
    r"""
    A 3D encoder module.
    """

    def __init__(
        self,
        in_channels: int = 3,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[True, True, False],
        dropout=0.0,
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

        # dimensions
        dims = [dim * u for u in [1] + list(dim_mult)]
        scale = 1.0

        # init block
        self.conv_in = WanCausalConv3d(in_channels, dims[0], 3, padding=1, rngs=rngs)

        # downsample blocks
        down_blocks = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            if is_residual:
                down_blocks.append(
                    WanResidualDownBlock(
                        in_dim,
                        out_dim,
                        dropout,
                        num_res_blocks,
                        temperal_downsample=temperal_downsample[i] if i != len(dim_mult) - 1 else False,
                        down_flag=i != len(dim_mult) - 1,
                        rngs=rngs,
                    )
                )
            else:
                for _ in range(num_res_blocks):
                    down_blocks.append(WanResidualBlock(in_dim, out_dim, dropout, rngs=rngs))
                    if scale in attn_scales:
                        down_blocks.append(WanAttentionBlock(out_dim, rngs=rngs))
                    in_dim = out_dim

                # downsample block
                if i != len(dim_mult) - 1:
                    mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
                    down_blocks.append(WanResample(out_dim, mode=mode, rngs=rngs))
                    scale /= 2.0

        self.down_blocks = nnx.List(down_blocks)

        # middle blocks
        self.mid_block = WanMidBlock(out_dim, dropout, non_linearity, num_layers=1, rngs=rngs)

        # output blocks
        self.norm_out = WanRMS_norm(out_dim, images=False, rngs=rngs)
        self.conv_out = WanCausalConv3d(out_dim, z_dim, 3, padding=1, rngs=rngs)

    def __call__(self, x, feat_cache=None, deterministic: bool = True):
        feat_idx = 0
        if feat_cache is not None:
            idx = feat_idx
            cache_x = x[:, -CACHE_T:, :, :, :]
            if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
                cache_x = jnp.concatenate([feat_cache[idx][:, -1:, :, :, :], cache_x], axis=1)
            x = self.conv_in(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx += 1
        else:
            x = self.conv_in(x)

        ## downsamples
        for layer in self.down_blocks:
            if isinstance(layer, (WanResidualBlock, WanResidualDownBlock, WanResample)):
                x, feat_idx, feat_cache = layer(x, feat_cache, feat_idx)
            elif isinstance(layer, WanAttentionBlock):
                x = layer(x)

        ## middle
        x, feat_idx, feat_cache = self.mid_block(x, feat_cache, feat_idx)

        ## head
        x = self.norm_out(x)
        x = jax.nn.silu(x)
        if feat_cache is not None:
            idx = feat_idx
            cache_x = x[:, -CACHE_T:, :, :, :]
            if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
                cache_x = jnp.concatenate([feat_cache[idx][:, -1:, :, :, :], cache_x], axis=1)
            x = self.conv_out(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx += 1
        else:
            x = self.conv_out(x)
        return x, feat_idx, feat_cache


class WanResidualUpBlock(nnx.Module):
    """
    A block that handles upsampling for the WanVAE decoder.
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
            self.avg_shortcut = DupUp3D(
                in_dim,
                out_dim,
                factor_t=2 if temperal_upsample else 1,
                factor_s=2,
                rngs=rngs,
            )
        else:
            self.avg_shortcut = None

        # create residual blocks
        resnets = []
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(WanResidualBlock(current_dim, out_dim, dropout, non_linearity, rngs=rngs))
            current_dim = out_dim

        self.resnets = nnx.List(resnets)

        # Add upsampling layer if needed
        if up_flag:
            upsample_mode = "upsample3d" if temperal_upsample else "upsample2d"
            self.upsampler = WanResample(out_dim, mode=upsample_mode, upsample_out_dim=out_dim, rngs=rngs)
        else:
            self.upsampler = None

    def __call__(self, x, feat_cache=None, feat_idx=0, first_chunk=False, deterministic: bool = True):
        x_copy = x

        for resnet in self.resnets:
            x, feat_idx, feat_cache = resnet(x, feat_cache, feat_idx)

        if self.upsampler is not None:
            x, feat_idx, feat_cache = self.upsampler(x, feat_cache, feat_idx)

        if self.avg_shortcut is not None:
            x = x + self.avg_shortcut(x_copy, first_chunk=first_chunk)

        return x, feat_idx, feat_cache


class WanUpBlock(nnx.Module):
    """
    A block that handles upsampling for the WanVAE decoder.
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

        # Create layers list
        resnets = []
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(WanResidualBlock(current_dim, out_dim, dropout, non_linearity, rngs=rngs))
            current_dim = out_dim

        self.resnets = nnx.List(resnets)

        # Add upsampling layer if needed
        # FLAX: Always create list (can be empty), avoid None -> nnx.List reassignment
        if upsample_mode is not None:
            self.upsamplers = nnx.List([WanResample(out_dim, mode=upsample_mode, rngs=rngs)])
        else:
            self.upsamplers = nnx.List([])

    def __call__(self, x, feat_cache=None, feat_idx=0, first_chunk=None, deterministic: bool = True):
        for resnet in self.resnets:
            x, feat_idx, feat_cache = resnet(x, feat_cache, feat_idx)

        if len(self.upsamplers) > 0:
            x, feat_idx, feat_cache = self.upsamplers[0](x, feat_cache, feat_idx)
        return x, feat_idx, feat_cache


class WanDecoder3d(nnx.Module):
    r"""
    A 3D decoder module.
    """

    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_upsample=[False, True, True],
        dropout=0.0,
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

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]

        # init block
        self.conv_in = WanCausalConv3d(z_dim, dims[0], 3, padding=1, rngs=rngs)

        # middle blocks
        self.mid_block = WanMidBlock(dims[0], dropout, non_linearity, num_layers=1, rngs=rngs)

        # upsample blocks
        up_blocks = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            if i > 0 and not is_residual:
                # wan vae 2.1
                in_dim = in_dim // 2

            # determine if we need upsampling
            up_flag = i != len(dim_mult) - 1
            # determine upsampling mode
            upsample_mode = None
            if up_flag and temperal_upsample[i]:
                upsample_mode = "upsample3d"
            elif up_flag:
                upsample_mode = "upsample2d"
            # Create and add the upsampling block
            if is_residual:
                up_block = WanResidualUpBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_res_blocks=num_res_blocks,
                    dropout=dropout,
                    temperal_upsample=temperal_upsample[i] if up_flag else False,
                    up_flag=up_flag,
                    non_linearity=non_linearity,
                    rngs=rngs,
                )
            else:
                up_block = WanUpBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_res_blocks=num_res_blocks,
                    dropout=dropout,
                    upsample_mode=upsample_mode,
                    non_linearity=non_linearity,
                    rngs=rngs,
                )
            up_blocks.append(up_block)

        self.up_blocks = nnx.List(up_blocks)

        # output blocks
        self.norm_out = WanRMS_norm(out_dim, images=False, rngs=rngs)
        self.conv_out = WanCausalConv3d(out_dim, out_channels, 3, padding=1, rngs=rngs)

    def __call__(self, x, feat_cache=None, first_chunk=False, deterministic: bool = True):
        feat_idx = 0
        ## conv1
        if feat_cache is not None:
            idx = feat_idx
            cache_x = x[:, -CACHE_T:, :, :, :]
            if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
                cache_x = jnp.concatenate([feat_cache[idx][:, -1:, :, :, :], cache_x], axis=1)
            x = self.conv_in(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx += 1
        else:
            x = self.conv_in(x)

        ## middle
        x, feat_idx, feat_cache = self.mid_block(x, feat_cache, feat_idx)

        ## upsamples
        for up_block in self.up_blocks:
            x, feat_idx, feat_cache = up_block(x, feat_cache, feat_idx, first_chunk=first_chunk)

        ## head
        x = self.norm_out(x)
        x = jax.nn.silu(x)
        if feat_cache is not None:
            idx = feat_idx
            cache_x = x[:, -CACHE_T:, :, :, :]
            if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
                cache_x = jnp.concatenate([feat_cache[idx][:, -1:, :, :, :], cache_x], axis=1)
            x = self.conv_out(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx += 1
        else:
            x = self.conv_out(x)

        # Replicate back to every devices
        x = mark_sharding(x, P())
        return x, feat_idx, feat_cache


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


@dataclass
class AutoencoderKLWanConfig:
    """
    Configuration class for AutoencoderKLWan (Flax version).
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


class AutoencoderKLWan(nnx.Module, pytree=False):
    r"""
    A VAE model with KL loss for encoding videos into latents and decoding latent representations into videos.
    Introduced in [Wan 2.1].
    
    FLAX: Flax NNX version of TorchAx AutoencoderKLWan.
    FLAX: pytree=False to allow mutable cache attributes.
    """

    config_class = AutoencoderKLWanConfig

    def __init__(
        self,
        config: AutoencoderKLWanConfig,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
    ):
        super().__init__()
        self.config = config
        self.dtype = dtype

        self.z_dim = config.z_dim
        self.temperal_downsample = config.temperal_downsample
        self.temperal_upsample = tuple(reversed(config.temperal_downsample))

        decoder_base_dim = config.decoder_base_dim
        if decoder_base_dim is None:
            decoder_base_dim = config.base_dim

        self.encoder = WanEncoder3d(
            in_channels=config.in_channels,
            dim=config.base_dim,
            z_dim=config.z_dim * 2,
            dim_mult=list(config.dim_mult),
            num_res_blocks=config.num_res_blocks,
            attn_scales=list(config.attn_scales),
            temperal_downsample=list(config.temperal_downsample),
            dropout=config.dropout,
            is_residual=config.is_residual,
            rngs=rngs,
        )
        self.quant_conv = WanCausalConv3d(config.z_dim * 2, config.z_dim * 2, 1, rngs=rngs)
        self.post_quant_conv = WanCausalConv3d(config.z_dim, config.z_dim, 1, rngs=rngs)

        self.decoder = WanDecoder3d(
            dim=decoder_base_dim,
            z_dim=config.z_dim,
            dim_mult=list(config.dim_mult),
            num_res_blocks=config.num_res_blocks,
            attn_scales=list(config.attn_scales),
            temperal_upsample=list(self.temperal_upsample),
            dropout=config.dropout,
            out_channels=config.out_channels,
            is_residual=config.is_residual,
            rngs=rngs,
        )

        self.spatial_compression_ratio = config.scale_factor_spatial

        self.use_slicing = False
        self.use_tiling = False
        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256
        self.tile_sample_stride_height = 192
        self.tile_sample_stride_width = 192

        # Precompute conv counts
        self._cached_conv_counts = {
            "decoder": self._count_causal_conv3d(self.decoder) if self.decoder is not None else 0,
            "encoder": self._count_causal_conv3d(self.encoder) if self.encoder is not None else 0,
        }

    def _count_causal_conv3d(self, module):
        count = 0
        for _, value in nnx.graph.iter_graph([module]):
            if isinstance(value, WanCausalConv3d):
                count += 1
        return count

    def clear_cache(self):
        self._conv_num = self._cached_conv_counts["decoder"]
        self._feat_map = [None] * self._conv_num
        self._enc_conv_num = self._cached_conv_counts["encoder"]
        self._enc_feat_map = [None] * self._enc_conv_num

    def _encode(self, x: jnp.ndarray):
        # FLAX: x is (B, T, H, W, C)
        B, T, H, W, C = x.shape

        self.clear_cache()
        if self.config.patch_size is not None:
            x = patchify(x, patch_size=self.config.patch_size)

        iter_ = 1 + (T - 1) // 4
        out = None
        for i in range(iter_):
            if i == 0:
                out, _, self._enc_feat_map = self.encoder(x[:, :1, :, :, :], feat_cache=self._enc_feat_map)
            else:
                out_, _, self._enc_feat_map = self.encoder(
                    x[:, 1 + 4 * (i - 1) : 1 + 4 * i, :, :, :],
                    feat_cache=self._enc_feat_map,
                )
                out = jnp.concatenate([out, out_], axis=1)

        enc = self.quant_conv(out)
        self.clear_cache()
        return enc

    def encode(self, x: jnp.ndarray):
        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self._encode(x[i:i+1]) for i in range(x.shape[0])]
            h = jnp.concatenate(encoded_slices, axis=0)
        else:
            h = self._encode(x)

        # DiagonalGaussianDistribution
        mean, logvar = jnp.split(h, 2, axis=-1)
        return mean, logvar

    def _decode(self, z: jnp.ndarray):
        # FLAX: z is (B, T, H, W, C)
        B, T, H, W, C = z.shape

        self.clear_cache()
        x = self.post_quant_conv(z)
        out = None
        for i in range(T):
            if i == 0:
                out, _, self._feat_map = self.decoder(
                    x[:, i : i + 1, :, :, :],
                    feat_cache=self._feat_map,
                    first_chunk=True,
                )
            else:
                out_, _, self._feat_map = self.decoder(x[:, i : i + 1, :, :, :], feat_cache=self._feat_map)
                out = jnp.concatenate([out, out_], axis=1)

        if self.config.patch_size is not None:
            out = unpatchify(out, patch_size=self.config.patch_size)

        out = jnp.clip(out, -1.0, 1.0)

        self.clear_cache()
        return out

    def decode(self, z: jnp.ndarray):
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z[i:i+1]) for i in range(z.shape[0])]
            decoded = jnp.concatenate(decoded_slices, axis=0)
        else:
            decoded = self._decode(z)
        return decoded

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        subfolder: Optional[str] = "vae",
        dtype: jnp.dtype = jnp.float32,
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

        # 3. Convert weights (PyTorch -> Flax)
        jax_weights = {}

        def rename_key(key):
            # Map PyTorch module paths to Flax paths
            key = key.replace("conv_in.bias", "conv_in.conv.bias")
            key = key.replace("conv_in.weight", "conv_in.conv.kernel")
            key = key.replace("conv_out.bias", "conv_out.conv.bias")
            key = key.replace("conv_out.weight", "conv_out.conv.kernel")

            key = re.sub(r"conv(\d+)\.weight", r"conv\1.conv.kernel", key)
            key = re.sub(r"conv(\d+)\.bias", r"conv\1.conv.bias", key)

            key = key.replace("time_conv.weight", "time_conv.conv.kernel")
            key = key.replace("time_conv.bias", "time_conv.conv.bias")

            key = key.replace("quant_conv.weight", "quant_conv.conv.kernel")
            key = key.replace("quant_conv.bias", "quant_conv.conv.bias")

            key = key.replace("conv_shortcut.weight", "conv_shortcut.conv.kernel")
            key = key.replace("conv_shortcut.bias", "conv_shortcut.conv.bias")

            # Resample layers
            key = key.replace("resample.1.weight", "resample_conv.kernel")
            key = key.replace("resample.1.bias", "resample_conv.bias")

            # Attention
            key = key.replace("to_qkv.weight", "to_qkv.kernel")
            key = key.replace("to_qkv.bias", "to_qkv.bias")
            key = key.replace("proj.weight", "proj.kernel")
            key = key.replace("proj.bias", "proj.bias")

            # Norm
            key = key.replace(".weight", ".gamma")

            return key

        for pt_key, pt_tensor in pytorch_weights.items():
            flax_key = rename_key(pt_key)

            # Convert tensor shape
            if "kernel" in flax_key:
                if pt_tensor.ndim == 5:
                    # Conv3d: (Out, In, T, H, W) -> (T, H, W, In, Out)
                    pt_tensor = pt_tensor.transpose(2, 3, 4, 1, 0)
                elif pt_tensor.ndim == 4:
                    # Conv2d: (Out, In, H, W) -> (H, W, In, Out)
                    pt_tensor = pt_tensor.transpose(2, 3, 1, 0)

            if "gamma" in flax_key:
                pt_tensor = pt_tensor.squeeze()

            jax_weights[flax_key] = jnp.array(pt_tensor, dtype=dtype)

        # 4. Initialize
        key = jax.random.key(0)
        rngs = nnx.Rngs(key)
        model = cls(config=config, rngs=rngs, dtype=dtype)

        # Load weights
        nested_weights = unflatten_dict(jax_weights, sep=".")
        graphdef, _ = nnx.split(model)
        model = nnx.merge(graphdef, nested_weights)

        return model
