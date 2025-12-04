# Copyright 2025 The Hunyuan Team and The HuggingFace Team. All rights reserved.
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
Flax/JAX implementation of HunyuanVideo-1.5 VAE.

This implementation includes:
- CausalConv3d with feat_cache support for frame-by-frame decoding
- RMS normalization
- Attention blocks with causal masking
- DCAE-style upsampling/downsampling with channel rearrangement
- Tiling for memory efficiency
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np


@dataclass
class FlaxAutoencoderKLHunyuanVideo15Config:
    """Configuration class for FlaxAutoencoderKLHunyuanVideo15."""
    config_name: str = "config.json"
    
    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 32
    block_out_channels: Tuple[int, ...] = (128, 256, 512, 1024, 1024)
    layers_per_block: int = 2
    spatial_compression_ratio: int = 16
    temporal_compression_ratio: int = 4
    downsample_match_channel: bool = True
    upsample_match_channel: bool = True
    scaling_factor: float = 1.03682
    
    @classmethod
    def from_dict(cls, config_dict: Dict):
        """Create config from dictionary"""
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(cls)}
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}
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


class FlaxHunyuanVideo15CausalConv3d(nnx.Module):
    """
    A 3D causal convolution layer for HunyuanVideo15.
    
    Supports:
    - Time dimension causal padding
    - feat_cache/feat_idx for frame-by-frame processing
    - Replicate padding mode
    """
    
    CACHE_T = 2
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        pad_mode: str = "replicate",
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        
        self.kernel_size = kernel_size
        self.pad_mode = pad_mode
        
        self.time_causal_padding = (
            kernel_size[0] // 2,
            kernel_size[0] // 2,
            kernel_size[1] // 2,
            kernel_size[1] // 2,
            kernel_size[2] - 1,
            0,
        )
        
        self.time_kernel_size = kernel_size[2]
        
        if isinstance(stride, int):
            stride = (stride,) * 3
            
        self.conv = FlaxConv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            rngs=rngs,
        )
    
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        feat_cache: Optional[list] = None,
        feat_idx: Optional[list] = None,
    ):
        if feat_cache is not None and feat_idx is not None:
            return self._call_with_feat_cache(hidden_states, feat_cache, feat_idx)
        
        return self._call_standard(hidden_states)
    
    def _call_standard(self, hidden_states: jnp.ndarray):
        # time_causal_padding = (h_left, h_right, w_left, w_right, t_before, t_after)
        # JAX format: (batch, time, height, width, channel)
        pad_width = [
            (0, 0),  # batch
            (self.time_causal_padding[4], self.time_causal_padding[5]),  # time
            (self.time_causal_padding[0], self.time_causal_padding[1]),  # height
            (self.time_causal_padding[2], self.time_causal_padding[3]),  # width
            (0, 0),  # channel
        ]
        
        if self.pad_mode == "replicate":
            hidden_states = jnp.pad(hidden_states, pad_width, mode='edge')
        else:
            hidden_states = jnp.pad(hidden_states, pad_width, mode='constant')
        
        output = self.conv(hidden_states)
        return output
    
    def _call_with_feat_cache(
        self,
        hidden_states: jnp.ndarray,
        feat_cache: list,
        feat_idx: list,
    ):
        """
        使用 feat_cache 的缓存模式（参考 CogVideoX 实现）。
        
        关键点：
        1. 先对输入应用空间填充
        2. 使用缓存处理时间维度填充
        3. 缓存存储空间填充后的输入（保持形状一致）
        """
        idx = feat_idx[0]
        
        # 空间填充 (height, width only, time handled by cache)
        # time_causal_padding = (h_left, h_right, w_left, w_right, t_before, t_after)
        spatial_pad_width = [
            (0, 0),  # batch
            (0, 0),  # time - 不在这里填充，由缓存处理
            (self.time_causal_padding[0], self.time_causal_padding[1]),  # height
            (self.time_causal_padding[2], self.time_causal_padding[3]),  # width
            (0, 0),  # channel
        ]
        
        if self.pad_mode == "replicate":
            x = jnp.pad(hidden_states, spatial_pad_width, mode='edge')
        else:
            x = jnp.pad(hidden_states, spatial_pad_width, mode='constant')
        
        # 时间维度填充（使用缓存）
        if self.time_kernel_size > 1:
            padding_needed = self.time_kernel_size - 1
            
            if feat_cache[idx] is not None:
                # 缓存已经是空间填充后的形状
                cache_len = feat_cache[idx].shape[1]
                x = jnp.concatenate([feat_cache[idx], x], axis=1)
                
                padding_needed -= cache_len
                if padding_needed > 0:
                    # 缓存不够，用第一帧扩展
                    extra_padding = jnp.tile(x[:, :1, :, :, :], (1, padding_needed, 1, 1, 1))
                    x = jnp.concatenate([extra_padding, x], axis=1)
                elif padding_needed < 0:
                    # 缓存太多，裁剪
                    x = x[:, -padding_needed:, ...]
            else:
                # 首次调用：用第一帧填充
                padding_frames = jnp.tile(x[:, :1, :, :, :], (1, padding_needed, 1, 1, 1))
                x = jnp.concatenate([padding_frames, x], axis=1)
            
            # 更新缓存：存储空间填充后的输入（保持形状一致）
            # 关键：存储 x 的最后 CACHE_T 帧（x 是空间填充后的）
            # 但要注意 x 现在包含了时间填充，需要从原始空间填充的输入取
            if self.pad_mode == "replicate":
                x_for_cache = jnp.pad(hidden_states, spatial_pad_width, mode='edge')
            else:
                x_for_cache = jnp.pad(hidden_states, spatial_pad_width, mode='constant')
            
            if hidden_states.shape[1] < self.CACHE_T and feat_cache[idx] is not None:
                # 输入帧数不足 CACHE_T：从旧缓存取最后 1 帧 + 当前输入
                feat_cache[idx] = jnp.concatenate([
                    jnp.expand_dims(feat_cache[idx][:, -1, :, :, :], axis=1),
                    x_for_cache[:, -self.CACHE_T:, :, :, :]
                ], axis=1)
            else:
                # 输入帧数足够：直接取最后 CACHE_T 帧
                feat_cache[idx] = x_for_cache[:, -self.CACHE_T:, :, :, :]
        
        feat_idx[0] += 1
        output = self.conv(x)
        return output


class FlaxHunyuanVideo15RMSNorm(nnx.Module):
    """RMS normalization layer for HunyuanVideo15."""
    
    def __init__(
        self,
        dim: int,
        channel_first: bool = True,
        images: bool = True,
        bias: bool = False,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        
        self.channel_first = channel_first
        self.scale = dim ** 0.5
        
        self.gamma = nnx.Param(jnp.ones((dim,)))
        self.has_bias = bias
        if bias:
            self.bias = nnx.Param(jnp.zeros((dim,)))
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x_norm = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
        result = x_norm * self.scale * self.gamma.value
        
        if self.has_bias:
            result = result + self.bias.value
        
        return result


class FlaxHunyuanVideo15AttnBlock(nnx.Module):
    """Attention block with causal masking for HunyuanVideo15."""
    
    def __init__(self, in_channels: int, rngs: nnx.Rngs = None):
        super().__init__()
        self.in_channels = in_channels
        
        self.norm = FlaxHunyuanVideo15RMSNorm(in_channels, images=False, rngs=rngs)
        
        self.to_q = FlaxConv3d(in_channels, in_channels, kernel_size=1, rngs=rngs)
        self.to_k = FlaxConv3d(in_channels, in_channels, kernel_size=1, rngs=rngs)
        self.to_v = FlaxConv3d(in_channels, in_channels, kernel_size=1, rngs=rngs)
        self.proj_out = FlaxConv3d(in_channels, in_channels, kernel_size=1, rngs=rngs)
    
    @staticmethod
    def prepare_causal_attention_mask(n_frame: int, n_hw: int, dtype, batch_size: int = None):
        seq_len = n_frame * n_hw
        mask = jnp.full((seq_len, seq_len), float("-inf"), dtype=dtype)
        
        for i in range(seq_len):
            i_frame = i // n_hw
            mask = mask.at[i, :(i_frame + 1) * n_hw].set(0)
        
        if batch_size is not None:
            mask = jnp.tile(mask[None, :, :], (batch_size, 1, 1))
        
        return mask
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        identity = x
        
        x = self.norm(x)
        
        query = self.to_q(x)
        key = self.to_k(x)
        value = self.to_v(x)
        
        batch_size, frames, height, width, channels = query.shape
        
        query = query.reshape(batch_size, frames * height * width, channels)
        key = key.reshape(batch_size, frames * height * width, channels)
        value = value.reshape(batch_size, frames * height * width, channels)
        
        query = query[:, None, :, :]
        key = key[:, None, :, :]
        value = value[:, None, :, :]
        
        attention_mask = self.prepare_causal_attention_mask(
            frames, height * width, query.dtype, batch_size=batch_size
        )
        attention_mask = attention_mask[:, None, :, :]
        
        scale = 1.0 / jnp.sqrt(channels)
        attn_weights = jnp.einsum('bhqc,bhkc->bhqk', query, key) * scale
        attn_weights = attn_weights + attention_mask
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        
        x = jnp.einsum('bhqk,bhkc->bhqc', attn_weights, value)
        
        x = x.squeeze(1)
        x = x.reshape(batch_size, frames, height, width, channels)
        
        x = self.proj_out(x)
        
        return x + identity


class FlaxHunyuanVideo15Upsample(nnx.Module):
    """
    DCAE-style upsampling for HunyuanVideo15.
    
    时间上采样逻辑：
    - 第一帧：只做空间上采样 (1 → 1 帧)
    - 后续帧：时空上采样 (1 → 2 帧)
    
    逐帧处理时需要 is_first_frame 参数来正确选择分支。
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        add_temporal_upsample: bool = True,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        
        factor = 2 * 2 * 2 if add_temporal_upsample else 1 * 2 * 2
        self.conv = FlaxHunyuanVideo15CausalConv3d(
            in_channels, out_channels * factor, kernel_size=3, rngs=rngs
        )
        
        self.add_temporal_upsample = add_temporal_upsample
        self.repeats = factor * out_channels // in_channels
    
    @staticmethod
    def _dcae_upsample_rearrange(tensor, r1=1, r2=2, r3=2):
        b, t, h, w, packed_c = tensor.shape
        factor = r1 * r2 * r3
        c = packed_c // factor
        
        tensor = tensor.reshape(b, t, h, w, r1, r2, r3, c)
        tensor = tensor.transpose(0, 1, 4, 2, 5, 3, 6, 7)
        tensor = tensor.reshape(b, t * r1, h * r2, w * r3, c)
        
        return tensor
    
    def __call__(
        self,
        x: jnp.ndarray,
        feat_cache: Optional[list] = None,
        feat_idx: Optional[list] = None,
        is_first_frame: Optional[bool] = None,
    ) -> jnp.ndarray:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, T, H, W, C)
            feat_cache: 缓存列表
            feat_idx: 当前索引
            is_first_frame: 是否是第一帧（逐帧解码时使用）
                - None: 完整处理（T > 1）
                - True: 第一帧，只空间上采样
                - False: 后续帧，时空上采样
        """
        r1 = 2 if self.add_temporal_upsample else 1
        
        if feat_cache is not None and feat_idx is not None:
            h = self.conv(x, feat_cache=feat_cache, feat_idx=feat_idx)
        else:
            h = self.conv(x)
        
        if self.add_temporal_upsample:
            # 逐帧处理模式
            if is_first_frame is not None:
                if is_first_frame:
                    # 第一帧：只空间上采样，输出 1 帧
                    h = self._dcae_upsample_rearrange(h, r1=1, r2=2, r3=2)
                    h = h[:, :, :, :, :h.shape[-1] // 2]
                    
                    shortcut = self._dcae_upsample_rearrange(x, r1=1, r2=2, r3=2)
                    shortcut = jnp.repeat(shortcut, self.repeats // 2, axis=-1)
                else:
                    # 后续帧：时空上采样，输出 2 帧
                    h = self._dcae_upsample_rearrange(h, r1=r1, r2=2, r3=2)
                    
                    shortcut = self._dcae_upsample_rearrange(x, r1=r1, r2=2, r3=2)
                    shortcut = jnp.repeat(shortcut, self.repeats, axis=-1)
            else:
                # 完整处理模式 (T > 1)
                h_first = h[:, :1, :, :, :]
                h_first = self._dcae_upsample_rearrange(h_first, r1=1, r2=2, r3=2)
                h_first = h_first[:, :, :, :, :h_first.shape[-1] // 2]
                
                h_next = h[:, 1:, :, :, :]
                h_next = self._dcae_upsample_rearrange(h_next, r1=r1, r2=2, r3=2)
                h = jnp.concatenate([h_first, h_next], axis=1)
                
                x_first = x[:, :1, :, :, :]
                x_first = self._dcae_upsample_rearrange(x_first, r1=1, r2=2, r3=2)
                x_first = jnp.repeat(x_first, self.repeats // 2, axis=-1)
                
                x_next = x[:, 1:, :, :, :]
                x_next = self._dcae_upsample_rearrange(x_next, r1=r1, r2=2, r3=2)
                x_next = jnp.repeat(x_next, self.repeats, axis=-1)
                shortcut = jnp.concatenate([x_first, x_next], axis=1)
        else:
            h = self._dcae_upsample_rearrange(h, r1=r1, r2=2, r3=2)
            shortcut = jnp.repeat(x, self.repeats, axis=-1)
            shortcut = self._dcae_upsample_rearrange(shortcut, r1=r1, r2=2, r3=2)
        
        return h + shortcut


class FlaxHunyuanVideo15Downsample(nnx.Module):
    """DCAE-style downsampling for HunyuanVideo15."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        add_temporal_downsample: bool = True,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        
        factor = 2 * 2 * 2 if add_temporal_downsample else 1 * 2 * 2
        self.conv = FlaxHunyuanVideo15CausalConv3d(
            in_channels, out_channels // factor, kernel_size=3, rngs=rngs
        )
        
        self.add_temporal_downsample = add_temporal_downsample
        self.group_size = factor * in_channels // out_channels
    
    @staticmethod
    def _dcae_downsample_rearrange(tensor, r1=1, r2=2, r3=2):
        b, packed_t, packed_h, packed_w, c = tensor.shape
        t, h, w = packed_t // r1, packed_h // r2, packed_w // r3
        
        tensor = tensor.reshape(b, t, r1, h, r2, w, r3, c)
        tensor = tensor.transpose(0, 1, 3, 5, 2, 4, 6, 7)
        tensor = tensor.reshape(b, t, h, w, r1 * r2 * r3 * c)
        
        return tensor
    
    def __call__(
        self,
        x: jnp.ndarray,
        feat_cache: Optional[list] = None,
        feat_idx: Optional[list] = None,
    ) -> jnp.ndarray:
        r1 = 2 if self.add_temporal_downsample else 1
        
        if feat_cache is not None and feat_idx is not None:
            h = self.conv(x, feat_cache=feat_cache, feat_idx=feat_idx)
        else:
            h = self.conv(x)
        
        if self.add_temporal_downsample:
            h_first = h[:, :1, :, :, :]
            h_first = self._dcae_downsample_rearrange(h_first, r1=1, r2=2, r3=2)
            h_first = jnp.concatenate([h_first, h_first], axis=-1)
            
            h_next = h[:, 1:, :, :, :]
            h_next = self._dcae_downsample_rearrange(h_next, r1=r1, r2=2, r3=2)
            h = jnp.concatenate([h_first, h_next], axis=1)
            
            x_first = x[:, :1, :, :, :]
            x_first = self._dcae_downsample_rearrange(x_first, r1=1, r2=2, r3=2)
            B, T, H, W, C = x_first.shape
            x_first = x_first.reshape(B, T, H, W, h.shape[-1], self.group_size // 2).mean(axis=-1)
            
            x_next = x[:, 1:, :, :, :]
            x_next = self._dcae_downsample_rearrange(x_next, r1=r1, r2=2, r3=2)
            B, T, H, W, C = x_next.shape
            x_next = x_next.reshape(B, T, H, W, h.shape[-1], self.group_size).mean(axis=-1)
            shortcut = jnp.concatenate([x_first, x_next], axis=1)
        else:
            h = self._dcae_downsample_rearrange(h, r1=r1, r2=2, r3=2)
            shortcut = self._dcae_downsample_rearrange(x, r1=r1, r2=2, r3=2)
            B, T, H, W, C = shortcut.shape
            shortcut = shortcut.reshape(B, T, H, W, h.shape[-1], self.group_size).mean(axis=-1)
        
        return h + shortcut


class FlaxHunyuanVideo15ResnetBlock(nnx.Module):
    """ResNet block for HunyuanVideo15."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        non_linearity: str = "swish",
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.norm1 = FlaxHunyuanVideo15RMSNorm(in_channels, images=False, rngs=rngs)
        self.conv1 = FlaxHunyuanVideo15CausalConv3d(in_channels, out_channels, kernel_size=3, rngs=rngs)
        
        self.norm2 = FlaxHunyuanVideo15RMSNorm(out_channels, images=False, rngs=rngs)
        self.conv2 = FlaxHunyuanVideo15CausalConv3d(out_channels, out_channels, kernel_size=3, rngs=rngs)
        
        self.conv_shortcut = None
        if in_channels != out_channels:
            self.conv_shortcut = FlaxConv3d(in_channels, out_channels, kernel_size=1, rngs=rngs)
    
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        feat_cache: Optional[list] = None,
        feat_idx: Optional[list] = None,
    ) -> jnp.ndarray:
        residual = hidden_states
        
        hidden_states = self.norm1(hidden_states)
        hidden_states = jax.nn.silu(hidden_states)
        
        if feat_cache is not None and feat_idx is not None:
            hidden_states = self.conv1(hidden_states, feat_cache=feat_cache, feat_idx=feat_idx)
        else:
            hidden_states = self.conv1(hidden_states)
        
        hidden_states = self.norm2(hidden_states)
        hidden_states = jax.nn.silu(hidden_states)
        
        if feat_cache is not None and feat_idx is not None:
            hidden_states = self.conv2(hidden_states, feat_cache=feat_cache, feat_idx=feat_idx)
        else:
            hidden_states = self.conv2(hidden_states)
        
        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)
        
        return hidden_states + residual


class FlaxHunyuanVideo15MidBlock(nnx.Module):
    """Middle block for HunyuanVideo15."""
    
    def __init__(
        self,
        in_channels: int,
        num_layers: int = 1,
        add_attention: bool = True,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        self.add_attention = add_attention
        self.num_layers = num_layers
        
        resnets = [FlaxHunyuanVideo15ResnetBlock(in_channels, in_channels, rngs=rngs)]
        attentions = []
        
        for _ in range(num_layers):
            if add_attention:
                attentions.append(FlaxHunyuanVideo15AttnBlock(in_channels, rngs=rngs))
            resnets.append(FlaxHunyuanVideo15ResnetBlock(in_channels, in_channels, rngs=rngs))
        
        self.resnets = nnx.List(resnets)
        # 只有在 add_attention=True 时才创建 attentions 列表
        if add_attention and len(attentions) > 0:
            self.attentions = nnx.List(attentions)
        else:
            self.attentions = None
    
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        feat_cache: Optional[list] = None,
        feat_idx: Optional[list] = None,
    ) -> jnp.ndarray:
        if feat_cache is not None and feat_idx is not None:
            hidden_states = self.resnets[0](hidden_states, feat_cache=feat_cache, feat_idx=feat_idx)
        else:
            hidden_states = self.resnets[0](hidden_states)
        
        resnets_rest = list(self.resnets)[1:]
        
        for i, resnet in enumerate(resnets_rest):
            # 只有在 add_attention=True 且有 attentions 时才应用 attention
            if self.attentions is not None and i < len(self.attentions):
                hidden_states = self.attentions[i](hidden_states)
            
            if feat_cache is not None and feat_idx is not None:
                hidden_states = resnet(hidden_states, feat_cache=feat_cache, feat_idx=feat_idx)
            else:
                hidden_states = resnet(hidden_states)
        
        return hidden_states


class FlaxHunyuanVideo15DownBlock3D(nnx.Module):
    """Downsampling block for HunyuanVideo15 encoder."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        downsample_out_channels: Optional[int] = None,
        add_temporal_downsample: bool = True,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        
        resnets = []
        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels
            resnets.append(
                FlaxHunyuanVideo15ResnetBlock(input_channels, out_channels, rngs=rngs)
            )
        self.resnets = nnx.List(resnets)
        
        if downsample_out_channels is not None:
            self.downsamplers = nnx.List([
                FlaxHunyuanVideo15Downsample(
                    out_channels,
                    downsample_out_channels,
                    add_temporal_downsample=add_temporal_downsample,
                    rngs=rngs,
                )
            ])
        else:
            self.downsamplers = None
    
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        feat_cache: Optional[list] = None,
        feat_idx: Optional[list] = None,
    ) -> jnp.ndarray:
        for resnet in self.resnets:
            if feat_cache is not None and feat_idx is not None:
                hidden_states = resnet(hidden_states, feat_cache=feat_cache, feat_idx=feat_idx)
            else:
                hidden_states = resnet(hidden_states)
        
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                if feat_cache is not None and feat_idx is not None:
                    hidden_states = downsampler(hidden_states, feat_cache=feat_cache, feat_idx=feat_idx)
                else:
                    hidden_states = downsampler(hidden_states)
        
        return hidden_states


class FlaxHunyuanVideo15UpBlock3D(nnx.Module):
    """Upsampling block for HunyuanVideo15 decoder."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        upsample_out_channels: Optional[int] = None,
        add_temporal_upsample: bool = True,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        
        resnets = []
        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels
            resnets.append(
                FlaxHunyuanVideo15ResnetBlock(input_channels, out_channels, rngs=rngs)
            )
        self.resnets = nnx.List(resnets)
        
        if upsample_out_channels is not None:
            self.upsamplers = nnx.List([
                FlaxHunyuanVideo15Upsample(
                    out_channels,
                    upsample_out_channels,
                    add_temporal_upsample=add_temporal_upsample,
                    rngs=rngs,
                )
            ])
        else:
            self.upsamplers = None
    
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        feat_cache: Optional[list] = None,
        feat_idx: Optional[list] = None,
        is_first_frame: Optional[bool] = None,
    ) -> jnp.ndarray:
        for resnet in self.resnets:
            if feat_cache is not None and feat_idx is not None:
                hidden_states = resnet(hidden_states, feat_cache=feat_cache, feat_idx=feat_idx)
            else:
                hidden_states = resnet(hidden_states)
        
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                if feat_cache is not None and feat_idx is not None:
                    hidden_states = upsampler(hidden_states, feat_cache=feat_cache, feat_idx=feat_idx, is_first_frame=is_first_frame)
                else:
                    hidden_states = upsampler(hidden_states, is_first_frame=is_first_frame)
        
        return hidden_states


class FlaxHunyuanVideo15Encoder3D(nnx.Module):
    """3D VAE encoder for HunyuanVideo15."""
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 64,
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 1024, 1024),
        layers_per_block: int = 2,
        temporal_compression_ratio: int = 4,
        spatial_compression_ratio: int = 16,
        downsample_match_channel: bool = True,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.group_size = block_out_channels[-1] // out_channels
        
        self.conv_in = FlaxHunyuanVideo15CausalConv3d(in_channels, block_out_channels[0], kernel_size=3, rngs=rngs)
        
        down_blocks = []
        input_channel = block_out_channels[0]
        
        for i in range(len(block_out_channels)):
            add_spatial_downsample = i < np.log2(spatial_compression_ratio)
            output_channel = block_out_channels[i]
            
            if not add_spatial_downsample:
                down_block = FlaxHunyuanVideo15DownBlock3D(
                    num_layers=layers_per_block,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    downsample_out_channels=None,
                    add_temporal_downsample=False,
                    rngs=rngs,
                )
                input_channel = output_channel
            else:
                add_temporal_downsample = i >= np.log2(spatial_compression_ratio // temporal_compression_ratio)
                downsample_out_channels = block_out_channels[i + 1] if downsample_match_channel else output_channel
                down_block = FlaxHunyuanVideo15DownBlock3D(
                    num_layers=layers_per_block,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    downsample_out_channels=downsample_out_channels,
                    add_temporal_downsample=add_temporal_downsample,
                    rngs=rngs,
                )
                input_channel = downsample_out_channels
            
            down_blocks.append(down_block)
        
        self.down_blocks = nnx.List(down_blocks)
        self.mid_block = FlaxHunyuanVideo15MidBlock(in_channels=block_out_channels[-1], rngs=rngs)
        
        self.norm_out = FlaxHunyuanVideo15RMSNorm(block_out_channels[-1], images=False, rngs=rngs)
        self.conv_out = FlaxHunyuanVideo15CausalConv3d(block_out_channels[-1], out_channels, kernel_size=3, rngs=rngs)
    
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        feat_cache: Optional[list] = None,
        feat_idx: Optional[list] = None,
    ) -> jnp.ndarray:
        if feat_cache is not None and feat_idx is not None:
            hidden_states = self.conv_in(hidden_states, feat_cache=feat_cache, feat_idx=feat_idx)
        else:
            hidden_states = self.conv_in(hidden_states)
        
        for down_block in self.down_blocks:
            if feat_cache is not None and feat_idx is not None:
                hidden_states = down_block(hidden_states, feat_cache=feat_cache, feat_idx=feat_idx)
            else:
                hidden_states = down_block(hidden_states)
        
        if feat_cache is not None and feat_idx is not None:
            hidden_states = self.mid_block(hidden_states, feat_cache=feat_cache, feat_idx=feat_idx)
        else:
            hidden_states = self.mid_block(hidden_states)
        
        batch_size, frame, height, width, _ = hidden_states.shape
        short_cut = hidden_states.reshape(batch_size, frame, height, width, -1, self.group_size).mean(axis=-1)
        
        hidden_states = self.norm_out(hidden_states)
        hidden_states = jax.nn.silu(hidden_states)
        
        if feat_cache is not None and feat_idx is not None:
            hidden_states = self.conv_out(hidden_states, feat_cache=feat_cache, feat_idx=feat_idx)
        else:
            hidden_states = self.conv_out(hidden_states)
        
        hidden_states = hidden_states + short_cut
        
        return hidden_states


class FlaxHunyuanVideo15Decoder3D(nnx.Module):
    """3D VAE decoder for HunyuanVideo15."""
    
    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (1024, 1024, 512, 256, 128),
        layers_per_block: int = 2,
        spatial_compression_ratio: int = 16,
        temporal_compression_ratio: int = 4,
        upsample_match_channel: bool = True,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        
        self.layers_per_block = layers_per_block
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.repeat = block_out_channels[0] // in_channels
        
        self.conv_in = FlaxHunyuanVideo15CausalConv3d(in_channels, block_out_channels[0], kernel_size=3, rngs=rngs)
        
        self.mid_block = FlaxHunyuanVideo15MidBlock(in_channels=block_out_channels[0], rngs=rngs)
        
        up_blocks = []
        input_channel = block_out_channels[0]
        
        for i in range(len(block_out_channels)):
            output_channel = block_out_channels[i]
            
            add_spatial_upsample = i < np.log2(spatial_compression_ratio)
            add_temporal_upsample = i < np.log2(temporal_compression_ratio)
            
            if add_spatial_upsample or add_temporal_upsample:
                upsample_out_channels = block_out_channels[i + 1] if upsample_match_channel else output_channel
                up_block = FlaxHunyuanVideo15UpBlock3D(
                    num_layers=layers_per_block + 1,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    upsample_out_channels=upsample_out_channels,
                    add_temporal_upsample=add_temporal_upsample,
                    rngs=rngs,
                )
                input_channel = upsample_out_channels
            else:
                up_block = FlaxHunyuanVideo15UpBlock3D(
                    num_layers=layers_per_block + 1,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    upsample_out_channels=None,
                    add_temporal_upsample=False,
                    rngs=rngs,
                )
                input_channel = output_channel
            
            up_blocks.append(up_block)
        
        self.up_blocks = nnx.List(up_blocks)
        
        self.norm_out = FlaxHunyuanVideo15RMSNorm(block_out_channels[-1], images=False, rngs=rngs)
        self.conv_out = FlaxHunyuanVideo15CausalConv3d(block_out_channels[-1], out_channels, kernel_size=3, rngs=rngs)
    
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        feat_cache: Optional[list] = None,
        feat_idx: Optional[list] = None,
        is_first_frame: Optional[bool] = None,
    ) -> jnp.ndarray:
        """
        Forward pass.
        
        Args:
            hidden_states: Latent tensor (B, T, H, W, C)
            feat_cache: 缓存列表
            feat_idx: 当前索引
            is_first_frame: 是否是第一帧（逐帧解码时使用）
        """
        shortcut = jnp.repeat(hidden_states, self.repeat, axis=-1)
        
        if feat_cache is not None and feat_idx is not None:
            hidden_states = self.conv_in(hidden_states, feat_cache=feat_cache, feat_idx=feat_idx)
        else:
            hidden_states = self.conv_in(hidden_states)
        
        hidden_states = hidden_states + shortcut
        
        if feat_cache is not None and feat_idx is not None:
            hidden_states = self.mid_block(hidden_states, feat_cache=feat_cache, feat_idx=feat_idx)
        else:
            hidden_states = self.mid_block(hidden_states)
        
        for up_block in self.up_blocks:
            if feat_cache is not None and feat_idx is not None:
                hidden_states = up_block(hidden_states, feat_cache=feat_cache, feat_idx=feat_idx, is_first_frame=is_first_frame)
            else:
                hidden_states = up_block(hidden_states, is_first_frame=is_first_frame)
        
        hidden_states = self.norm_out(hidden_states)
        hidden_states = jax.nn.silu(hidden_states)
        
        if feat_cache is not None and feat_idx is not None:
            hidden_states = self.conv_out(hidden_states, feat_cache=feat_cache, feat_idx=feat_idx)
        else:
            hidden_states = self.conv_out(hidden_states)
        
        return hidden_states


class FlaxHunyuanVideo15Cache:
    """Cache manager for frame-by-frame decoding."""
    
    def __init__(self, module):
        self.module = module
        self.clear_cache()
    
    def clear_cache(self):
        self._conv_num = self._count_causal_conv3d(self.module)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
    
    @staticmethod
    def _count_causal_conv3d(module):
        count = 0
        node_types = nnx.graph.iter_graph([module])
        for _, value in node_types:
            if isinstance(value, FlaxHunyuanVideo15CausalConv3d):
                count += 1
        return count


class FlaxAutoencoderKLHunyuanVideo15(nnx.Module):
    """
    A VAE model with KL loss for encoding videos into latents and decoding latent 
    representations into videos. Used for HunyuanVideo-1.5.
    
    This is the JAX/Flax implementation with full feature parity including:
    - Tiling for memory efficiency
    - Frame-by-frame decoding with caching
    """
    
    config_class = FlaxAutoencoderKLHunyuanVideo15Config
    
    def __init__(
        self,
        config: FlaxAutoencoderKLHunyuanVideo15Config,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
    ):
        super().__init__()
        self.config = config
        self.dtype = dtype
        
        self.encoder = FlaxHunyuanVideo15Encoder3D(
            in_channels=config.in_channels,
            out_channels=config.latent_channels * 2,
            block_out_channels=config.block_out_channels,
            layers_per_block=config.layers_per_block,
            temporal_compression_ratio=config.temporal_compression_ratio,
            spatial_compression_ratio=config.spatial_compression_ratio,
            downsample_match_channel=config.downsample_match_channel,
            rngs=rngs,
        )
        
        self.decoder = FlaxHunyuanVideo15Decoder3D(
            in_channels=config.latent_channels,
            out_channels=config.out_channels,
            block_out_channels=tuple(reversed(config.block_out_channels)),
            layers_per_block=config.layers_per_block,
            temporal_compression_ratio=config.temporal_compression_ratio,
            spatial_compression_ratio=config.spatial_compression_ratio,
            upsample_match_channel=config.upsample_match_channel,
            rngs=rngs,
        )
        
        self.spatial_compression_ratio = config.spatial_compression_ratio
        self.temporal_compression_ratio = config.temporal_compression_ratio
        
        self.use_tiling = False
        self.use_slicing = False
        
        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256
        self.tile_latent_min_height = self.tile_sample_min_height // config.spatial_compression_ratio
        self.tile_latent_min_width = self.tile_sample_min_width // config.spatial_compression_ratio
        self.tile_overlap_factor = 0.25
    
    def enable_tiling(
        self,
        tile_sample_min_height: Optional[int] = None,
        tile_sample_min_width: Optional[int] = None,
        tile_latent_min_height: Optional[int] = None,
        tile_latent_min_width: Optional[int] = None,
        tile_overlap_factor: Optional[float] = None,
    ):
        self.use_tiling = True
        self.tile_sample_min_height = tile_sample_min_height or self.tile_sample_min_height
        self.tile_sample_min_width = tile_sample_min_width or self.tile_sample_min_width
        self.tile_latent_min_height = tile_latent_min_height or self.tile_latent_min_height
        self.tile_latent_min_width = tile_latent_min_width or self.tile_latent_min_width
        self.tile_overlap_factor = tile_overlap_factor or self.tile_overlap_factor
    
    def disable_tiling(self):
        self.use_tiling = False
    
    def enable_slicing(self):
        self.use_slicing = True
    
    def disable_slicing(self):
        self.use_slicing = False
    
    @staticmethod
    def blend_v(a: jnp.ndarray, b: jnp.ndarray, blend_extent: int) -> jnp.ndarray:
        blend_extent = min(a.shape[2], b.shape[2], blend_extent)
        for y in range(blend_extent):
            alpha = y / blend_extent
            b = b.at[:, :, y, :, :].set(
                a[:, :, -blend_extent + y, :, :] * (1 - alpha) + b[:, :, y, :, :] * alpha
            )
        return b
    
    @staticmethod
    def blend_h(a: jnp.ndarray, b: jnp.ndarray, blend_extent: int) -> jnp.ndarray:
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for x in range(blend_extent):
            alpha = x / blend_extent
            b = b.at[:, :, :, x, :].set(
                a[:, :, :, -blend_extent + x, :] * (1 - alpha) + b[:, :, :, x, :] * alpha
            )
        return b
    
    @staticmethod
    def blend_t(a: jnp.ndarray, b: jnp.ndarray, blend_extent: int) -> jnp.ndarray:
        blend_extent = min(a.shape[1], b.shape[1], blend_extent)
        for t in range(blend_extent):
            alpha = t / blend_extent
            b = b.at[:, t, :, :, :].set(
                a[:, -blend_extent + t, :, :, :] * (1 - alpha) + b[:, t, :, :, :] * alpha
            )
        return b
    
    def _encode(self, x: jnp.ndarray) -> jnp.ndarray:
        _, _, height, width, _ = x.shape
        
        if self.use_tiling and (width > self.tile_sample_min_width or height > self.tile_sample_min_height):
            return self.tiled_encode(x)
        
        return self.encoder(x)
    
    def encode(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        h = self._encode(x)
        mean, logvar = jnp.split(h, 2, axis=-1)
        return mean, logvar
    
    def _decode(self, z: jnp.ndarray) -> jnp.ndarray:
        """
        Decode latents to video using frame-by-frame processing.
        
        HunyuanVideo-1.5 使用 DCAE-style 时间上采样：
        - 第一帧：只空间上采样，1 → 1 帧
        - 后续帧：时空上采样，1 → 2 帧
        - 经过 2 次上采样：L latent 帧 → 4L-3 video 帧
        
        例如：16 latent 帧 → 61 video 帧
        
        逐帧处理时使用 is_first_frame 参数来正确选择上采样分支。
        """
        _, num_frames, height, width, _ = z.shape
        
        if self.use_tiling and (width > self.tile_latent_min_width or height > self.tile_latent_min_height):
            return self.tiled_decode(z)
        
        feat_cache_manager = FlaxHunyuanVideo15Cache(self.decoder)
        
        decoded_frames_list = []
        
        try:
            for i in range(num_frames):
                feat_cache_manager._conv_idx = [0]
                z_frame = z[:, i:i+1, :, :, :]
                
                # 第一帧使用特殊分支，后续帧使用时间上采样
                is_first_frame = (i == 0)
                
                decoded_frame = self.decoder(
                    z_frame,
                    feat_cache=feat_cache_manager._feat_map,
                    feat_idx=feat_cache_manager._conv_idx,
                    is_first_frame=is_first_frame,
                )
                
                decoded_frames_list.append(decoded_frame)
            
            decoded = jnp.concatenate(decoded_frames_list, axis=1)
            decoded = jnp.clip(decoded, -1.0, 1.0)
            
            return decoded
        finally:
            for i in range(len(feat_cache_manager._feat_map)):
                feat_cache_manager._feat_map[i] = None
            del feat_cache_manager
    
    def decode(self, z: jnp.ndarray) -> jnp.ndarray:
        return self._decode(z)
    
    def tiled_encode(self, x: jnp.ndarray) -> jnp.ndarray:
        _, _, height, width, _ = x.shape
        
        overlap_height = int(self.tile_sample_min_height * (1 - self.tile_overlap_factor))
        overlap_width = int(self.tile_sample_min_width * (1 - self.tile_overlap_factor))
        blend_height = int(self.tile_latent_min_height * self.tile_overlap_factor)
        blend_width = int(self.tile_latent_min_width * self.tile_overlap_factor)
        row_limit_height = self.tile_latent_min_height - blend_height
        row_limit_width = self.tile_latent_min_width - blend_width
        
        rows = []
        for i in range(0, height, overlap_height):
            row = []
            for j in range(0, width, overlap_width):
                tile = x[:, :, i:i + self.tile_sample_min_height, j:j + self.tile_sample_min_width, :]
                tile = self.encoder(tile)
                row.append(tile)
            rows.append(row)
        
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_width)
                result_row.append(tile[:, :, :row_limit_height, :row_limit_width, :])
            result_rows.append(jnp.concatenate(result_row, axis=3))
        
        moments = jnp.concatenate(result_rows, axis=2)
        return moments
    
    def tiled_decode(self, z: jnp.ndarray) -> jnp.ndarray:
        _, _, height, width, _ = z.shape
        
        overlap_height = int(self.tile_latent_min_height * (1 - self.tile_overlap_factor))
        overlap_width = int(self.tile_latent_min_width * (1 - self.tile_overlap_factor))
        blend_height = int(self.tile_sample_min_height * self.tile_overlap_factor)
        blend_width = int(self.tile_sample_min_width * self.tile_overlap_factor)
        row_limit_height = self.tile_sample_min_height - blend_height
        row_limit_width = self.tile_sample_min_width - blend_width
        
        rows = []
        for i in range(0, height, overlap_height):
            row = []
            for j in range(0, width, overlap_width):
                tile = z[:, :, i:i + self.tile_latent_min_height, j:j + self.tile_latent_min_width, :]
                decoded = self.decoder(tile)
                row.append(decoded)
            rows.append(row)
        
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_width)
                result_row.append(tile[:, :, :row_limit_height, :row_limit_width, :])
            result_rows.append(jnp.concatenate(result_row, axis=3))
        
        dec = jnp.concatenate(result_rows, axis=2)
        return dec
    
    def __call__(
        self,
        x: jnp.ndarray,
        sample_posterior: bool = False,
        rng: Optional[jax.random.PRNGKey] = None,
    ) -> jnp.ndarray:
        mean, logvar = self.encode(x)
        
        if sample_posterior:
            if rng is None:
                raise ValueError("rng must be provided when sample_posterior=True")
            std = jnp.exp(0.5 * logvar)
            z = mean + std * jax.random.normal(rng, mean.shape)
        else:
            z = mean
        
        dec = self.decode(z)
        return dec
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        subfolder: Optional[str] = "vae",
        dtype: jnp.dtype = jnp.float32,
    ):
        """
        Load pre-trained HunyuanVideo15 VAE weights from HuggingFace.
        
        Args:
            pretrained_model_name_or_path: Model ID
            subfolder: Subfolder containing VAE weights
            dtype: Target dtype for weights
            
        Returns:
            FlaxAutoencoderKLHunyuanVideo15: Initialized model with loaded weights
        """
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open
        import re
        import json
        
        print(f"[1/4] Loading config: {pretrained_model_name_or_path}")
        
        config_path = hf_hub_download(
            pretrained_model_name_or_path,
            subfolder=subfolder,
            filename="config.json"
        )
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        config = cls.config_class.from_dict(config_dict)
        
        print(f"[2/4] Downloading PyTorch weights...")
        
        ckpt_path = hf_hub_download(
            pretrained_model_name_or_path,
            subfolder=subfolder,
            filename="diffusion_pytorch_model.safetensors"
        )
        
        print(f"[3/4] Converting weights to JAX format...")
        
        pytorch_weights = {}
        with safe_open(ckpt_path, framework="np") as f:
            for key in f.keys():
                pytorch_weights[key] = f.get_tensor(key)
        
        print(f"  Loaded {len(pytorch_weights)} PyTorch weight tensors")
        
        jax_weights = {}
        
        for pt_key, pt_tensor in pytorch_weights.items():
            if pt_key.startswith("_orig_mod."):
                pt_key = pt_key[len("_orig_mod."):]
            
            jax_key = pt_key
            jax_tensor = pt_tensor
            
            needs_conv = False
            conv_patterns = ['.conv_in.', '.conv_out.', '.conv1.', '.conv2.',
                           '.conv_shortcut.', '.to_q.', '.to_k.', '.to_v.', '.proj_out.',
                           '.downsamplers.', '.upsamplers.']
            if any(pattern in jax_key for pattern in conv_patterns):
                if not (jax_key.endswith('.conv.weight') or jax_key.endswith('.conv.bias') or
                       jax_key.endswith('.conv.kernel')):
                    if jax_key.endswith('.weight') or jax_key.endswith('.bias') or jax_key.endswith('.kernel'):
                        needs_conv = True
            
            if needs_conv:
                parts = jax_key.rsplit('.', 1)
                jax_key = f"{parts[0]}.conv.{parts[1]}"
            
            if "conv" in jax_key and "weight" in jax_key:
                jax_key = jax_key.replace(".weight", ".kernel")
                
                if len(jax_tensor.shape) == 5:
                    jax_tensor = jax_tensor.transpose(2, 3, 4, 1, 0)
                elif len(jax_tensor.shape) == 4:
                    jax_tensor = jax_tensor.transpose(2, 3, 1, 0)
            
            if ".weight" in jax_key and ("norm" in jax_key or "gamma" in jax_key):
                jax_key = jax_key.replace(".weight", ".gamma")
            
            jax_weights[jax_key] = jnp.array(jax_tensor, dtype=dtype)
        
        print(f"  Converted {len(jax_weights)} weights to JAX format")
        
        print(f"[4/4] Initializing model and loading weights...")
        
        key = jax.random.key(0)
        rngs = nnx.Rngs(key)
        model = cls(config=config, rngs=rngs, dtype=dtype)
        
        from flax.traverse_util import unflatten_dict
        nested_weights = unflatten_dict(jax_weights, sep=".")
        
        graphdef, _ = nnx.split(model)
        model = nnx.merge(graphdef, nested_weights)
        
        print(f"Model loaded successfully!")
        print(f"  Config: {config}")
        
        return model