# Copyright 2025 The CogVideoX team, Tsinghua University & ZhipuAI and The HuggingFace Team.
# All rights reserved.
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
Flax/JAX implementation of CogVideoX VAE with full feature parity to PyTorch version.

This implementation includes:
- CausalConv3d with conv_cache support
- Tiling for memory efficiency
- Frame batch processing
- Spatial normalization
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from flax import nnx

from ...configuration_utils import ConfigMixin


@dataclass
class FlaxAutoencoderKLCogVideoXConfig:
    """
    Configuration class for FlaxAutoencoderKLCogVideoX.
    """
    config_name: str = "config.json"
    
    in_channels: int = 3
    out_channels: int = 3
    down_block_types: Tuple[str, ...] = (
        "CogVideoXDownBlock3D",
        "CogVideoXDownBlock3D",
        "CogVideoXDownBlock3D",
        "CogVideoXDownBlock3D",
    )
    up_block_types: Tuple[str, ...] = (
        "CogVideoXUpBlock3D",
        "CogVideoXUpBlock3D",
        "CogVideoXUpBlock3D",
        "CogVideoXUpBlock3D",
    )
    block_out_channels: Tuple[int, ...] = (128, 256, 256, 512)
    latent_channels: int = 16
    layers_per_block: int = 3
    act_fn: str = "silu"
    norm_eps: float = 1e-6
    norm_num_groups: int = 32
    temporal_compression_ratio: float = 4
    sample_height: int = 480
    sample_width: int = 720
    scaling_factor: float = 1.15258426
    shift_factor: Optional[float] = None
    latents_mean: Optional[Tuple[float]] = None
    latents_std: Optional[Tuple[float]] = None
    force_upcast: bool = True
    use_quant_conv: bool = False
    use_post_quant_conv: bool = False
    pad_mode: str = "first"
    
    @classmethod
    def from_dict(cls, config_dict: Dict):
        """从字典创建配置"""
        # 过滤掉不在 dataclass 字段中的键
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(cls)}
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}
        return cls(**filtered_dict)
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str, ...] = (
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
        ),
        up_block_types: Tuple[str, ...] = (
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
        ),
        block_out_channels: Tuple[int, ...] = (128, 256, 256, 512),
        latent_channels: int = 16,
        layers_per_block: int = 3,
        act_fn: str = "silu",
        norm_eps: float = 1e-6,
        norm_num_groups: int = 32,
        temporal_compression_ratio: float = 4,
        sample_height: int = 480,
        sample_width: int = 720,
        scaling_factor: float = 1.15258426,
        shift_factor: Optional[float] = None,
        latents_mean: Optional[Tuple[float]] = None,
        latents_std: Optional[Tuple[float]] = None,
        force_upcast: bool = True,
        use_quant_conv: bool = False,
        use_post_quant_conv: bool = False,
        pad_mode: str = "first",
        **kwargs,
    ):
        # Dataclass 会自动处理，不需要手动设置
        pass
    
    def OLD__init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str, ...] = (
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
        ),
        up_block_types: Tuple[str, ...] = (
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
        ),
        block_out_channels: Tuple[int, ...] = (128, 256, 256, 512),
        latent_channels: int = 16,
        layers_per_block: int = 3,
        act_fn: str = "silu",
        norm_eps: float = 1e-6,
        norm_num_groups: int = 32,
        temporal_compression_ratio: float = 4,
        sample_height: int = 480,
        sample_width: int = 720,
        scaling_factor: float = 1.15258426,
        shift_factor: Optional[float] = None,
        latents_mean: Optional[Tuple[float]] = None,
        latents_std: Optional[Tuple[float]] = None,
        force_upcast: bool = True,
        use_quant_conv: bool = False,
        use_post_quant_conv: bool = False,
        pad_mode: str = "first",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.down_block_types = down_block_types
        self.up_block_types = up_block_types
        self.block_out_channels = block_out_channels
        self.latent_channels = latent_channels
        self.layers_per_block = layers_per_block
        self.act_fn = act_fn
        self.norm_eps = norm_eps
        self.norm_num_groups = norm_num_groups
        self.temporal_compression_ratio = temporal_compression_ratio
        self.sample_height = sample_height
        self.sample_width = sample_width
        self.scaling_factor = scaling_factor
        self.shift_factor = shift_factor
        self.latents_mean = latents_mean
        self.latents_std = latents_std
        self.force_upcast = force_upcast
        self.use_quant_conv = use_quant_conv
        self.use_post_quant_conv = use_post_quant_conv
        # 这些赋值由 dataclass 自动处理
        pass


class FlaxConv3d(nnx.Module):
    """Basic 3D convolution wrapper for Flax NNX."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]] = (3, 3, 3),
        stride: Union[int, Tuple[int, int, int]] = (1, 1, 1),
        padding: Union[int, Tuple[int, int, int], str] = 1,
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
        padding: Union[int, str] = 1,
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


class FlaxCogVideoXCausalConv3d(nnx.Module):
    """
    A 3D causal convolution layer with feat_cache support for CogVideoX.
    
    这是支持逐帧解码的 CausalConv3d 实现，参考 WanCausalConv3d 的设计。
    主要特性：
    - 时间维度因果填充
    - 通过 feat_cache/feat_idx 支持逐帧处理
    - 支持 'constant' 和 'replicate' 填充模式
    
    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        kernel_size (int or Tuple[int, int, int]): 卷积核大小
        stride (int): 步长（仅应用于时间维度）
        dilation (int): 膨胀率
        pad_mode (str): 填充模式，'constant' 或 'replicate'
        rngs (nnx.Rngs): 随机数生成器
    """
    
    # 类似 WAN，定义缓存需要的帧数
    CACHE_T = 2
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: int = 1,
        dilation: int = 1,
        pad_mode: str = "constant",
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        
        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size
        
        # Calculate padding
        self.time_pad = time_kernel_size - 1
        self.height_pad = (height_kernel_size - 1) // 2
        self.width_pad = (width_kernel_size - 1) // 2
        
        self.pad_mode = pad_mode
        self.time_kernel_size = time_kernel_size
        self.temporal_dim = 1  # In JAX NTHWC format, time is dim 1
        
        # Padding for constant mode (spatial only, time handled separately)
        const_padding_conv3d = (0, self.height_pad, self.width_pad)
        
        # Create the underlying convolution
        stride_tuple = (stride, 1, 1) if isinstance(stride, int) else stride
        dilation_tuple = (dilation, 1, 1)
        
        self.conv = FlaxConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride_tuple,
            padding=0 if self.pad_mode == "replicate" else const_padding_conv3d,
            rngs=rngs,
        )
    
    def __call__(
        self,
        inputs: jnp.ndarray,
        conv_cache: Optional[jnp.ndarray] = None,
        feat_cache: Optional[list] = None,
        feat_idx: Optional[list] = None,
    ):
        """
        前向传播，支持两种缓存模式：
        1. conv_cache: 旧的缓存模式（为了兼容性保留）
        2. feat_cache/feat_idx: 新的逐帧解码缓存模式（用于解决 OOM）
        
        Args:
            inputs: 输入张量 (B, T, H, W, C)
            conv_cache: 旧模式的缓存（已弃用）
            feat_cache: 缓存列表（新模式）
            feat_idx: 当前索引列表（新模式）
            
        Returns:
            output: 卷积输出
            new_cache: 更新的缓存（仅在旧模式下使用）
        """
        # 新模式：使用 feat_cache/feat_idx 进行逐帧解码
        if feat_cache is not None and feat_idx is not None:
            return self._call_with_feat_cache(inputs, feat_cache, feat_idx)
        
        # 旧模式：使用 conv_cache（保持向后兼容）
        return self._call_with_conv_cache(inputs, conv_cache)
    
    def _call_with_feat_cache(
        self,
        inputs: jnp.ndarray,
        feat_cache: list,
        feat_idx: list,
    ):
        """
        使用 feat_cache 的新缓存模式（参考 WanCausalConv3d）。
        
        Args:
            inputs: 输入张量 (B, T, H, W, C)，通常 T=1（逐帧处理）
            feat_cache: 缓存列表
            feat_idx: 当前索引列表 [idx]
            
        Returns:
            output: 卷积输出
        """
        idx = feat_idx[0]
        
        # 处理时间填充
        if self.pad_mode == "replicate":
            # Replicate 模式：直接填充
            pad_width = [
                (0, 0),  # batch
                (self.time_pad, 0),  # time (only pad before, causal)
                (self.height_pad, self.height_pad),  # height
                (self.width_pad, self.width_pad),  # width
                (0, 0),  # channels
            ]
            x = jnp.pad(inputs, pad_width, mode='edge')
        else:
            # Constant 模式：使用缓存（参考 WAN 的实现）
            if self.time_kernel_size > 1:
                padding_needed = self.time_kernel_size - 1  # 需要的 padding 帧数
                
                # 如果缓存中有数据，使用它
                if feat_cache[idx] is not None:
                    cache_len = feat_cache[idx].shape[1]
                    # 拼接缓存和当前输入
                    x = jnp.concatenate([feat_cache[idx], inputs], axis=1)
                    
                    # 调整还需要的 padding
                    padding_needed -= cache_len
                    if padding_needed > 0:
                        # 缓存不够，补充额外的 padding
                        extra_padding = jnp.tile(x[:, :1, :, :, :], (1, padding_needed, 1, 1, 1))
                        x = jnp.concatenate([extra_padding, x], axis=1)
                    elif padding_needed < 0:
                        # 缓存太多，裁剪
                        x = x[:, -padding_needed:, ...]
                else:
                    # 第一次调用：重复第一帧作为填充
                    padding_frames = jnp.tile(
                        inputs[:, :1, :, :, :],
                        (1, padding_needed, 1, 1, 1)
                    )
                    x = jnp.concatenate([padding_frames, inputs], axis=1)
                
                # 更新缓存：保存拼接后 x 的最后 CACHE_T 帧（参考 WAN 第 432-436 行）
                # 如果 inputs 本身不足 2 帧，需要从旧缓存和新输入组合
                if inputs.shape[1] < self.CACHE_T and feat_cache[idx] is not None:
                    # inputs 不足 2 帧：从旧缓存取最后 1 帧 + inputs 的最后 CACHE_T 帧
                    cache_x = jnp.concatenate([
                        jnp.expand_dims(feat_cache[idx][:, -1, :, :, :], axis=1),
                        inputs[:, -self.CACHE_T:, :, :, :]
                    ], axis=1)
                else:
                    # inputs 足够或第一次：直接取 inputs 的最后 CACHE_T 帧
                    cache_x = inputs[:, -self.CACHE_T:, :, :, :]
                
                feat_cache[idx] = cache_x
            else:
                x = inputs
        
        # 执行卷积
        output = self.conv(x)
        
        # 索引递增
        feat_idx[0] += 1
        
        # 旧 API 兼容性：返回 (output, None)
        return output, None
    
    def _call_with_conv_cache(self, inputs: jnp.ndarray, conv_cache: Optional[jnp.ndarray]):
        """
        旧的 conv_cache 模式（保持向后兼容）。
        
        Args:
            inputs: 输入张量 (B, T, H, W, C)
            conv_cache: 上一次的缓存
            
        Returns:
            output: 卷积输出
            new_cache: 更新的缓存
        """
        # Apply causal padding
        if self.pad_mode == "replicate":
            # Replicate padding mode: pad all dimensions including time
            pad_width = [
                (0, 0),  # batch
                (self.time_pad, 0),  # time (only pad before, causal)
                (self.height_pad, self.height_pad),  # height
                (self.width_pad, self.width_pad),  # width
                (0, 0),  # channels
            ]
            inputs = jnp.pad(inputs, pad_width, mode='edge')
            conv_cache = None
        else:
            # Constant padding mode: use conv_cache for time dimension
            if self.time_kernel_size > 1:
                if conv_cache is not None:
                    # Use provided cache
                    cached_inputs = conv_cache
                else:
                    # First call: repeat first frame
                    cached_inputs = jnp.tile(
                        inputs[:, :1, :, :, :],
                        (1, self.time_kernel_size - 1, 1, 1, 1)
                    )
                # Concatenate cache with current input
                inputs = jnp.concatenate([cached_inputs, inputs], axis=1)
        
        # Apply convolution
        output = self.conv(inputs)
        
        # Update cache
        if self.pad_mode == "replicate":
            new_cache = None
        else:
            # Save last (time_kernel_size - 1) frames for next iteration
            new_cache = inputs[:, -(self.time_kernel_size - 1):, :, :, :]
        
        return output, new_cache


class FlaxGroupNorm(nnx.Module):
    """
    Group Normalization matching PyTorch's GroupNorm exactly.
    
    Based on diffusers_tpu implementation but adapted for channel-last format.
    Internally converts to channel-first for computation to match PyTorch precisely.
    """
    
    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        epsilon: float = 1e-6,
        rngs: nnx.Rngs = None,
    ):
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.epsilon = epsilon
        
        # Create parameters matching PyTorch's GroupNorm
        self.scale = nnx.Param(jnp.ones((num_channels,)))
        self.bias = nnx.Param(jnp.zeros((num_channels,)))
    
    def __call__(self, x):
        """
        Apply group normalization matching PyTorch's implementation.
        
        PyTorch GroupNorm works on channel-first format (N, C, *).
        We convert channel-last to channel-first, apply GroupNorm, then convert back.
        
        Args:
            x: Input of shape (B, T, H, W, C) or (B, H, W, C) [channel-last]
            
        Returns:
            Normalized output with same shape as input [channel-last]
        """
        if len(x.shape) == 5:
            # 5D: (B, T, H, W, C) -> (B, C, T, H, W)
            B, T, H, W, C = x.shape
            assert C == self.num_channels
            assert C % self.num_groups == 0
            
            # Convert to channel-first: (B, T, H, W, C) -> (B, C, T, H, W)
            x_cf = x.transpose(0, 4, 1, 2, 3)
            
            # Now apply GroupNorm in channel-first format (matching diffusers_tpu)
            # Reshape to group structure: (B, num_groups, C//num_groups, T, H, W)
            x_grouped = x_cf.reshape(B, self.num_groups, C // self.num_groups, T, H, W)
            
            # Compute mean and variance over (C//num_groups, T, H, W)
            # This matches PyTorch GroupNorm exactly
            mean = jnp.mean(x_grouped, axis=(2, 3, 4, 5), keepdims=True)
            var = jnp.var(x_grouped, axis=(2, 3, 4, 5), keepdims=True)
            
            # Normalize
            x_norm = (x_grouped - mean) / jnp.sqrt(var + self.epsilon)
            
            # Reshape back: (B, C, T, H, W)
            x_norm = x_norm.reshape(B, C, T, H, W)
            
            # Apply affine transformation (still in channel-first)
            x_out = x_norm * self.scale.value.reshape(1, C, 1, 1, 1) + self.bias.value.reshape(1, C, 1, 1, 1)
            
            # Convert back to channel-last: (B, C, T, H, W) -> (B, T, H, W, C)
            x_out = x_out.transpose(0, 2, 3, 4, 1)
            
        else:
            # 4D: (B, H, W, C) -> (B, C, H, W)
            B, H, W, C = x.shape
            assert C == self.num_channels
            assert C % self.num_groups == 0
            
            # Convert to channel-first: (B, H, W, C) -> (B, C, H, W)
            x_cf = x.transpose(0, 3, 1, 2)
            
            # Reshape to group structure: (B, num_groups, C//num_groups, H, W)
            x_grouped = x_cf.reshape(B, self.num_groups, C // self.num_groups, H, W)
            
            # Compute statistics over (C//num_groups, H, W)
            mean = jnp.mean(x_grouped, axis=(2, 3, 4), keepdims=True)
            var = jnp.var(x_grouped, axis=(2, 3, 4), keepdims=True)
            
            # Normalize
            x_norm = (x_grouped - mean) / jnp.sqrt(var + self.epsilon)
            
            # Reshape back: (B, C, H, W)
            x_norm = x_norm.reshape(B, C, H, W)
            
            # Apply affine
            x_out = x_norm * self.scale.value.reshape(1, C, 1, 1) + self.bias.value.reshape(1, C, 1, 1)
            
            # Convert back to channel-last: (B, C, H, W) -> (B, H, W, C)
            x_out = x_out.transpose(0, 2, 3, 1)
        
        return x_out


class FlaxCogVideoXSpatialNorm3D(nnx.Module):
    """
    Spatially conditioned normalization for CogVideoX decoder.
    
    This matches the PyTorch CogVideoXSpatialNorm3D implementation.
    """
    
    def __init__(
        self,
        f_channels: int,
        zq_channels: int,
        groups: int = 32,
        rngs: nnx.Rngs = None,
    ):
        self.norm_layer = FlaxGroupNorm(
            num_groups=groups,
            num_channels=f_channels,
            epsilon=1e-6,
            rngs=rngs
        )
        # Using CausalConv3d for the conditioning convolutions
        self.conv_y = FlaxCogVideoXCausalConv3d(
            zq_channels, f_channels, kernel_size=1, stride=1, pad_mode="constant", rngs=rngs
        )
        self.conv_b = FlaxCogVideoXCausalConv3d(
            zq_channels, f_channels, kernel_size=1, stride=1, pad_mode="constant", rngs=rngs
        )
    
    def __call__(
        self,
        f: jnp.ndarray,
        zq: jnp.ndarray,
        conv_cache: Optional[Dict[str, jnp.ndarray]] = None,
        feat_cache: Optional[list] = None,
        feat_idx: Optional[list] = None,
    ):
        """
        Apply spatial normalization with conditioning.
        
        支持两种模式：
        1. conv_cache: 旧的缓存模式（保持兼容性）
        2. feat_cache/feat_idx: 新的逐帧解码模式
        
        Args:
            f: Feature map of shape (B, T, H, W, C)
            zq: Conditioning latent of shape (B, T', H', W', C')
            conv_cache: 旧模式的缓存字典
            feat_cache: 新模式的缓存列表
            feat_idx: 新模式的索引列表
            
        Returns:
            Normalized and conditioned features
            Updated cache (conv_cache dict 或 None)
        """
        # 新模式：使用 feat_cache/feat_idx
        if feat_cache is not None and feat_idx is not None:
            return self._call_with_feat_cache(f, zq, feat_cache, feat_idx)
        
        # 旧模式：使用 conv_cache
        return self._call_with_conv_cache(f, zq, conv_cache)
    
    def _call_with_feat_cache(
        self,
        f: jnp.ndarray,
        zq: jnp.ndarray,
        feat_cache: list,
        feat_idx: list,
    ):
        """新缓存模式的实现"""
        # Handle odd frame counts specially (matching PyTorch implementation)
        B, T, H, W, C = f.shape
        if T > 1 and T % 2 == 1:
            # Split first frame and rest
            f_first = f[:, :1, :, :, :]
            f_rest = f[:, 1:, :, :, :]
            z_first = zq[:, :1, :, :, :]
            z_rest = zq[:, 1:, :, :, :]
            
            # Resize separately
            z_first = jax.image.resize(z_first, (B, 1, H, W, zq.shape[-1]), method='nearest')
            z_rest = jax.image.resize(z_rest, (B, T-1, H, W, zq.shape[-1]), method='nearest')
            
            # Concatenate back
            zq = jnp.concatenate([z_first, z_rest], axis=1)
        else:
            # Regular resize
            zq = jax.image.resize(zq, (B, T, H, W, zq.shape[-1]), method='nearest')
        
        # Apply conditioning convolutions with feat_cache
        conv_y, _ = self.conv_y(zq, feat_cache=feat_cache, feat_idx=feat_idx)
        conv_b, _ = self.conv_b(zq, feat_cache=feat_cache, feat_idx=feat_idx)
        
        # Normalize and condition
        norm_f = self.norm_layer(f)
        new_f = norm_f * conv_y + conv_b
        
        return new_f, None
    
    def _call_with_conv_cache(
        self,
        f: jnp.ndarray,
        zq: jnp.ndarray,
        conv_cache: Optional[Dict[str, jnp.ndarray]],
    ):
        """旧缓存模式的实现"""
        new_conv_cache = {}
        conv_cache = conv_cache or {}
        
        # Handle odd frame counts specially (matching PyTorch implementation)
        B, T, H, W, C = f.shape
        if T > 1 and T % 2 == 1:
            # Split first frame and rest
            f_first = f[:, :1, :, :, :]
            f_rest = f[:, 1:, :, :, :]
            z_first = zq[:, :1, :, :, :]
            z_rest = zq[:, 1:, :, :, :]
            
            # Resize separately
            z_first = jax.image.resize(z_first, (B, 1, H, W, zq.shape[-1]), method='nearest')
            z_rest = jax.image.resize(z_rest, (B, T-1, H, W, zq.shape[-1]), method='nearest')
            
            # Concatenate back
            zq = jnp.concatenate([z_first, z_rest], axis=1)
        else:
            # Regular resize
            zq = jax.image.resize(zq, (B, T, H, W, zq.shape[-1]), method='nearest')
        
        # Apply conditioning convolutions
        conv_y, new_conv_cache["conv_y"] = self.conv_y(zq, conv_cache=conv_cache.get("conv_y"))
        conv_b, new_conv_cache["conv_b"] = self.conv_b(zq, conv_cache=conv_cache.get("conv_b"))
        
        # Normalize and condition
        norm_f = self.norm_layer(f)
        new_f = norm_f * conv_y + conv_b
        
        return new_f, new_conv_cache


# Continue in next message due to length...


class FlaxCogVideoXResnetBlock3D(nnx.Module):
    """
    A 3D ResNet block for CogVideoX with optional spatial normalization.
    
    Matches the PyTorch CogVideoXResnetBlock3D implementation.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        temb_channels: int = 512,
        groups: int = 32,
        eps: float = 1e-6,
        non_linearity: str = "swish",
        conv_shortcut: bool = False,
        spatial_norm_dim: Optional[int] = None,
        pad_mode: str = "first",
        rngs: nnx.Rngs = None,
    ):
        out_channels = out_channels or in_channels
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.spatial_norm_dim = spatial_norm_dim
        
        # Normalization layers
        if spatial_norm_dim is None:
            # Encoder: use GroupNorm
            self.norm1 = FlaxGroupNorm(num_groups=groups, num_channels=in_channels, epsilon=eps, rngs=rngs)
            self.norm2 = FlaxGroupNorm(num_groups=groups, num_channels=out_channels, epsilon=eps, rngs=rngs)
        else:
            # Decoder: use SpatialNorm3D
            self.norm1 = FlaxCogVideoXSpatialNorm3D(
                f_channels=in_channels,
                zq_channels=spatial_norm_dim,
                groups=groups,
                rngs=rngs
            )
            self.norm2 = FlaxCogVideoXSpatialNorm3D(
                f_channels=out_channels,
                zq_channels=spatial_norm_dim,
                groups=groups,
                rngs=rngs
            )
        
        # Convolution layers
        self.conv1 = FlaxCogVideoXCausalConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            pad_mode=pad_mode,
            rngs=rngs
        )
        
        # Time embedding projection (if needed)
        if temb_channels > 0:
            self.temb_proj = nnx.Linear(temb_channels, out_channels, rngs=rngs)
        else:
            self.temb_proj = None
        
        # Dropout
        self.dropout_rate = dropout
        
        self.conv2 = FlaxCogVideoXCausalConv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            pad_mode=pad_mode,
            rngs=rngs
        )
        
        # Shortcut connection
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = FlaxCogVideoXCausalConv3d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    pad_mode=pad_mode,
                    rngs=rngs
                )
            else:
                # Use 1x1x1 conv for shortcut
                self.conv_shortcut = FlaxConv3d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    rngs=rngs
                )
        else:
            self.conv_shortcut = None
    
    def __call__(
        self,
        inputs: jnp.ndarray,
        temb: Optional[jnp.ndarray] = None,
        zq: Optional[jnp.ndarray] = None,
        conv_cache: Optional[Dict[str, jnp.ndarray]] = None,
        feat_cache: Optional[list] = None,
        feat_idx: Optional[list] = None,
        deterministic: bool = True,
    ):
        """
        Forward pass of the ResNet block.
        
        支持两种缓存模式：
        1. conv_cache: 旧模式（保持兼容性）
        2. feat_cache/feat_idx: 新模式（逐帧解码）
        
        Args:
            inputs: Input tensor (B, T, H, W, C)
            temb: Time embedding (B, temb_channels)
            zq: Spatial conditioning (for decoder)
            conv_cache: 旧模式的缓存字典
            feat_cache: 新模式的缓存列表
            feat_idx: 新模式的索引列表
            deterministic: Whether to use dropout
            
        Returns:
            Output tensor and updated cache (或 None)
        """
        # 新模式：使用 feat_cache/feat_idx
        if feat_cache is not None and feat_idx is not None:
            return self._call_with_feat_cache(inputs, temb, zq, feat_cache, feat_idx, deterministic)
        
        # 旧模式：使用 conv_cache
        return self._call_with_conv_cache(inputs, temb, zq, conv_cache, deterministic)
    
    def _call_with_feat_cache(
        self,
        inputs: jnp.ndarray,
        temb: Optional[jnp.ndarray],
        zq: Optional[jnp.ndarray],
        feat_cache: list,
        feat_idx: list,
        deterministic: bool,
    ):
        """新缓存模式的实现"""
        hidden_states = inputs
        
        # First norm and conv
        if zq is not None:
            hidden_states, _ = self.norm1(hidden_states, zq, feat_cache=feat_cache, feat_idx=feat_idx)
        else:
            hidden_states = self.norm1(hidden_states)
        
        hidden_states = jax.nn.silu(hidden_states)
        hidden_states, _ = self.conv1(hidden_states, feat_cache=feat_cache, feat_idx=feat_idx)
        
        # Time embedding
        if temb is not None and self.temb_proj is not None:
            temb_proj = self.temb_proj(jax.nn.silu(temb))
            hidden_states = hidden_states + temb_proj[:, None, None, None, :]
        
        # Second norm and conv
        if zq is not None:
            hidden_states, _ = self.norm2(hidden_states, zq, feat_cache=feat_cache, feat_idx=feat_idx)
        else:
            hidden_states = self.norm2(hidden_states)
        
        hidden_states = jax.nn.silu(hidden_states)
        
        # Dropout
        if self.dropout_rate > 0 and not deterministic:
            hidden_states = nnx.Dropout(rate=self.dropout_rate)(hidden_states)
        
        hidden_states, _ = self.conv2(hidden_states, feat_cache=feat_cache, feat_idx=feat_idx)
        
        # Shortcut
        if self.conv_shortcut is not None:
            if self.use_conv_shortcut:
                inputs, _ = self.conv_shortcut(inputs, feat_cache=feat_cache, feat_idx=feat_idx)
            else:
                inputs = self.conv_shortcut(inputs)
        
        # Residual connection
        hidden_states = hidden_states + inputs
        
        return hidden_states, None
    
    def _call_with_conv_cache(
        self,
        inputs: jnp.ndarray,
        temb: Optional[jnp.ndarray],
        zq: Optional[jnp.ndarray],
        conv_cache: Optional[Dict[str, jnp.ndarray]],
        deterministic: bool,
    ):
        """旧缓存模式的实现"""
        new_conv_cache = {}
        conv_cache = conv_cache or {}
        
        hidden_states = inputs
        
        # First norm and conv
        if zq is not None:
            hidden_states, new_conv_cache["norm1"] = self.norm1(
                hidden_states, zq, conv_cache=conv_cache.get("norm1")
            )
        else:
            hidden_states = self.norm1(hidden_states)
        
        hidden_states = jax.nn.silu(hidden_states)
        hidden_states, new_conv_cache["conv1"] = self.conv1(
            hidden_states, conv_cache=conv_cache.get("conv1")
        )
        
        # Time embedding
        if temb is not None and self.temb_proj is not None:
            temb_proj = self.temb_proj(jax.nn.silu(temb))
            hidden_states = hidden_states + temb_proj[:, None, None, None, :]
        
        # Second norm and conv
        if zq is not None:
            hidden_states, new_conv_cache["norm2"] = self.norm2(
                hidden_states, zq, conv_cache=conv_cache.get("norm2")
            )
        else:
            hidden_states = self.norm2(hidden_states)
        
        hidden_states = jax.nn.silu(hidden_states)
        
        # Dropout
        if self.dropout_rate > 0 and not deterministic:
            hidden_states = nnx.Dropout(rate=self.dropout_rate)(hidden_states)
        
        hidden_states, new_conv_cache["conv2"] = self.conv2(
            hidden_states, conv_cache=conv_cache.get("conv2")
        )
        
        # Shortcut
        if self.conv_shortcut is not None:
            if self.use_conv_shortcut:
                inputs, new_conv_cache["conv_shortcut"] = self.conv_shortcut(
                    inputs, conv_cache=conv_cache.get("conv_shortcut")
                )
            else:
                inputs = self.conv_shortcut(inputs)
        
        # Residual connection
        hidden_states = hidden_states + inputs
        
        return hidden_states, new_conv_cache


class FlaxCogVideoXDownBlock3D(nnx.Module):
    """Downsampling block for CogVideoX encoder."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        add_downsample: bool = True,
        downsample_padding: int = 0,
        compress_time: bool = False,
        pad_mode: str = "first",
        rngs: nnx.Rngs = None,
    ):
        # Create ResNet layers
        resnets = []
        for i in range(num_layers):
            in_channel = in_channels if i == 0 else out_channels
            resnet = FlaxCogVideoXResnetBlock3D(
                in_channels=in_channel,
                out_channels=out_channels,
                dropout=dropout,
                temb_channels=temb_channels,
                groups=resnet_groups,
                eps=resnet_eps,
                non_linearity=resnet_act_fn,
                pad_mode=pad_mode,
                rngs=rngs
            )
            resnets.append(resnet)
        self.resnets = nnx.List(resnets)
        
        # Downsampler
        if add_downsample:
            # CogVideoX uses Conv2d for spatial downsampling
            # Note: PyTorch adds manual padding (0,1,0,1) before conv, not in conv itself
            downsampler = FlaxConv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=0,  # No padding in conv - we'll add it manually
                rngs=rngs
            )
            self.downsamplers = nnx.List([downsampler])
            self.compress_time = compress_time
            self.downsample_padding = downsample_padding
        else:
            self.downsamplers = None
            self.compress_time = False
            self.downsample_padding = 0
    
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        temb: Optional[jnp.ndarray] = None,
        zq: Optional[jnp.ndarray] = None,
        conv_cache: Optional[Dict[str, jnp.ndarray]] = None,
        deterministic: bool = True,
    ):
        new_conv_cache = {}
        conv_cache = conv_cache or {}
        
        for i, resnet in enumerate(self.resnets):
            conv_cache_key = f"resnet_{i}"
            hidden_states, new_conv_cache[conv_cache_key] = resnet(
                hidden_states, temb, zq, conv_cache=conv_cache.get(conv_cache_key), deterministic=deterministic
            )
        
        if self.downsamplers is not None:
            # Handle time compression if needed
            if self.compress_time:
                B, T, H, W, C = hidden_states.shape
                # Compress time dimension using avg pooling
                # Match PyTorch's implementation
                hidden_states = hidden_states.reshape(B * H * W, T, C)
                hidden_states = hidden_states.transpose(0, 2, 1)  # (B*H*W, C, T)
                
                if T % 2 == 1:
                    # Handle odd frames: keep first, avg pool rest
                    first_frame = hidden_states[:, :, 0:1]
                    rest_frames = hidden_states[:, :, 1:]
                    if rest_frames.shape[2] > 0:
                        # Simple avg pooling
                        rest_frames = jnp.mean(
                            rest_frames.reshape(B*H*W, C, rest_frames.shape[2]//2, 2),
                            axis=-1
                        )
                    hidden_states = jnp.concatenate([first_frame, rest_frames], axis=2)
                else:
                    # Even frames: regular avg pooling
                    hidden_states = jnp.mean(
                        hidden_states.reshape(B*H*W, C, T//2, 2),
                        axis=-1
                    )
                
                # Reshape back
                T_new = hidden_states.shape[2]
                hidden_states = hidden_states.transpose(0, 2, 1)  # (B*H*W, T_new, C)
                hidden_states = hidden_states.reshape(B, H, W, T_new, C)
                hidden_states = hidden_states.transpose(0, 3, 1, 2, 4)  # (B, T_new, H, W, C)
            
            # Apply 2D spatial downsampling
            for downsampler in self.downsamplers:
                B, T, H, W, C = hidden_states.shape
                
                # Add manual padding (0, 1, 0, 1) matching PyTorch's implementation
                # JAX padding format: ((before_1, after_1), (before_2, after_2), ...)
                pad_width = [
                    (0, 0),  # batch
                    (0, 0),  # time
                    (0, 1),  # height: pad bottom
                    (0, 1),  # width: pad right
                    (0, 0),  # channels
                ]
                hidden_states = jnp.pad(hidden_states, pad_width, mode='constant', constant_values=0)
                
                # Reshape to apply 2D conv: (B, T, H, W, C) → (B*T, H, W, C)
                _, _, H_padded, W_padded, _ = hidden_states.shape
                hidden_states = hidden_states.reshape(B * T, H_padded, W_padded, C)
                hidden_states = downsampler(hidden_states)
                # Reshape back: (B*T, H', W', C) → (B, T, H', W', C)
                _, H_new, W_new, _ = hidden_states.shape
                hidden_states = hidden_states.reshape(B, T, H_new, W_new, C)
        
        return hidden_states, new_conv_cache


class FlaxCogVideoXMidBlock3D(nnx.Module):
    """Middle block for CogVideoX encoder/decoder."""
    
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        spatial_norm_dim: Optional[int] = None,
        pad_mode: str = "first",
        rngs: nnx.Rngs = None,
    ):
        resnets = []
        for i in range(num_layers):
            resnet = FlaxCogVideoXResnetBlock3D(
                in_channels=in_channels,
                out_channels=in_channels,
                dropout=dropout,
                temb_channels=temb_channels,
                groups=resnet_groups,
                eps=resnet_eps,
                spatial_norm_dim=spatial_norm_dim,
                non_linearity=resnet_act_fn,
                pad_mode=pad_mode,
                rngs=rngs
            )
            resnets.append(resnet)
        self.resnets = nnx.List(resnets)
    
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        temb: Optional[jnp.ndarray] = None,
        zq: Optional[jnp.ndarray] = None,
        conv_cache: Optional[Dict[str, jnp.ndarray]] = None,
        feat_cache: Optional[list] = None,
        feat_idx: Optional[list] = None,
        deterministic: bool = True,
    ):
        """
        MidBlock forward pass.
        
        支持两种缓存模式：
        1. conv_cache: 旧模式（保持兼容性）
        2. feat_cache/feat_idx: 新模式（逐帧解码）
        """
        # 新模式：使用 feat_cache/feat_idx
        if feat_cache is not None and feat_idx is not None:
            for resnet in self.resnets:
                hidden_states, _ = resnet(
                    hidden_states, temb, zq,
                    feat_cache=feat_cache, feat_idx=feat_idx,
                    deterministic=deterministic
                )
            return hidden_states, None
        
        # 旧模式：使用 conv_cache
        new_conv_cache = {}
        conv_cache = conv_cache or {}
        
        for i, resnet in enumerate(self.resnets):
            conv_cache_key = f"resnet_{i}"
            hidden_states, new_conv_cache[conv_cache_key] = resnet(
                hidden_states, temb, zq, conv_cache=conv_cache.get(conv_cache_key), deterministic=deterministic
            )
        
        return hidden_states, new_conv_cache


class FlaxCogVideoXUpBlock3D(nnx.Module):
    """Upsampling block for CogVideoX decoder."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        spatial_norm_dim: int = 16,
        add_upsample: bool = True,
        upsample_padding: int = 1,
        compress_time: bool = False,
        pad_mode: str = "first",
        rngs: nnx.Rngs = None,
    ):
        # Create ResNet layers
        resnets = []
        for i in range(num_layers):
            in_channel = in_channels if i == 0 else out_channels
            resnet = FlaxCogVideoXResnetBlock3D(
                in_channels=in_channel,
                out_channels=out_channels,
                dropout=dropout,
                temb_channels=temb_channels,
                groups=resnet_groups,
                eps=resnet_eps,
                non_linearity=resnet_act_fn,
                spatial_norm_dim=spatial_norm_dim,
                pad_mode=pad_mode,
                rngs=rngs
            )
            resnets.append(resnet)
        self.resnets = nnx.List(resnets)
        
        # Upsampler
        if add_upsample:
            # CogVideoX uses Conv2d for spatial upsampling (matches PyTorch)
            # padding=1 by default in upsampler
            upsampler = FlaxConv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=upsample_padding,  # default is 1
                rngs=rngs
            )
            self.upsamplers = nnx.List([upsampler])
            self.compress_time = compress_time
        else:
            self.upsamplers = None
            self.compress_time = False
    
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        temb: Optional[jnp.ndarray] = None,
        zq: Optional[jnp.ndarray] = None,
        conv_cache: Optional[Dict[str, jnp.ndarray]] = None,
        feat_cache: Optional[list] = None,
        feat_idx: Optional[list] = None,
        deterministic: bool = True,
    ):
        # 新模式：使用 feat_cache/feat_idx（逐帧解码）
        if feat_cache is not None and feat_idx is not None:
            print(f"[UpBlock] 输入形状: {hidden_states.shape}, compress_time={self.compress_time}")
            
            for resnet in self.resnets:
                hidden_states, _ = resnet(
                    hidden_states, temb, zq,
                    feat_cache=feat_cache, feat_idx=feat_idx,
                    deterministic=deterministic
                )
            
            if self.upsamplers is not None:
                for upsampler in self.upsamplers:
                    B, T, H, W, C = hidden_states.shape
                    
                    # compress_time：时间 + 空间上采样
                    if self.compress_time:
                        # 逐帧输入 T=1：1 -> 2（时间）+ 2x空间
                        if T == 1:
                            hidden_states = jax.image.resize(hidden_states, (B, 2, H * 2, W * 2, C), method='nearest')
                        elif T > 1 and T % 2 == 1:
                            first_frame = hidden_states[:, 0, :, :, :]
                            rest_frames = hidden_states[:, 1:, :, :, :]
                            first_frame = jax.image.resize(first_frame, (B, H * 2, W * 2, C), method='nearest')
                            first_frame = first_frame[:, None, :, :, :]
                            rest_frames = jax.image.resize(rest_frames, (B, 2 * (T-1), H * 2, W * 2, C), method='nearest')
                            hidden_states = jnp.concatenate([first_frame, rest_frames], axis=1)
                        else:
                            hidden_states = jax.image.resize(hidden_states, (B, T * 2, H * 2, W * 2, C), method='nearest')
                    else:
                        # 非 compress_time：只做空间上采样
                        hidden_states = hidden_states.reshape(B * T, H, W, C)
                        hidden_states = jax.image.resize(hidden_states, (B * T, H * 2, W * 2, C), method='nearest')
                        hidden_states = hidden_states.reshape(B, T, H * 2, W * 2, C)
                    
                    # 应用 2D 卷积到空间维度
                    B, T_new, H_new, W_new, C = hidden_states.shape
                    hidden_states = hidden_states.reshape(B * T_new, H_new, W_new, C)
                    hidden_states = upsampler(hidden_states)
                    _, H_final, W_final, _ = hidden_states.shape
                    hidden_states = hidden_states.reshape(B, T_new, H_final, W_final, C)
            
            return hidden_states, None
        
        # 旧模式：使用 conv_cache
        new_conv_cache = {}
        conv_cache = conv_cache or {}
        
        for i, resnet in enumerate(self.resnets):
            conv_cache_key = f"resnet_{i}"
            hidden_states, new_conv_cache[conv_cache_key] = resnet(
                hidden_states, temb, zq, conv_cache=conv_cache.get(conv_cache_key), deterministic=deterministic
            )
        
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                B, T, H, W, C = hidden_states.shape
                
                # Match PyTorch's CogVideoXUpsample3D behavior
                if self.compress_time:
                    # PyTorch uses F.interpolate which interpolates ALL dimensions
                    # For compress_time, we need 2x upsampling in both time AND space
                    if T > 1 and T % 2 == 1:
                        # Odd frames: split first frame from rest
                        # PyTorch: x_first, x_rest = inputs[:, :, 0], inputs[:, :, 1:]
                        first_frame = hidden_states[:, 0, :, :, :]  # (B, H, W, C)
                        rest_frames = hidden_states[:, 1:, :, :, :]  # (B, T-1, H, W, C)
                        
                        # Upsample first frame spatially only (no time dimension)
                        # PyTorch: x_first = F.interpolate(x_first, scale_factor=2.0)
                        first_frame = jax.image.resize(
                            first_frame,
                            (B, H * 2, W * 2, C),
                            method='nearest'
                        )
                        first_frame = first_frame[:, None, :, :, :]  # Add time dim back
                        
                        # Upsample rest frames (both time and space)
                        # PyTorch: x_rest = F.interpolate(x_rest, scale_factor=2.0)
                        # This is 3D interpolation: (B, C, T-1, H, W) -> (B, C, 2*(T-1), 2*H, 2*W)
                        # In JAX format: (B, T-1, H, W, C) -> (B, 2*(T-1), 2*H, 2*W, C)
                        rest_frames = jax.image.resize(
                            rest_frames,
                            (B, 2 * (T-1), H * 2, W * 2, C),
                            method='nearest'
                        )
                        
                        # Concatenate: (B, 1, H*2, W*2, C) + (B, 2*(T-1), H*2, W*2, C)
                        # Result: (B, 1+2*(T-1), H*2, W*2, C) = (B, 2*T-1, H*2, W*2, C)
                        hidden_states = jnp.concatenate([first_frame, rest_frames], axis=1)
                    elif T > 1:
                        # Even frames: regular 3D interpolation
                        # PyTorch: inputs = F.interpolate(inputs, scale_factor=2.0)
                        # (B, T, H, W, C) -> (B, 2*T, 2*H, 2*W, C)
                        hidden_states = jax.image.resize(
                            hidden_states,
                            (B, T * 2, H * 2, W * 2, C),
                            method='nearest'
                        )
                    else:
                        # Single frame with compress_time: upsample to 2 frames AND spatial 2x
                        # 输入: (B, 1, H, W, C) → 输出: (B, 2, H*2, W*2, C)
                        # 使用 jax.image.resize 同时对时间和空间维度进行上采样
                        hidden_states = jax.image.resize(
                            hidden_states,
                            (B, 2, H * 2, W * 2, C),  # 时间维度 1→2，空间维度 2x
                            method='nearest'
                        )
                else:
                    # Only interpolate spatial dimensions (2D)
                    # Combine batch and time for processing
                    hidden_states = hidden_states.reshape(B * T, H, W, C)
                    
                    # Nearest neighbor upsampling 2x spatial only
                    hidden_states = jax.image.resize(
                        hidden_states,
                        (B * T, H * 2, W * 2, C),
                        method='nearest'
                    )
                    
                    # Reshape back to 5D
                    hidden_states = hidden_states.reshape(B, T, H * 2, W * 2, C)
                
                # Apply 2D convolution on spatial dimensions
                # Reshape: (B, T, H', W', C) → (B*T, H', W', C)
                B, T_new, H_new, W_new, C = hidden_states.shape
                hidden_states = hidden_states.reshape(B * T_new, H_new, W_new, C)
                hidden_states = upsampler(hidden_states)
                
                # Reshape back: (B*T, H', W', C) → (B, T, H', W', C)
                _, H_final, W_final, _ = hidden_states.shape
                hidden_states = hidden_states.reshape(B, T_new, H_final, W_final, C)
        
        return hidden_states, new_conv_cache


class FlaxCogVideoXEncoder3D(nnx.Module):
    """
    Complete encoder network for CogVideoX VAE.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 16,
        down_block_types: Tuple[str, ...] = (
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
        ),
        block_out_channels: Tuple[int, ...] = (128, 256, 256, 512),
        layers_per_block: int = 3,
        act_fn: str = "silu",
        norm_eps: float = 1e-6,
        norm_num_groups: int = 32,
        dropout: float = 0.0,
        pad_mode: str = "first",
        temporal_compression_ratio: float = 4,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        
        # Calculate temporal compression level
        import numpy as np
        temporal_compress_level = int(np.log2(temporal_compression_ratio))
        
        # Input convolution
        self.conv_in = FlaxCogVideoXCausalConv3d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            pad_mode=pad_mode,
            rngs=rngs
        )
        
        # Down blocks
        down_blocks = []
        output_channel = block_out_channels[0]
        
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            compress_time = i < temporal_compress_level
            
            down_block = FlaxCogVideoXDownBlock3D(
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=0,
                dropout=dropout,
                num_layers=layers_per_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                add_downsample=not is_final_block,
                compress_time=compress_time,
                pad_mode=pad_mode,
                rngs=rngs
            )
            
            down_blocks.append(down_block)
        self.down_blocks = nnx.List(down_blocks)
        
        # Mid block
        self.mid_block = FlaxCogVideoXMidBlock3D(
            in_channels=block_out_channels[-1],
            temb_channels=0,
            dropout=dropout,
            num_layers=2,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            resnet_groups=norm_num_groups,
            pad_mode=pad_mode,
            rngs=rngs
        )
        
        # Output layers
        self.norm_out = FlaxGroupNorm(norm_num_groups, block_out_channels[-1], epsilon=1e-6, rngs=rngs)
        self.conv_out = FlaxCogVideoXCausalConv3d(
            block_out_channels[-1],
            2 * out_channels,
            kernel_size=3,
            pad_mode=pad_mode,
            rngs=rngs
        )
    
    def __call__(
        self,
        sample: jnp.ndarray,
        temb: Optional[jnp.ndarray] = None,
        conv_cache: Optional[Dict[str, jnp.ndarray]] = None,
        deterministic: bool = True,
    ):
        new_conv_cache = {}
        conv_cache = conv_cache or {}
        
        # Input conv
        hidden_states, new_conv_cache["conv_in"] = self.conv_in(
            sample, conv_cache=conv_cache.get("conv_in")
        )
        
        # Down blocks
        for i, down_block in enumerate(self.down_blocks):
            conv_cache_key = f"down_block_{i}"
            hidden_states, new_conv_cache[conv_cache_key] = down_block(
                hidden_states, temb, None, conv_cache=conv_cache.get(conv_cache_key), deterministic=deterministic
            )
        
        # Mid block
        hidden_states, new_conv_cache["mid_block"] = self.mid_block(
            hidden_states, temb, None, conv_cache=conv_cache.get("mid_block"), deterministic=deterministic
        )
        
        # Output
        hidden_states = self.norm_out(hidden_states)
        hidden_states = jax.nn.silu(hidden_states)
        hidden_states, new_conv_cache["conv_out"] = self.conv_out(
            hidden_states, conv_cache=conv_cache.get("conv_out")
        )
        
        return hidden_states, new_conv_cache


class FlaxCogVideoXDecoder3D(nnx.Module):
    """
    Complete decoder network for CogVideoX VAE.
    """
    
    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 3,
        up_block_types: Tuple[str, ...] = (
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
        ),
        block_out_channels: Tuple[int, ...] = (128, 256, 256, 512),
        layers_per_block: int = 3,
        act_fn: str = "silu",
        norm_eps: float = 1e-6,
        norm_num_groups: int = 32,
        dropout: float = 0.0,
        pad_mode: str = "first",
        temporal_compression_ratio: float = 4,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        
        reversed_block_out_channels = list(reversed(block_out_channels))
        
        # Input convolution
        self.conv_in = FlaxCogVideoXCausalConv3d(
            in_channels,
            reversed_block_out_channels[0],
            kernel_size=3,
            pad_mode=pad_mode,
            rngs=rngs
        )
        
        # Mid block
        self.mid_block = FlaxCogVideoXMidBlock3D(
            in_channels=reversed_block_out_channels[0],
            temb_channels=0,
            num_layers=2,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            resnet_groups=norm_num_groups,
            spatial_norm_dim=in_channels,
            pad_mode=pad_mode,
            rngs=rngs
        )
        
        # Up blocks
        import numpy as np
        temporal_compress_level = int(np.log2(temporal_compression_ratio))
        
        up_blocks = []
        output_channel = reversed_block_out_channels[0]
        
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            compress_time = i < temporal_compress_level
            
            up_block = FlaxCogVideoXUpBlock3D(
                in_channels=prev_output_channel,
                out_channels=output_channel,
                temb_channels=0,
                dropout=dropout,
                num_layers=layers_per_block + 1,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                spatial_norm_dim=in_channels,
                add_upsample=not is_final_block,
                compress_time=compress_time,
                pad_mode=pad_mode,
                rngs=rngs
            )
            
            up_blocks.append(up_block)
        self.up_blocks = nnx.List(up_blocks)
        
        # Output layers
        self.norm_out = FlaxCogVideoXSpatialNorm3D(
            reversed_block_out_channels[-1],
            in_channels,
            groups=norm_num_groups,
            rngs=rngs
        )
        self.conv_out = FlaxCogVideoXCausalConv3d(
            reversed_block_out_channels[-1],
            out_channels,
            kernel_size=3,
            pad_mode=pad_mode,
            rngs=rngs
        )
    
    def __call__(
        self,
        sample: jnp.ndarray,
        zq: jnp.ndarray,
        temb: Optional[jnp.ndarray] = None,
        conv_cache: Optional[Dict[str, jnp.ndarray]] = None,
        feat_cache: Optional[list] = None,
        feat_idx: Optional[list] = None,
        deterministic: bool = True,
    ):
        """
        Decoder forward pass.
        
        支持两种缓存模式：
        1. conv_cache: 旧模式（保持兼容性）
        2. feat_cache/feat_idx: 新模式（逐帧解码）
        
        Args:
            sample: Latent representation (B, T, H, W, C)
            zq: Spatial conditioning (same as sample for CogVideoX)
            temb: Time embedding (optional)
            conv_cache: 旧模式的缓存字典
            feat_cache: 新模式的缓存列表
            feat_idx: 新模式的索引列表
            deterministic: Whether to use dropout
        """
        # 新模式：使用 feat_cache/feat_idx
        if feat_cache is not None and feat_idx is not None:
            print(f"[Decoder] 输入 sample 形状: {sample.shape}")
            
            # Input conv
            hidden_states, _ = self.conv_in(sample, feat_cache=feat_cache, feat_idx=feat_idx)
            
            # Mid block
            hidden_states, _ = self.mid_block(
                hidden_states, temb, sample,
                feat_cache=feat_cache, feat_idx=feat_idx,
                deterministic=deterministic
            )
            
            # Up blocks
            for i, up_block in enumerate(self.up_blocks):
                hidden_states, _ = up_block(
                    hidden_states, temb, sample,
                    feat_cache=feat_cache, feat_idx=feat_idx,
                    deterministic=deterministic
                )
            
            # Output
            hidden_states, _ = self.norm_out(hidden_states, sample, feat_cache=feat_cache, feat_idx=feat_idx)
            hidden_states = jax.nn.silu(hidden_states)
            hidden_states, _ = self.conv_out(hidden_states, feat_cache=feat_cache, feat_idx=feat_idx)
            
            return hidden_states, None
        
        # 旧模式：使用 conv_cache
        new_conv_cache = {}
        conv_cache = conv_cache or {}
        
        # Input conv
        hidden_states, new_conv_cache["conv_in"] = self.conv_in(
            sample, conv_cache=conv_cache.get("conv_in")
        )
        
        # Mid block
        hidden_states, new_conv_cache["mid_block"] = self.mid_block(
            hidden_states, temb, sample, conv_cache=conv_cache.get("mid_block"), deterministic=deterministic
        )
        
        # Up blocks
        for i, up_block in enumerate(self.up_blocks):
            conv_cache_key = f"up_block_{i}"
            hidden_states, new_conv_cache[conv_cache_key] = up_block(
                hidden_states, temb, sample, conv_cache=conv_cache.get(conv_cache_key), deterministic=deterministic
            )
        
        # Output
        hidden_states, new_conv_cache["norm_out"] = self.norm_out(
            hidden_states, sample, conv_cache=conv_cache.get("norm_out")
        )
        hidden_states = jax.nn.silu(hidden_states)
        hidden_states, new_conv_cache["conv_out"] = self.conv_out(
            hidden_states, conv_cache=conv_cache.get("conv_out")
        )
        
        return hidden_states, new_conv_cache


# Continued in next part...

class FlaxCogVideoXCache:
    """
    缓存管理类，用于逐帧解码时存储 CausalConv3d 层的中间结果。
    
    基于 AutoencoderKLWanCache 的设计，但针对 CogVideoX 的结构优化。
    这个缓存允许我们逐帧处理 decoder，避免一次性加载所有帧导致 OOM。
    """
    
    def __init__(self, decoder_module):
        """
        初始化缓存。
        
        Args:
            decoder_module: FlaxCogVideoXDecoder3D 实例
        """
        self.decoder_module = decoder_module
        self.clear_cache()
    
    def clear_cache(self):
        """重置所有缓存和索引"""
        # 计算 decoder 中有多少个 CausalConv3d 层
        self._conv_num = self._count_causal_conv3d(self.decoder_module)
        self._conv_idx = [0]  # 使用列表以便在函数间传递引用
        self._feat_map = [None] * self._conv_num
    
    @staticmethod
    def _count_causal_conv3d(module):
        """
        递归计算模块中 FlaxCogVideoXCausalConv3d 层的数量。
        
        Args:
            module: nnx.Module 实例
            
        Returns:
            int: CausalConv3d 层的数量
        """
        count = 0
        # 使用 nnx.graph.iter_graph 遍历所有子模块
        node_types = nnx.graph.iter_graph([module])
        for _, value in node_types:
            if isinstance(value, FlaxCogVideoXCausalConv3d):
                count += 1
        return count


class FlaxAutoencoderKLCogVideoX(nnx.Module):
    """
    A complete VAE model with KL loss for encoding videos into latents and decoding latent representations into videos.
    
    This is the JAX/Flax implementation of AutoencoderKLCogVideoX with full feature parity including:
    - Tiling for memory efficiency
    - Frame batch processing
    - Conv cache for long sequences
    
    Args:
        config: Configuration object with all VAE hyperparameters
        rngs: Random number generators
        dtype: Data type (e.g., jnp.float32, jnp.bfloat16)
    """
    
    config_class = FlaxAutoencoderKLCogVideoXConfig
    
    def __init__(
        self,
        config: FlaxAutoencoderKLCogVideoXConfig,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
    ):
        super().__init__()
        self.config = config
        self.dtype = dtype
        
        # Create encoder and decoder
        self.encoder = FlaxCogVideoXEncoder3D(
            in_channels=config.in_channels,
            out_channels=config.latent_channels,
            down_block_types=config.down_block_types,
            block_out_channels=config.block_out_channels,
            layers_per_block=config.layers_per_block,
            act_fn=config.act_fn,
            norm_eps=config.norm_eps,
            norm_num_groups=config.norm_num_groups,
            temporal_compression_ratio=config.temporal_compression_ratio,
            pad_mode=config.pad_mode,
            rngs=rngs
        )
        
        self.decoder = FlaxCogVideoXDecoder3D(
            in_channels=config.latent_channels,
            out_channels=config.out_channels,
            up_block_types=config.up_block_types,
            block_out_channels=config.block_out_channels,
            layers_per_block=config.layers_per_block,
            act_fn=config.act_fn,
            norm_eps=config.norm_eps,
            norm_num_groups=config.norm_num_groups,
            temporal_compression_ratio=config.temporal_compression_ratio,
            pad_mode=config.pad_mode,
            rngs=rngs
        )
        
        # Optional quant/post_quant conv (typically not used in CogVideoX)
        if config.use_quant_conv:
            self.quant_conv = FlaxConv3d(
                2 * config.latent_channels,
                2 * config.latent_channels,
                kernel_size=1,
                rngs=rngs
            )
        else:
            self.quant_conv = None
        
        if config.use_post_quant_conv:
            self.post_quant_conv = FlaxConv3d(
                config.latent_channels,
                config.latent_channels,
                kernel_size=1,
                rngs=rngs
            )
        else:
            self.post_quant_conv = None
        
        # Tiling parameters
        self.use_tiling = False
        self.tile_sample_min_height = config.sample_height // 2
        self.tile_sample_min_width = config.sample_width // 2
        self.tile_latent_min_height = int(
            self.tile_sample_min_height / (2 ** (len(config.block_out_channels) - 1))
        )
        self.tile_latent_min_width = int(
            self.tile_sample_min_width / (2 ** (len(config.block_out_channels) - 1))
        )
        self.tile_overlap_factor_height = 1 / 6
        self.tile_overlap_factor_width = 1 / 5
        
        # Frame batch sizes for processing
        # decode 时必须逐帧处理（batch=1）以避免 OOM
        # batch=2 会导致 40GB 内存需求，超过 TPU v6e 的 32GB 限制
        self.num_latent_frames_batch_size = 1  # decode 时每批 1 帧
        self.num_sample_frames_batch_size = 8  # encode 时每批 8 帧
    
    def enable_tiling(
        self,
        tile_sample_min_height: Optional[int] = None,
        tile_sample_min_width: Optional[int] = None,
        tile_overlap_factor_height: Optional[float] = None,
        tile_overlap_factor_width: Optional[float] = None,
    ):
        """Enable tiled VAE decoding for memory efficiency."""
        self.use_tiling = True
        if tile_sample_min_height is not None:
            self.tile_sample_min_height = tile_sample_min_height
        if tile_sample_min_width is not None:
            self.tile_sample_min_width = tile_sample_min_width
        if tile_overlap_factor_height is not None:
            self.tile_overlap_factor_height = tile_overlap_factor_height
        if tile_overlap_factor_width is not None:
            self.tile_overlap_factor_width = tile_overlap_factor_width
        
        self.tile_latent_min_height = int(
            self.tile_sample_min_height / (2 ** (len(self.config.block_out_channels) - 1))
        )
        self.tile_latent_min_width = int(
            self.tile_sample_min_width / (2 ** (len(self.config.block_out_channels) - 1))
        )
    
    def disable_tiling(self):
        """Disable tiled VAE decoding."""
        self.use_tiling = False
    
    @staticmethod
    def blend_v(a: jnp.ndarray, b: jnp.ndarray, blend_extent: int) -> jnp.ndarray:
        """Blend two tiles vertically with smooth transition."""
        blend_extent = min(a.shape[2], b.shape[2], blend_extent)
        for y in range(blend_extent):
            alpha = y / blend_extent
            b = b.at[:, :, y, :, :].set(
                a[:, :, -blend_extent + y, :, :] * (1 - alpha) + b[:, :, y, :, :] * alpha
            )
        return b
    
    @staticmethod
    def blend_h(a: jnp.ndarray, b: jnp.ndarray, blend_extent: int) -> jnp.ndarray:
        """Blend two tiles horizontally with smooth transition."""
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for x in range(blend_extent):
            alpha = x / blend_extent
            b = b.at[:, :, :, x, :].set(
                a[:, :, :, -blend_extent + x, :] * (1 - alpha) + b[:, :, :, x, :] * alpha
            )
        return b
    
    def _encode(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """
        Internal encode with frame batching.
        
        Args:
            x: Input video (B, T, H, W, C)
            deterministic: Whether to use dropout
            
        Returns:
            Encoded latents
        """
        batch_size, num_frames, height, width, num_channels = x.shape
        
        if self.use_tiling and (width > self.tile_sample_min_width or height > self.tile_sample_min_height):
            return self.tiled_encode(x, deterministic=deterministic)
        
        # Frame batching
        frame_batch_size = self.num_sample_frames_batch_size
        num_batches = max(num_frames // frame_batch_size, 1)
        conv_cache = None
        enc = []
        
        for i in range(num_batches):
            remaining_frames = num_frames % frame_batch_size
            start_frame = frame_batch_size * i + (0 if i == 0 else remaining_frames)
            end_frame = frame_batch_size * (i + 1) + remaining_frames
            x_intermediate = x[:, start_frame:end_frame, :, :, :]
            
            x_intermediate, conv_cache = self.encoder(
                x_intermediate, conv_cache=conv_cache, deterministic=deterministic
            )
            
            if self.quant_conv is not None:
                x_intermediate = self.quant_conv(x_intermediate)
            
            enc.append(x_intermediate)
        
        enc = jnp.concatenate(enc, axis=1)
        return enc
    
    def encode(self, x: jnp.ndarray, deterministic: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Encode a batch of videos into latents.
        
        Args:
            x: Input batch of videos (B, T, H, W, C)
            deterministic: Whether to use dropout
            
        Returns:
            mean: Mean of the latent distribution
            logvar: Log variance of the latent distribution
        """
        h = self._encode(x, deterministic=deterministic)
        
        # Split into mean and logvar
        mean, logvar = jnp.split(h, 2, axis=-1)
        
        return mean, logvar
    
    def _decode(self, z: jnp.ndarray, zq: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """
        Internal decode with frame-by-frame processing using feat_cache.
        
        这是参考 WAN VAE 的逐帧解码实现，通过 FlaxCogVideoXCache 管理所有 CausalConv3d 层的缓存。
        每个 latent 帧独立处理并上采样到 temporal_compression_ratio 倍的视频帧。
        
        Args:
            z: Latent representation (B, T, H, W, C)
            zq: Spatial conditioning (same as z for CogVideoX)
            deterministic: Whether to use dropout
            
        Returns:
            Decoded video (B, T*temporal_compression_ratio, H, W, C)
        """
        batch_size, num_frames, height, width, num_channels = z.shape
        
        if self.use_tiling and (width > self.tile_latent_min_width or height > self.tile_latent_min_height):
            return self.tiled_decode(z, zq, deterministic=deterministic)
        
        # 创建缓存管理器（参考 WAN 的 _decode 实现）
        # 缓存在所有 latent 帧之间共享，保持时间连续性
        feat_cache_manager = FlaxCogVideoXCache(self.decoder)
        
        # 应用 post_quant_conv 到整个 latent（如果存在）
        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)
        
        # 逐帧解码（缓存共享，每帧只重置索引）
        decoded_frames_list = []
        
        for i in range(num_frames):
            # 每帧重置索引（不清空缓存，保持帧间连续性）
            feat_cache_manager._conv_idx = [0]
            
            # 提取当前 latent 帧
            z_frame = z[:, i:i+1, :, :, :]
            zq_frame = zq[:, i:i+1, :, :, :]
            
            print(f"[_decode] latent 帧 {i}: z_frame 形状 {z_frame.shape}")
            
            # 使用共享缓存解码当前帧
            # 每个 latent 帧会被上采样到 temporal_compression_ratio 倍的视频帧
            decoded_frame, _ = self.decoder(
                z_frame, zq_frame,
                feat_cache=feat_cache_manager._feat_map,
                feat_idx=feat_cache_manager._conv_idx,
                deterministic=deterministic
            )
            
            print(f"[_decode] latent 帧 {i}: decoded_frame 形状 {decoded_frame.shape}")
            decoded_frames_list.append(decoded_frame)
        
        # 拼接所有解码后的帧
        decoded = jnp.concatenate(decoded_frames_list, axis=1)
        decoded = jnp.clip(decoded, min=-1.0, max=1.0)
        
        return decoded
    
    def decode(self, z: jnp.ndarray, zq: Optional[jnp.ndarray] = None, deterministic: bool = True) -> jnp.ndarray:
        """
        Decode a batch of latents to videos.
        
        Args:
            z: Input batch of latent vectors (B, T, H, W, C)
            zq: Spatial conditioning (defaults to z if not provided)
            deterministic: Whether to use dropout
            
        Returns:
            Decoded video
        """
        if zq is None:
            zq = z
        
        decoded = self._decode(z, zq, deterministic=deterministic)
        return decoded
    
    def tiled_encode(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """
        Encode using tiling for memory efficiency.
        
        Args:
            x: Input video (B, T, H, W, C)
            deterministic: Whether to use dropout
            
        Returns:
            Encoded latents
        """
        batch_size, num_frames, height, width, num_channels = x.shape
        
        overlap_height = int(self.tile_sample_min_height * (1 - self.tile_overlap_factor_height))
        overlap_width = int(self.tile_sample_min_width * (1 - self.tile_overlap_factor_width))
        blend_extent_height = int(self.tile_latent_min_height * self.tile_overlap_factor_height)
        blend_extent_width = int(self.tile_latent_min_width * self.tile_overlap_factor_width)
        row_limit_height = self.tile_latent_min_height - blend_extent_height
        row_limit_width = self.tile_latent_min_width - blend_extent_width
        frame_batch_size = self.num_sample_frames_batch_size
        
        # Split into tiles
        rows = []
        for i in range(0, height, overlap_height):
            row = []
            for j in range(0, width, overlap_width):
                num_batches = max(num_frames // frame_batch_size, 1)
                conv_cache = None
                time = []
                
                for k in range(num_batches):
                    remaining_frames = num_frames % frame_batch_size
                    start_frame = frame_batch_size * k + (0 if k == 0 else remaining_frames)
                    end_frame = frame_batch_size * (k + 1) + remaining_frames
                    
                    tile = x[
                        :,
                        start_frame:end_frame,
                        i:i + self.tile_sample_min_height,
                        j:j + self.tile_sample_min_width,
                        :
                    ]
                    
                    tile, conv_cache = self.encoder(tile, conv_cache=conv_cache, deterministic=deterministic)
                    
                    if self.quant_conv is not None:
                        tile = self.quant_conv(tile)
                    
                    time.append(tile)
                
                row.append(jnp.concatenate(time, axis=1))
            rows.append(row)
        
        # Blend tiles
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent_width)
                result_row.append(tile[:, :, :row_limit_height, :row_limit_width, :])
            result_rows.append(jnp.concatenate(result_row, axis=3))
        
        enc = jnp.concatenate(result_rows, axis=2)
        return enc
    
    def tiled_decode(self, z: jnp.ndarray, zq: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """
        Decode using tiling for memory efficiency.
        
        Args:
            z: Latent representation
            zq: Spatial conditioning
            deterministic: Whether to use dropout
            
        Returns:
            Decoded video
        """
        batch_size, num_frames, height, width, num_channels = z.shape
        
        overlap_height = int(self.tile_latent_min_height * (1 - self.tile_overlap_factor_height))
        overlap_width = int(self.tile_latent_min_width * (1 - self.tile_overlap_factor_width))
        blend_extent_height = int(self.tile_sample_min_height * self.tile_overlap_factor_height)
        blend_extent_width = int(self.tile_sample_min_width * self.tile_overlap_factor_width)
        row_limit_height = self.tile_sample_min_height - blend_extent_height
        row_limit_width = self.tile_sample_min_width - blend_extent_width
        frame_batch_size = self.num_latent_frames_batch_size
        
        # Split into tiles
        rows = []
        for i in range(0, height, overlap_height):
            row = []
            for j in range(0, width, overlap_width):
                num_batches = max(num_frames // frame_batch_size, 1)
                conv_cache = None
                time = []
                
                for k in range(num_batches):
                    remaining_frames = num_frames % frame_batch_size
                    start_frame = frame_batch_size * k + (0 if k == 0 else remaining_frames)
                    end_frame = frame_batch_size * (k + 1) + remaining_frames
                    
                    tile = z[
                        :,
                        start_frame:end_frame,
                        i:i + self.tile_latent_min_height,
                        j:j + self.tile_latent_min_width,
                        :
                    ]
                    
                    tile_zq = zq[
                        :,
                        start_frame:end_frame,
                        i:i + self.tile_latent_min_height,
                        j:j + self.tile_latent_min_width,
                        :
                    ]
                    
                    if self.post_quant_conv is not None:
                        tile = self.post_quant_conv(tile)
                    
                    tile, conv_cache = self.decoder(tile, tile_zq, conv_cache=conv_cache, deterministic=deterministic)
                    time.append(tile)
                
                row.append(jnp.concatenate(time, axis=1))
            rows.append(row)
        
        # Blend tiles
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent_width)
                result_row.append(tile[:, :, :row_limit_height, :row_limit_width, :])
            result_rows.append(jnp.concatenate(result_row, axis=3))
        
        dec = jnp.concatenate(result_rows, axis=2)
        return dec
    
    def __call__(
        self,
        x: jnp.ndarray,
        sample_posterior: bool = False,
        deterministic: bool = True,
        rng: Optional[jax.random.PRNGKey] = None,
    ) -> jnp.ndarray:
        """
        Forward pass through the VAE.
        
        Args:
            x: Input video
            sample_posterior: Whether to sample from posterior or use mode
            deterministic: Whether to use dropout
            rng: Random key for sampling
            
        Returns:
            Reconstructed video
        """
        mean, logvar = self.encode(x, deterministic=deterministic)
        
        if sample_posterior:
            if rng is None:
                raise ValueError("rng must be provided when sample_posterior=True")
            std = jnp.exp(0.5 * logvar)
            z = mean + std * jax.random.normal(rng, mean.shape)
        else:
            z = mean  # mode() = mean for diagonal Gaussian
        
        # decoder uses z as both latent and spatial conditioning (zq)
        dec = self.decode(z, zq=z, deterministic=deterministic)
        return dec
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        subfolder: Optional[str] = "vae",
        dtype: jnp.dtype = jnp.float32,
        **kwargs,
    ):
        """
        Load pre-trained CogVideoX VAE weights from HuggingFace.
        
        This method downloads PyTorch weights and converts them to JAX/Flax format on-the-fly,
        following the approach used in jax-huggingface examples.
        
        Args:
            pretrained_model_name_or_path: Model ID (e.g., "THUDM/CogVideoX-2b")
            subfolder: Subfolder containing VAE weights (default: "vae")
            dtype: Target dtype for weights (default: jnp.float32)
            **kwargs: Additional config overrides
            
        Returns:
            FlaxAutoencoderKLCogVideoX: Initialized model with loaded weights
            
        Example:
            ```python
            vae = FlaxAutoencoderKLCogVideoX.from_pretrained(
                "THUDM/CogVideoX-2b",
                subfolder="vae",
                dtype=jnp.bfloat16
            )
            ```
        """
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open
        import re
        import json
        
        print(f"[1/4] 加载配置: {pretrained_model_name_or_path}")
        
        # Download and load config.json
        config_path = hf_hub_download(
            pretrained_model_name_or_path,
            subfolder=subfolder,
            filename="config.json"
        )
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Override with kwargs
        config_dict.update(kwargs)
        
        # Create config using from_dict
        config = cls.config_class.from_dict(config_dict)
        
        print(f"[2/4] 下载 PyTorch 权重...")
        
        # Download weights
        ckpt_path = hf_hub_download(
            pretrained_model_name_or_path,
            subfolder=subfolder,
            filename="diffusion_pytorch_model.safetensors"
        )
        
        print(f"[3/4] 转换权重到 JAX 格式...")
        
        # Load PyTorch weights
        pytorch_weights = {}
        with safe_open(ckpt_path, framework="np") as f:
            for key in f.keys():
                pytorch_weights[key] = f.get_tensor(key)
        
        print(f"  ✓ 加载了 {len(pytorch_weights)} 个 PyTorch 权重张量")
        
        # Convert PyTorch weights to JAX format
        jax_weights = {}
        
        for pt_key, pt_tensor in pytorch_weights.items():
            # Remove _orig_mod prefix if present
            if pt_key.startswith("_orig_mod."):
                pt_key = pt_key[len("_orig_mod."):]
            
            jax_key = pt_key
            jax_tensor = pt_tensor
            
            # Handle encoder down blocks structure mapping
            if m := re.match(r'encoder\.down_blocks\.(\d+)\.resnets\.(\d+)\.(.*)', jax_key):
                block_idx, resnet_idx, rest = m.groups()
                jax_key = f'encoder.down_blocks.{block_idx}.resnets.{resnet_idx}.{rest}'
            
            elif m := re.match(r'encoder\.down_blocks\.(\d+)\.downsamplers\.0\.(.*)', jax_key):
                block_idx, rest = m.groups()
                jax_key = f'encoder.down_blocks.{block_idx}.downsamplers.0.{rest}'
            
            # Handle encoder mid block
            elif m := re.match(r'encoder\.mid_block\.resnets\.(\d+)\.(.*)', jax_key):
                resnet_idx, rest = m.groups()
                jax_key = f'encoder.mid_block.resnets.{resnet_idx}.{rest}'
            
            # Handle decoder mid block
            elif m := re.match(r'decoder\.mid_block\.resnets\.(\d+)\.(.*)', jax_key):
                resnet_idx, rest = m.groups()
                jax_key = f'decoder.mid_block.resnets.{resnet_idx}.{rest}'
            
            # Handle decoder up blocks
            elif m := re.match(r'decoder\.up_blocks\.(\d+)\.resnets\.(\d+)\.(.*)', jax_key):
                block_idx, resnet_idx, rest = m.groups()
                jax_key = f'decoder.up_blocks.{block_idx}.resnets.{resnet_idx}.{rest}'
            
            elif m := re.match(r'decoder\.up_blocks\.(\d+)\.upsamplers\.0\.(.*)', jax_key):
                block_idx, rest = m.groups()
                jax_key = f'decoder.up_blocks.{block_idx}.upsamplers.0.{rest}'
            
            # Add .conv for conv layers (our FlaxConv3d/FlaxConv2d wraps actual conv)
            needs_conv = False
            conv_patterns = ['.conv_in.', '.conv_out.', '.conv1.', '.conv2.',
                           '.conv_shortcut.', '.conv_y.', '.conv_b.',
                           '.downsamplers.', '.upsamplers.']
            if any(pattern in jax_key for pattern in conv_patterns):
                if not (jax_key.endswith('.conv.weight') or jax_key.endswith('.conv.bias') or
                       jax_key.endswith('.conv.kernel') or jax_key.endswith('.conv.bias')):
                    if jax_key.endswith('.weight') or jax_key.endswith('.bias') or jax_key.endswith('.kernel'):
                        needs_conv = True
            
            if needs_conv:
                parts = jax_key.rsplit('.', 1)
                jax_key = f"{parts[0]}.conv.{parts[1]}"
            
            # Convert conv weights: PyTorch (O,I,T,H,W) -> JAX (T,H,W,I,O)
            if "conv" in jax_key and "weight" in jax_key:
                jax_key = jax_key.replace(".weight", ".kernel")
                
                if len(jax_tensor.shape) == 5:  # 3D conv
                    jax_tensor = jax_tensor.transpose(2, 3, 4, 1, 0)
                elif len(jax_tensor.shape) == 4:  # 2D conv (downsampler and upsampler both use FlaxConv2d)
                    # Convert (O, I, H, W) -> (H, W, I, O) for FlaxConv2d
                    jax_tensor = jax_tensor.transpose(2, 3, 1, 0)
            
            # Handle norm layers
            if ".weight" in jax_key and "norm" in jax_key:
                jax_key = jax_key.replace(".weight", ".scale")
            
            # Convert norm_layer in SpatialNorm3D
            if "norm_layer.weight" in jax_key:
                jax_key = jax_key.replace("norm_layer.weight", "norm_layer.scale")
            
            jax_weights[jax_key] = jnp.array(jax_tensor, dtype=dtype)
        
        print(f"  ✓ 转换了 {len(jax_weights)} 个权重到 JAX 格式")
        
        print(f"[4/4] 初始化模型并加载权重...")
        
        # Create model
        key = jax.random.key(0)
        rngs = nnx.Rngs(key)
        model = cls(config=config, rngs=rngs, dtype=dtype)
        
        # Convert flat dict to nested dict for nnx
        from flax.traverse_util import unflatten_dict
        nested_weights = unflatten_dict(jax_weights, sep=".")
        
        # Load weights into model
        graphdef, _ = nnx.split(model)
        model = nnx.merge(graphdef, nested_weights)
        
        print(f"✓ 模型加载完成!")
        print(f"  配置: {config}")
        
        return model