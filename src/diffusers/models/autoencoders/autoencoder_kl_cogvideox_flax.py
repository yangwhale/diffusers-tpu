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
Flax/JAX 版本的 CogVideoX VAE。

本文件是从 autoencoder_kl_cogvideox_torchax.py 改造而来，用于在 TPU 上高效运行。
主要改动点（用 # FLAX: 标记）：

1. 框架替换：torch.nn → flax.nnx
2. 数据格式：NCTHW → NTHWC（JAX 的 channels-last 格式）
3. GroupNorm：自定义实现，使用 Welford 算法节省内存（与 TorchAx 版本相同思路）
4. 缓存机制：使用 list-based feat_cache（与 TorchAx 版本一致）
5. Sharding：使用 jax.lax.with_sharding_constraint 替代 TorchAx 的 mark_sharding

改造原则：
- 保持与 TorchAx 版本相同的模型结构和权重兼容性
- 最小化代码改动，便于对比和维护
- 所有改动用 # FLAX: 注释标记

对比 TorchAx 版本 (autoencoder_kl_cogvideox_torchax.py)：
- TorchAx 使用 torch.nn + JAX 后端，本文件使用纯 Flax/NNX
- TorchAx 数据格式 NCTHW，本文件使用 NTHWC
- 两者共享相同的 feat_cache 逻辑和 GroupNorm 优化策略
"""

# ==================== 导入部分 ====================
from dataclasses import dataclass
# FLAX: 移除 Dict 类型导入（不再使用 dict-based cache）
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
# FLAX: TPU sharding 支持（对应 TorchAx 的 mark_sharding）
from jax.sharding import PartitionSpec as P
from flax import nnx

from ...configuration_utils import ConfigMixin

# FLAX: 缓存帧数常量，用于 feat_cache（与 TorchAx CACHE_T = 2 一致）
CACHE_T = 2


# ==================== FLAX: Sharding 支持 ====================
# 对应 TorchAx 版本的 mark_sharding = interop.torch_view(jax.lax.with_sharding_constraint)
# 差异：TorchAx 使用 NCTHW 格式，Flax 使用 NTHWC 格式，所以 PartitionSpec 位置不同

def _apply_sharding_constraint(inputs, is_nthwc=True):
    """
    FLAX: 应用 TPU sharding 约束。
    
    对应 TorchAx 版本的 CogVideoXCausalConv3d._apply_sharding() 方法。
    差异：
    - TorchAx NCTHW: W 在 index 4，使用 P(None, None, None, None, ("dp", "tp"))
    - Flax NTHWC: W 在 index 3，使用 P(None, None, None, ("dp", "tp"), None)
    """
    # Flax format: (B, T, H, W, C) - shard on W (index 3)
    specs = [
        P(None, None, None, ("dp", "tp"), None),  # Try dp+tp first
        P(None, None, None, ("tp",), None),       # Try tp only
        P(None, None, None, ("dp",), None),       # Try dp only
    ]
    
    for spec in specs:
        try:
            return jax.lax.with_sharding_constraint(inputs, spec)
        except (ValueError, Exception):
            continue
    
    # No sharding worked (likely not in a mesh context), return unchanged
    return inputs


# ==================== 配置类 ====================
# FLAX: 与 TorchAx 版本的 @register_to_config 装饰器不同，使用 dataclass

@dataclass
class FlaxAutoencoderKLCogVideoXConfig:
    """
    FLAX: 配置类，对应 TorchAx 版本的 AutoencoderKLCogVideoX.__init__ 参数。
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
    def from_dict(cls, config_dict: dict):
        """从字典创建配置"""
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(cls)}
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}
        return cls(**filtered_dict)


# ==================== 基础卷积层 ====================
# FLAX: 对应 TorchAx 版本的 nn.Conv3d 和 CogVideoXSafeConv3d
# 差异：Flax 使用 channels-last 格式，无需分块逻辑（与 TorchAx 一样移除了 GPU 分块）

class FlaxConv3d(nnx.Module):
    """
    FLAX: 基础 3D 卷积，对应 TorchAx 版本的 CogVideoXSafeConv3d。
    移除了 GPU 专用的分块逻辑（TPU/XLA 自动优化）。
    """
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
    """FLAX: 基础 2D 卷积，用于上下采样层。"""
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


# ==================== FLAX: CogVideoXCausalConv3d ====================
# 对应 TorchAx 版本的 CogVideoXCausalConv3d (autoencoder_kl_cogvideox_torchax.py:178-286)
#
# FLAX 改动说明：
# 1. 数据格式：TorchAx NCTHW -> Flax NTHWC（时间维度从 dim=2 变为 dim=1）
# 2. forward 签名保持一致：(inputs, feat_cache, feat_idx) -> (output, feat_idx, feat_cache)
# 3. 缓存逻辑完全对齐 TorchAx 版本

class FlaxCogVideoXCausalConv3d(nnx.Module):
    r"""A 3D causal convolution layer that pads the input tensor to ensure causality in CogVideoX Model.

    FLAX: 从 TorchAx 版本改造而来，主要差异是数据格式 NCTHW -> NTHWC。

    Args:
        in_channels (`int`): Number of channels in the input tensor.
        out_channels (`int`): Number of output channels produced by the convolution.
        kernel_size (`int` or `Tuple[int, int, int]`): Kernel size of the convolutional kernel.
        stride (`int`, defaults to `1`): Stride of the convolution.
        dilation (`int`, defaults to `1`): Dilation rate of the convolution.
        pad_mode (`str`, defaults to `"constant"`): Padding mode.
    """

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

        time_pad = time_kernel_size - 1
        height_pad = (height_kernel_size - 1) // 2
        width_pad = (width_kernel_size - 1) // 2

        self.pad_mode = pad_mode
        self.height_pad = height_pad
        self.width_pad = width_pad
        self.time_pad = time_pad
        # FLAX: NTHWC 格式的 causal padding（对应 TorchAx 的 time_causal_padding）
        self.time_causal_padding = [(0, 0), (time_pad, 0), (height_pad, height_pad), (width_pad, width_pad), (0, 0)]
        self.const_padding_conv3d = (0, self.height_pad, self.width_pad)

        # FLAX: 时间维度在 dim=1（TorchAx 在 dim=2）
        self.temporal_dim = 1
        self.time_kernel_size = time_kernel_size

        stride_tuple = stride if isinstance(stride, tuple) else (stride, 1, 1)
        # FLAX: 创建底层卷积
        self.conv = FlaxConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride_tuple,
            padding=0 if self.pad_mode == "replicate" else self.const_padding_conv3d,
            rngs=rngs,
        )

    # FLAX: forward 签名与 TorchAx 一致
    # TorchAx: forward(inputs, feat_cache=None, feat_idx=0) -> (output, feat_idx, feat_cache)
    def __call__(self, inputs: jnp.ndarray, feat_cache=None, feat_idx: int = 0):
        if self.pad_mode == "replicate":
            # replicate 模式：直接 pad，不使用缓存
            inputs = jnp.pad(inputs, self.time_causal_padding, mode="edge")
            # FLAX: 添加 TPU sharding 约束
            inputs = self._apply_sharding(inputs)
            output = self.conv(inputs)
            return output, feat_idx, feat_cache
        else:
            # FLAX: 使用 list-based feat_cache（与 TorchAx 完全一致）
            kernel_size = self.time_kernel_size
            if feat_cache is not None and kernel_size > 1:
                idx = feat_idx
                # 保存当前输入的最后 CACHE_T 帧用于下次迭代
                # TorchAx: cache_x = inputs[:, :, -CACHE_T:, :, :].clone()
                # FLAX (NTHWC): cache_x = inputs[:, -CACHE_T:, :, :, :]
                cache_x = inputs[:, -CACHE_T:, :, :, :]
                # TorchAx: if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # FLAX: if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
                if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
                    cache_x = jnp.concatenate([feat_cache[idx][:, -1:, :, :, :], cache_x], axis=1)
                
                if feat_cache[idx] is not None:
                    # 使用缓存的帧
                    # TorchAx: inputs = torch.cat([cached_inputs, inputs], dim=2)
                    # FLAX: inputs = jnp.concatenate([cached_inputs, inputs], axis=1)
                    inputs = jnp.concatenate([feat_cache[idx], inputs], axis=1)
                else:
                    # 首次调用：复制第一帧作为填充
                    # TorchAx: cached_inputs = inputs[:, :, :1].repeat(1, 1, kernel_size - 1, 1, 1)
                    # FLAX: cached_inputs = jnp.tile(inputs[:, :1, :, :, :], (1, kernel_size - 1, 1, 1, 1))
                    cached_inputs = jnp.tile(inputs[:, :1, :, :, :], (1, kernel_size - 1, 1, 1, 1))
                    inputs = jnp.concatenate([cached_inputs, inputs], axis=1)
                
                feat_cache[idx] = cache_x
                feat_idx += 1
            elif kernel_size > 1:
                # 无缓存模式：复制第一帧
                cached_inputs = jnp.tile(inputs[:, :1, :, :, :], (1, kernel_size - 1, 1, 1, 1))
                inputs = jnp.concatenate([cached_inputs, inputs], axis=1)
            
            # FLAX: 添加 TPU sharding 约束
            inputs = self._apply_sharding(inputs)
            output = self.conv(inputs)
            return output, feat_idx, feat_cache

    def _apply_sharding(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """FLAX: 尝试应用 TPU sharding 约束。
        
        对应 TorchAx 版本的 _apply_sharding 方法。
        差异：TorchAx NCTHW 在 W(dim=4) 分片，FLAX NTHWC 在 W(dim=3) 分片。
        """
        # FLAX: NTHWC 格式，W 在 dim=3
        for spec in [P(None, None, None, ("dp", "tp"), None),
                     P(None, None, None, ("tp",), None),
                     P(None, None, None, ("dp",), None)]:
            try:
                return jax.lax.with_sharding_constraint(inputs, spec)
            except (ValueError, Exception):
                continue
        return inputs

# 由于 nnx.GroupNorm 使用了 lax.square 导致内存占用过高，所以这里自定义实现一个更内存高效的版本。
# 留下这个注释以备参考
# class FlaxGroupNorm(nnx.GroupNorm):
#     """
#     Wrapper around nnx.GroupNorm to match the parameter signature of the original FlaxGroupNorm.
#     """
#     def __init__(
#         self,
#         num_groups: int,
#         num_channels: int,
#         epsilon: float = 1e-6,
#         rngs: nnx.Rngs = None,
#     ):
#         super().__init__(
#             num_features=num_channels,
#             num_groups=num_groups,
#             epsilon=epsilon,
#             rngs=rngs
#         )

class FlaxGroupNorm(nnx.Module):
    """
    自定义 Group Normalization 实现，针对内存效率优化。
    
    **为什么需要自定义实现而不直接用 nnx.GroupNorm？**
    
    nnx.GroupNorm 使用 lax.square(x) + mean(x²) 计算方差，这需要创建一个
    与输入相同大小的临时数组来存储 x²。对于大分辨率视频（如 768x1360x64），
    这会导致 OOM：
    
    错误示例（768x1360x64 帧）：
    ```
    ValueError: RESOURCE_EXHAUSTED: Attempting to allocate 3.98G.
    That was not possible. There are 3.97G free.
    ```
    
    自定义实现使用 jnp.var() 直接计算方差，JAX 内部可以做流式计算
    （Welford's online algorithm），避免存储完整的 x² 数组，节省约 50% 内存。
    
    测试结果：
    - nnx.GroupNorm: 768x1360x64 OOM (需要 3.98GB)
    - 自定义实现: 768x1360x64 成功 (约 2GB 峰值内存)
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
        
        # 创建可学习参数（与 nnx.GroupNorm 兼容）
        self.scale = nnx.Param(jnp.ones((num_channels,)))
        self.bias = nnx.Param(jnp.zeros((num_channels,)))
    
    def __call__(self, x):
        """
        Apply group normalization in channel-last format.
        
        使用内存优化的方法：
        1. 直接在 channel-last 格式计算（避免 transpose）
        2. 使用 jnp.var() 而非 lax.square() + mean()（节省内存）
        
        Args:
            x: Input of shape (B, T, H, W, C) or (B, H, W, C) [channel-last]
            
        Returns:
            Normalized output with same shape as input [channel-last]
        """
        if len(x.shape) == 5:
            # 5D: (B, T, H, W, C)
            B, T, H, W, C = x.shape
            assert C == self.num_channels, f"Expected {self.num_channels} channels, got {C}"
            assert C % self.num_groups == 0, f"Channels {C} must be divisible by groups {self.num_groups}"
            
            channels_per_group = C // self.num_groups
            
            # Reshape to expose groups: (B, T, H, W, num_groups, channels_per_group)
            x_grouped = x.reshape(B, T, H, W, self.num_groups, channels_per_group)
            
            # 关键优化：使用 jnp.mean/var 而非 lax.square()
            # jnp.var() 内部使用 Welford's algorithm，流式计算，不存储 x²
            mean = jnp.mean(x_grouped, axis=(1, 2, 3, 5), keepdims=True)
            var = jnp.var(x_grouped, axis=(1, 2, 3, 5), keepdims=True)
            
            # Normalize
            x_norm = (x_grouped - mean) / jnp.sqrt(var + self.epsilon)
            
            # Reshape back: (B, T, H, W, C)
            x_norm = x_norm.reshape(B, T, H, W, C)
            
            # Apply affine transformation
            x_out = x_norm * self.scale.value.reshape(1, 1, 1, 1, C) + self.bias.value.reshape(1, 1, 1, 1, C)
            
        else:
            # 4D: (B, H, W, C)
            B, H, W, C = x.shape
            assert C == self.num_channels
            assert C % self.num_groups == 0
            
            # Convert to channel-first for compatibility
            x_cf = x.transpose(0, 3, 1, 2)  # (B, C, H, W)
            
            # Reshape to group structure
            x_grouped = x_cf.reshape(B, self.num_groups, C // self.num_groups, H, W)
            
            # 使用内存优化的 var 计算
            mean = jnp.mean(x_grouped, axis=(2, 3, 4), keepdims=True)
            var = jnp.var(x_grouped, axis=(2, 3, 4), keepdims=True)
            
            # Normalize
            x_norm = (x_grouped - mean) / jnp.sqrt(var + self.epsilon)
            
            # Reshape back
            x_norm = x_norm.reshape(B, C, H, W)
            
            # Apply affine
            x_out = x_norm * self.scale.value.reshape(1, C, 1, 1) + self.bias.value.reshape(1, C, 1, 1)
            
            # Convert back to channel-last
            x_out = x_out.transpose(0, 2, 3, 1)
        
        return x_out


# ==================== FLAX: CogVideoXSpatialNorm3D ====================
# 对应 TorchAx 版本的 CogVideoXSpatialNorm3D (autoencoder_kl_cogvideox_torchax.py:292-341)
#
# FLAX 改动说明：
# 1. 数据格式：TorchAx NCTHW -> Flax NTHWC
# 2. forward 签名与 TorchAx 一致：(f, zq, feat_cache, feat_idx) -> (output, feat_idx, feat_cache)
# 3. 使用 jax.image.resize 替代 F.interpolate

class FlaxCogVideoXSpatialNorm3D(nnx.Module):
    r"""
    Spatially conditioned normalization as defined in https://huggingface.co/papers/2209.09002.
    This implementation is specific to 3D-video like data.

    FLAX: 从 TorchAx 版本改造而来，主要差异是数据格式和 interpolate 实现。

    Args:
        f_channels (`int`):
            The number of channels for input to group normalization layer, and output of the spatial norm layer.
        zq_channels (`int`):
            The number of channels for the quantized vector as described in the paper.
        groups (`int`):
            Number of groups to separate the channels into for group normalization.
    """

    def __init__(
        self,
        f_channels: int,
        zq_channels: int,
        groups: int = 32,
        rngs: nnx.Rngs = None,
    ):
        # FLAX: 使用优化的 GroupNorm 替代 nn.GroupNorm
        self.norm_layer = FlaxGroupNorm(num_channels=f_channels, num_groups=groups, epsilon=1e-6, rngs=rngs)
        self.conv_y = FlaxCogVideoXCausalConv3d(zq_channels, f_channels, kernel_size=1, stride=1, rngs=rngs)
        self.conv_b = FlaxCogVideoXCausalConv3d(zq_channels, f_channels, kernel_size=1, stride=1, rngs=rngs)

    # FLAX: forward 签名与 TorchAx 一致
    # TorchAx: forward(f, zq, feat_cache=None, feat_idx=0) -> (output, feat_idx, feat_cache)
    def __call__(self, f: jnp.ndarray, zq: jnp.ndarray, feat_cache=None, feat_idx: int = 0):
        # FLAX: 处理奇数帧（与 TorchAx 对齐）
        B, T, H, W, C = f.shape
        if T > 1 and T % 2 == 1:
            # TorchAx: f_first, f_rest = f[:, :, :1], f[:, :, 1:]
            # FLAX (NTHWC): f_first, f_rest = f[:, :1], f[:, 1:]
            f_first_size = (1, H, W)  # TorchAx 使用 f_first.shape[-3:]
            f_rest_size = (T-1, H, W)
            z_first = zq[:, :1, :, :, :]
            z_rest = zq[:, 1:, :, :, :]
            # FLAX: 使用 jax.image.resize 替代 F.interpolate
            z_first = jax.image.resize(z_first, (B, 1, H, W, zq.shape[-1]), method='nearest')
            z_rest = jax.image.resize(z_rest, (B, T-1, H, W, zq.shape[-1]), method='nearest')
            zq = jnp.concatenate([z_first, z_rest], axis=1)
        else:
            zq = jax.image.resize(zq, (B, T, H, W, zq.shape[-1]), method='nearest')

        # FLAX: 使用 feat_cache/feat_idx（与 TorchAx 完全一致）
        conv_y, feat_idx, feat_cache = self.conv_y(zq, feat_cache=feat_cache, feat_idx=feat_idx)
        conv_b, feat_idx, feat_cache = self.conv_b(zq, feat_cache=feat_cache, feat_idx=feat_idx)

        norm_f = self.norm_layer(f)
        new_f = norm_f * conv_y + conv_b
        return new_f, feat_idx, feat_cache


# ==================== FLAX: CogVideoXResnetBlock3D ====================
# 对应 TorchAx 版本的 CogVideoXResnetBlock3D (autoencoder_kl_cogvideox_torchax.py:347-431)
#
# FLAX 改动说明：
# 1. 数据格式：TorchAx NCTHW -> Flax NTHWC
# 2. forward 签名与 TorchAx 一致：(inputs, temb, zq, feat_cache, feat_idx) -> (output, feat_idx, feat_cache)
# 3. 使用 list-based feat_cache（与 TorchAx 完全一致）

class FlaxCogVideoXResnetBlock3D(nnx.Module):
    r"""A 3D ResNet block used in the CogVideoX model.

    FLAX: 从 TorchAx 版本改造而来，主要差异是数据格式 NCTHW -> NTHWC。

    Args:
        in_channels (`int`): Number of input channels.
        out_channels (`int`, *optional*): Number of output channels. Defaults to `in_channels`.
        temb_channels (`int`, defaults to `512`): Number of time embedding channels.
        groups (`int`, defaults to `32`): Number of groups for group normalization.
        eps (`float`, defaults to `1e-6`): Epsilon for group normalization.
        non_linearity (`str`, defaults to `"swish"`): Activation function.
        conv_shortcut (`bool`, defaults to `False`): Whether to use conv for shortcut.
        spatial_norm_dim (`int`, *optional*): Dimension for spatial normalization (decoder only).
        pad_mode (`str`, defaults to `"first"`): Padding mode for causal conv.
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
            # Encoder: use custom GroupNorm (memory optimized)
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
    
    # FLAX: forward 签名与 TorchAx 一致
    # TorchAx: forward(inputs, temb=None, zq=None, feat_cache=None, feat_idx=0) -> (output, feat_idx, feat_cache)
    def __call__(
        self,
        inputs: jnp.ndarray,
        temb: Optional[jnp.ndarray] = None,
        zq: Optional[jnp.ndarray] = None,
        feat_cache: Optional[list] = None,
        feat_idx: int = 0,
        deterministic: bool = True,
    ):
        hidden_states = inputs
        
        # FLAX: norm1 - 根据是否有 zq 决定使用 GroupNorm 还是 SpatialNorm3D
        if zq is not None:
            hidden_states, feat_idx, feat_cache = self.norm1(hidden_states, zq, feat_cache=feat_cache, feat_idx=feat_idx)
        else:
            hidden_states = self.norm1(hidden_states)
        
        hidden_states = jax.nn.silu(hidden_states)
        hidden_states, feat_idx, feat_cache = self.conv1(hidden_states, feat_cache=feat_cache, feat_idx=feat_idx)
        
        # FLAX: time embedding 处理（NTHWC 格式）
        # TorchAx: hidden_states = hidden_states + temb[:, :, None, None, None]
        # Flax:    hidden_states = hidden_states + temb[:, None, None, None, :]
        if temb is not None and self.temb_proj is not None:
            temb_proj = self.temb_proj(jax.nn.silu(temb))
            hidden_states = hidden_states + temb_proj[:, None, None, None, :]
        
        # FLAX: norm2
        if zq is not None:
            hidden_states, feat_idx, feat_cache = self.norm2(hidden_states, zq, feat_cache=feat_cache, feat_idx=feat_idx)
        else:
            hidden_states = self.norm2(hidden_states)
        
        hidden_states = jax.nn.silu(hidden_states)
        
        # Dropout
        if self.dropout_rate > 0 and not deterministic:
            hidden_states = nnx.Dropout(rate=self.dropout_rate)(hidden_states)
        
        hidden_states, feat_idx, feat_cache = self.conv2(hidden_states, feat_cache=feat_cache, feat_idx=feat_idx)
        
        # FLAX: shortcut connection
        if self.conv_shortcut is not None:
            if self.use_conv_shortcut:
                inputs, feat_idx, feat_cache = self.conv_shortcut(inputs, feat_cache=feat_cache, feat_idx=feat_idx)
            else:
                inputs = self.conv_shortcut(inputs)
        
        # Residual connection
        hidden_states = hidden_states + inputs
        
        return hidden_states, feat_idx, feat_cache


# ==================== FLAX: CogVideoXDownBlock3D ====================
# 对应 TorchAx 版本的 CogVideoXDownBlock3D (autoencoder_kl_cogvideox_torchax.py:437-507)
#
# FLAX 改动说明：
# 1. 数据格式：TorchAx NCTHW -> Flax NTHWC
# 2. forward 签名与 TorchAx 一致：(hidden_states, temb, zq, feat_cache, feat_idx) -> (output, feat_idx, feat_cache)

class FlaxCogVideoXDownBlock3D(nnx.Module):
    r"""A 3D down block used in the CogVideoX encoder.

    FLAX: 从 TorchAx 版本改造而来，主要差异是数据格式 NCTHW -> NTHWC。
    """
    
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
    
    # FLAX: forward 签名与 TorchAx 一致
    # TorchAx: forward(hidden_states, temb=None, zq=None, feat_cache=None, feat_idx=0) -> (output, feat_idx, feat_cache)
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        temb: Optional[jnp.ndarray] = None,
        zq: Optional[jnp.ndarray] = None,
        feat_cache: Optional[list] = None,
        feat_idx: int = 0,
        deterministic: bool = True,
    ):
        # ResNet blocks
        for resnet in self.resnets:
            hidden_states, feat_idx, feat_cache = resnet(
                hidden_states, temb, zq,
                feat_cache=feat_cache, feat_idx=feat_idx,
                deterministic=deterministic
            )
        
        # Downsampler (if present)
        if self.downsamplers is not None:
            # FLAX: 时间压缩（与 TorchAx 对齐）
            if self.compress_time:
                B, T, H, W, C = hidden_states.shape
                # TorchAx 使用 F.avg_pool1d，这里用 reshape + mean 实现
                hidden_states = hidden_states.reshape(B * H * W, T, C)
                hidden_states = hidden_states.transpose(0, 2, 1)  # (B*H*W, C, T)
                
                if T % 2 == 1:
                    # 奇数帧：保留第一帧，avg pool 其余帧
                    first_frame = hidden_states[:, :, 0:1]
                    rest_frames = hidden_states[:, :, 1:]
                    if rest_frames.shape[2] > 0:
                        rest_frames = jnp.mean(
                            rest_frames.reshape(B*H*W, C, rest_frames.shape[2]//2, 2),
                            axis=-1
                        )
                    hidden_states = jnp.concatenate([first_frame, rest_frames], axis=2)
                else:
                    # 偶数帧：直接 avg pooling
                    hidden_states = jnp.mean(
                        hidden_states.reshape(B*H*W, C, T//2, 2),
                        axis=-1
                    )
                
                # Reshape back
                T_new = hidden_states.shape[2]
                hidden_states = hidden_states.transpose(0, 2, 1)  # (B*H*W, T_new, C)
                hidden_states = hidden_states.reshape(B, H, W, T_new, C)
                hidden_states = hidden_states.transpose(0, 3, 1, 2, 4)  # (B, T_new, H, W, C)
            
            # FLAX: 空间下采样（使用 2D 卷积）
            for downsampler in self.downsamplers:
                B, T, H, W, C = hidden_states.shape
                
                # FLAX: 手动添加 padding (0, 1, 0, 1)
                # TorchAx: hidden_states = F.pad(hidden_states, (0, 1, 0, 1))
                pad_width = [(0, 0), (0, 0), (0, 1), (0, 1), (0, 0)]
                hidden_states = jnp.pad(hidden_states, pad_width, mode='constant', constant_values=0)
                
                # Reshape to apply 2D conv: (B, T, H, W, C) → (B*T, H, W, C)
                _, _, H_padded, W_padded, _ = hidden_states.shape
                hidden_states = hidden_states.reshape(B * T, H_padded, W_padded, C)
                hidden_states = downsampler(hidden_states)
                _, H_new, W_new, _ = hidden_states.shape
                hidden_states = hidden_states.reshape(B, T, H_new, W_new, C)
        
        return hidden_states, feat_idx, feat_cache


# ==================== FLAX: CogVideoXMidBlock3D ====================
# 对应 TorchAx 版本的 CogVideoXMidBlock3D (autoencoder_kl_cogvideox_torchax.py:513-554)
#
# FLAX 改动说明：
# 1. 数据格式：TorchAx NCTHW -> Flax NTHWC
# 2. forward 签名与 TorchAx 一致：(hidden_states, temb, zq, feat_cache, feat_idx) -> (output, feat_idx, feat_cache)

class FlaxCogVideoXMidBlock3D(nnx.Module):
    r"""A 3D middle block used in the CogVideoX encoder/decoder.

    FLAX: 从 TorchAx 版本改造而来，主要差异是数据格式 NCTHW -> NTHWC。
    """
    
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
    
    # FLAX: forward 签名与 TorchAx 一致
    # TorchAx: forward(hidden_states, temb=None, zq=None, feat_cache=None, feat_idx=0) -> (output, feat_idx, feat_cache)
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        temb: Optional[jnp.ndarray] = None,
        zq: Optional[jnp.ndarray] = None,
        feat_cache: Optional[list] = None,
        feat_idx: int = 0,
        deterministic: bool = True,
    ):
        for resnet in self.resnets:
            hidden_states, feat_idx, feat_cache = resnet(
                hidden_states, temb, zq,
                feat_cache=feat_cache, feat_idx=feat_idx,
                deterministic=deterministic
            )
        return hidden_states, feat_idx, feat_cache


# ==================== FLAX: CogVideoXUpBlock3D ====================
# 对应 TorchAx 版本的 CogVideoXUpBlock3D (autoencoder_kl_cogvideox_torchax.py:560-654)
#
# FLAX 改动说明：
# 1. 数据格式：TorchAx NCTHW -> Flax NTHWC
# 2. forward 签名与 TorchAx 一致：(hidden_states, temb, zq, feat_cache, feat_idx) -> (output, feat_idx, feat_cache)
# 3. 使用 jax.image.resize 替代 F.interpolate

class FlaxCogVideoXUpBlock3D(nnx.Module):
    r"""A 3D up block used in the CogVideoX decoder.

    FLAX: 从 TorchAx 版本改造而来，主要差异是数据格式 NCTHW -> NTHWC。
    """
    
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
    
    # FLAX: forward 签名与 TorchAx 一致
    # TorchAx: forward(hidden_states, temb=None, zq=None, feat_cache=None, feat_idx=0) -> (output, feat_idx, feat_cache)
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        temb: Optional[jnp.ndarray] = None,
        zq: Optional[jnp.ndarray] = None,
        feat_cache: Optional[list] = None,
        feat_idx: int = 0,
        deterministic: bool = True,
    ):
        # FLAX: ResNet blocks（与 TorchAx 完全一致）
        for resnet in self.resnets:
            hidden_states, feat_idx, feat_cache = resnet(
                hidden_states, temb, zq,
                feat_cache=feat_cache, feat_idx=feat_idx,
                deterministic=deterministic
            )
        
        # FLAX: 上采样（与 TorchAx 对齐，使用 jax.image.resize 替代 F.interpolate）
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                B, T, H, W, C = hidden_states.shape
                
                # FLAX: compress_time 模式：时间 + 空间上采样
                # TorchAx: F.interpolate(x, scale_factor=2.0)
                if self.compress_time:
                    if T == 1:
                        # 单帧：1 -> 2（时间）+ 2x空间
                        hidden_states = jax.image.resize(hidden_states, (B, 2, H * 2, W * 2, C), method='nearest')
                    elif T > 1 and T % 2 == 1:
                        # 奇数帧：第一帧只做空间上采样，其余帧做时间+空间上采样
                        first_frame = hidden_states[:, 0, :, :, :]
                        rest_frames = hidden_states[:, 1:, :, :, :]
                        first_frame = jax.image.resize(first_frame, (B, H * 2, W * 2, C), method='nearest')
                        first_frame = first_frame[:, None, :, :, :]
                        rest_frames = jax.image.resize(rest_frames, (B, 2 * (T-1), H * 2, W * 2, C), method='nearest')
                        hidden_states = jnp.concatenate([first_frame, rest_frames], axis=1)
                    else:
                        # 偶数帧：全部做时间+空间上采样
                        hidden_states = jax.image.resize(hidden_states, (B, T * 2, H * 2, W * 2, C), method='nearest')
                else:
                    # FLAX: 非 compress_time：只做空间上采样
                    hidden_states = hidden_states.reshape(B * T, H, W, C)
                    hidden_states = jax.image.resize(hidden_states, (B * T, H * 2, W * 2, C), method='nearest')
                    hidden_states = hidden_states.reshape(B, T, H * 2, W * 2, C)
                
                # FLAX: 应用 2D 卷积到空间维度
                B, T_new, H_new, W_new, C = hidden_states.shape
                hidden_states = hidden_states.reshape(B * T_new, H_new, W_new, C)
                hidden_states = upsampler(hidden_states)
                _, H_final, W_final, _ = hidden_states.shape
                hidden_states = hidden_states.reshape(B, T_new, H_final, W_final, C)
        
        return hidden_states, feat_idx, feat_cache


# ==================== FLAX: CogVideoXEncoder3D ====================
# 对应 TorchAx 版本的 CogVideoXEncoder3D (autoencoder_kl_cogvideox_torchax.py:660-727)
#
# FLAX 改动说明：
# 1. 数据格式：TorchAx NCTHW -> Flax NTHWC
# 2. forward 签名与 TorchAx 一致：(sample, feat_cache) -> (output, feat_cache)

class FlaxCogVideoXEncoder3D(nnx.Module):
    r"""The Encoder for the CogVideoX VAE.

    FLAX: 从 TorchAx 版本改造而来，主要差异是数据格式 NCTHW -> NTHWC。
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
        self.norm_out = FlaxGroupNorm(num_groups=norm_num_groups, num_channels=block_out_channels[-1], epsilon=1e-6, rngs=rngs)
        self.conv_out = FlaxCogVideoXCausalConv3d(
            block_out_channels[-1],
            2 * out_channels,
            kernel_size=3,
            pad_mode=pad_mode,
            rngs=rngs
        )
    
    # FLAX: forward 签名与 TorchAx 一致
    # TorchAx: forward(sample, feat_cache=None) -> (output, feat_cache)
    def __call__(
        self,
        sample: jnp.ndarray,
        temb: Optional[jnp.ndarray] = None,
        feat_cache: Optional[list] = None,
        feat_idx: int = 0,
        deterministic: bool = True,
    ):
        # FLAX: 输入卷积（与 TorchAx 完全一致）
        hidden_states, feat_idx, feat_cache = self.conv_in(sample, feat_cache=feat_cache, feat_idx=feat_idx)
        
        # FLAX: Down blocks
        for down_block in self.down_blocks:
            hidden_states, feat_idx, feat_cache = down_block(
                hidden_states, temb, None,
                feat_cache=feat_cache, feat_idx=feat_idx,
                deterministic=deterministic
            )
        
        # FLAX: Mid block
        hidden_states, feat_idx, feat_cache = self.mid_block(
            hidden_states, temb, None,
            feat_cache=feat_cache, feat_idx=feat_idx,
            deterministic=deterministic
        )
        
        # FLAX: 输出层
        hidden_states = self.norm_out(hidden_states)
        hidden_states = jax.nn.silu(hidden_states)
        hidden_states, feat_idx, feat_cache = self.conv_out(
            hidden_states, feat_cache=feat_cache, feat_idx=feat_idx
        )
        
        return hidden_states, feat_cache


# ==================== FLAX: CogVideoXDecoder3D ====================
# 对应 TorchAx 版本的 CogVideoXDecoder3D (autoencoder_kl_cogvideox_torchax.py:733-811)
#
# FLAX 改动说明：
# 1. 数据格式：TorchAx NCTHW -> Flax NTHWC
# 2. forward 签名与 TorchAx 一致：(sample, zq, feat_cache) -> (output, feat_cache)

class FlaxCogVideoXDecoder3D(nnx.Module):
    r"""The Decoder for the CogVideoX VAE.

    FLAX: 从 TorchAx 版本改造而来，主要差异是数据格式 NCTHW -> NTHWC。
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
    
    # FLAX: forward 签名与 TorchAx 一致
    # TorchAx: forward(sample, zq, feat_cache=None) -> (output, feat_cache)
    def __call__(
        self,
        sample: jnp.ndarray,
        zq: jnp.ndarray,
        temb: Optional[jnp.ndarray] = None,
        feat_cache: Optional[list] = None,
        feat_idx: int = 0,
        deterministic: bool = True,
    ):
        # FLAX: 输入卷积（与 TorchAx 完全一致）
        hidden_states, feat_idx, feat_cache = self.conv_in(sample, feat_cache=feat_cache, feat_idx=feat_idx)
        
        # FLAX: Mid block（使用 sample 作为 zq）
        hidden_states, feat_idx, feat_cache = self.mid_block(
            hidden_states, temb, sample,
            feat_cache=feat_cache, feat_idx=feat_idx,
            deterministic=deterministic
        )
        
        # FLAX: Up blocks
        for up_block in self.up_blocks:
            hidden_states, feat_idx, feat_cache = up_block(
                hidden_states, temb, sample,
                feat_cache=feat_cache, feat_idx=feat_idx,
                deterministic=deterministic
            )
        
        # FLAX: 输出层
        hidden_states, feat_idx, feat_cache = self.norm_out(hidden_states, sample, feat_cache=feat_cache, feat_idx=feat_idx)
        hidden_states = jax.nn.silu(hidden_states)
        hidden_states, feat_idx, feat_cache = self.conv_out(hidden_states, feat_cache=feat_cache, feat_idx=feat_idx)
        
        return hidden_states, feat_cache


# ==================== FLAX: AutoencoderKLCogVideoX ====================
# 对应 TorchAx 版本的 AutoencoderKLCogVideoX (autoencoder_kl_cogvideox_torchax.py:817-1123)
#
# FLAX 改动说明：
# 1. 数据格式：TorchAx NCTHW -> Flax NTHWC
# 2. 配置类：TorchAx 使用 @register_to_config 装饰器，Flax 使用 dataclass
# 3. 缓存机制：使用 list-based feat_cache（与 TorchAx 完全一致）
# 4. from_pretrained：自定义权重转换逻辑（PyTorch -> JAX）

class FlaxAutoencoderKLCogVideoX(nnx.Module):
    r"""A VAE model with KL loss for encoding videos into latents and decoding latent representations into videos.

    FLAX: 从 TorchAx 版本改造而来。

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        config (`FlaxAutoencoderKLCogVideoXConfig`):
            Configuration object with all VAE hyperparameters.
        rngs (`nnx.Rngs`):
            Random number generators for parameter initialization.
        dtype (`jnp.dtype`, *optional*, defaults to `jnp.float32`):
            The dtype of the parameters.
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
        # Update: 配合 feat_cache 优化，可以尝试 batch=2
        self.num_latent_frames_batch_size = 2  # decode 时每批 2 帧
        self.num_sample_frames_batch_size = 8  # encode 时每批 8 帧
        
        # Precompute and cache conv counts for encoder and decoder for clear_cache speedup
        self._cached_conv_counts = {
            "decoder": self._count_causal_conv3d(self.decoder) if self.decoder is not None else 0,
            "encoder": self._count_causal_conv3d(self.encoder) if self.encoder is not None else 0,
        }

    def _count_causal_conv3d(self, module):
        """Count the number of CausalConv3d layers in a module."""
        count = 0
        # Use nnx.graph.iter_graph to traverse all submodules
        node_types = nnx.graph.iter_graph([module])
        for _, value in node_types:
            if isinstance(value, FlaxCogVideoXCausalConv3d):
                count += 1
        return count

    def clear_cache(self):
        """Initialize feat_cache for decoder and encoder.
        
        In Flax/NNX, we don't store the cache as an attribute to avoid static/dynamic type issues.
        Instead, we return initialized caches that should be passed as local variables.
        This method is kept for API compatibility but doesn't modify self.
        """
        pass

    def _init_feat_cache(self, mode="decoder"):
        """Initialize feature cache list."""
        count = self._cached_conv_counts.get(mode, 0)
        return [None] * count
    
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
        enc_feat_map = self._init_feat_cache("encoder")
        
        frame_batch_size = self.num_sample_frames_batch_size
        num_batches = max(num_frames // frame_batch_size, 1)
        # conv_cache = None  # Removed in favor of enc_feat_map
        enc = []
        
        for i in range(num_batches):
            remaining_frames = num_frames % frame_batch_size
            start_frame = frame_batch_size * i + (0 if i == 0 else remaining_frames)
            end_frame = frame_batch_size * (i + 1) + remaining_frames
            x_intermediate = x[:, start_frame:end_frame, :, :, :]
            
            x_intermediate, enc_feat_map = self.encoder(
                x_intermediate, feat_cache=enc_feat_map, deterministic=deterministic
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
        
        这是参考 WAN VAE 的逐帧解码实现，使用实例属性 _feat_map 管理缓存。
        每个 latent 帧独立处理并上采样到 temporal_compression_ratio 倍的视频帧。
        
        Args:
            z: Latent representation (B, T, H, W, C)
            zq: Spatial conditioning (same as z for CogVideoX)
            deterministic: Whether to use dropout
            
        Returns:
            Decoded video (B, T*temporal_compression_ratio, H, W, C)
        """
        import os
        
        # 内存监控开关（通过环境变量控制）
        # 使用方法: export JAX_MEMORY_DEBUG=1
        enable_memory_debug = os.getenv('JAX_MEMORY_DEBUG', '0') == '1'
        
        def get_memory_stats():
            """获取当前设备内存统计信息"""
            if not enable_memory_debug:
                return ""
            try:
                # 获取所有设备的内存统计
                for device in jax.devices():
                    stats = device.memory_stats()
                    if stats:
                        used_gb = stats.get('bytes_in_use', 0) / 1e9
                        limit_gb = stats.get('bytes_limit', 0) / 1e9
                        return f"{used_gb:.2f}GB / {limit_gb:.2f}GB"
            except:
                pass
            return "N/A"
        
        def log_memory(msg):
            """记录内存状态（仅在开启调试时）"""
            if enable_memory_debug:
                print(f"[内存] {msg}: {get_memory_stats()}")
        
        batch_size, num_frames, height, width, num_channels = z.shape
        
        if enable_memory_debug:
            print(f"\n[VAE Decode] 开始解码: {num_frames} latent frames -> {num_frames * 4} video frames")
        log_memory("解码前")
        
        if self.use_tiling and (width > self.tile_latent_min_width or height > self.tile_latent_min_height):
            return self.tiled_decode(z, zq, deterministic=deterministic)
        
        # 使用实例属性初始化缓存
        # self.clear_cache()  # Removed: we use local variable for cache
        feat_map = self._init_feat_cache("decoder")
        log_memory("初始化缓存后")
        
        # 应用 post_quant_conv 到整个 latent（如果存在）
        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)
            log_memory("post_quant_conv 后")
        
        # 逐帧解码（或按批次解码）
        decoded_frames_list = []
        frame_batch_size = self.num_latent_frames_batch_size
        num_batches = max(num_frames // frame_batch_size, 1)
        
        try:
            for i in range(num_batches):
                remaining_frames = num_frames % frame_batch_size
                start_frame = frame_batch_size * i + (0 if i == 0 else remaining_frames)
                end_frame = frame_batch_size * (i + 1) + remaining_frames
                
                # 提取当前批次 latent 帧
                z_frame = z[:, start_frame:end_frame, :, :, :]
                zq_frame = zq[:, start_frame:end_frame, :, :, :]
                
                # 使用实例属性缓存解码
                decoded_frame, feat_map = self.decoder(
                    z_frame, zq_frame,
                    feat_cache=feat_map,
                    deterministic=deterministic
                )
                
                decoded_frames_list.append(decoded_frame)
                
                if enable_memory_debug:
                    log_memory(f"解码批次 {i+1}/{num_batches} 后")
            
            log_memory("所有帧解码完成，准备拼接")
            
            # 拼接所有解码后的帧
            decoded = jnp.concatenate(decoded_frames_list, axis=1)
            log_memory("拼接完成")
            
            decoded = jnp.clip(decoded, min=-1.0, max=1.0)
            log_memory("clip 完成")
            
            return decoded
        finally:
            log_memory("开始清理缓存")
            # self.clear_cache() # Removed
            log_memory("清理缓存完成")
            if enable_memory_debug:
                print(f"[内存] 解码结束\n")
    
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
                enc_feat_map = self._init_feat_cache("encoder")
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
                    
                    tile, enc_feat_map = self.encoder(
                        tile, feat_cache=enc_feat_map, deterministic=deterministic
                    )
                    
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
                feat_map = self._init_feat_cache("decoder")
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
                    
                    tile, feat_map = self.decoder(
                        tile, tile_zq, feat_cache=feat_map, deterministic=deterministic
                    )
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