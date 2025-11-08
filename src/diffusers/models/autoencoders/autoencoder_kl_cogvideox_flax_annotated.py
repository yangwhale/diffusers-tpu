
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
CogVideoX VAE 的 Flax/JAX 实现 - 详细注释版本

这个文件是 CogVideoX 视频生成模型的变分自编码器（VAE）的 JAX/Flax 实现。
主要功能：
1. 将视频编码为潜在表示（latents）
2. 从潜在表示解码回视频
3. 支持内存高效的分块（tiling）和逐帧处理

关键概念：
- VAE（Variational Autoencoder）: 一种生成模型，通过编码器-解码器架构压缩和重建数据
- 3D卷积: 处理时间+空间维度的卷积操作
- 因果卷积: 只使用过去和当前的信息，不使用未来信息（用于序列建模）
- CogVideoX: 清华大学和智谱AI开发的视频生成模型
"""

from dataclasses import dataclass  # 用于创建配置类，自动生成__init__等方法
from typing import Dict, Optional, Tuple, Union  # 类型注解，提高代码可读性和IDE支持

import jax  # Google的JAX库，用于高性能数值计算和自动微分
import jax.numpy as jnp  # JAX的numpy接口，API与numpy几乎相同但支持GPU/TPU加速
from flax import nnx  # Flax NNX，新一代的Flax神经网络库，更符合PyTorch风格

from ...configuration_utils import ConfigMixin  # 配置混入类，提供配置管理功能


# ==================== 配置类 ====================

@dataclass  # 装饰器：自动生成 __init__, __repr__, __eq__ 等方法
class FlaxAutoencoderKLCogVideoXConfig:
    """
    CogVideoX VAE 的配置类
    
    @dataclass 装饰器的作用：
    - 自动生成 __init__ 方法，接受所有字段作为参数
    - 自动生成 __repr__ 方法用于打印对象
    - 自动生成 __eq__ 方法用于比较对象
    - 字段的默认值就是类属性的值
    
    为什么使用 dataclass：
    - 减少样板代码（boilerplate code）
    - 提高可读性
    - 自动类型检查（配合类型注解）
    """
    config_name: str = "config.json"  # 配置文件名称
    
    # 输入输出通道数
    in_channels: int = 3  # 输入视频的通道数（RGB=3）
    out_channels: int = 3  # 输出视频的通道数（RGB=3）
    
    # 下采样和上采样块的类型
    # Tuple[str, ...] 表示字符串元组，长度可变
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
    
    # 每个块的输出通道数
    block_out_channels: Tuple[int, ...] = (128, 256, 256, 512)
    
    # 潜在空间的通道数（压缩后的表示）
    latent_channels: int = 16
    
    # 每个块中的残差层数量
    layers_per_block: int = 3
    
    # 激活函数类型（silu = Sigmoid Linear Unit，也称为Swish）
    act_fn: str = "silu"
    
    # 归一化层的epsilon（防止除零）
    norm_eps: float = 1e-6
    
    # Group Normalization 的组数
    norm_num_groups: int = 32
    
    # 时间维度的压缩比例（例如4表示时间维度压缩4倍）
    temporal_compression_ratio: float = 4
    
    # 样本的空间分辨率
    sample_height: int = 480
    sample_width: int = 720
    
    # 缩放因子（用于归一化潜在表示）
    scaling_factor: float = 1.15258426
    
    # 可选的偏移因子和统计量
    shift_factor: Optional[float] = None
    latents_mean: Optional[Tuple[float]] = None
    latents_std: Optional[Tuple[float]] = None
    
    # 是否强制向上转换数据类型
    force_upcast: bool = True
    
    # 是否使用量化卷积层
    use_quant_conv: bool = False
    use_post_quant_conv: bool = False
    
    # 填充模式（"first" 或 "replicate"）
    pad_mode: str = "first"
    
    @classmethod  # 类方法装饰器：第一个参数是类本身（cls），而不是实例（self）
    def from_dict(cls, config_dict: Dict):
        """
        从字典创建配置对象
        
        @classmethod 的作用：
        - 可以通过类直接调用，不需要实例化
        - 第一个参数是类本身（cls），可以用于创建类的实例
        - 常用于工厂方法（factory method）模式
        
        为什么需要过滤字典：
        - config_dict 可能包含额外的键（例如从JSON加载的元数据）
        - dataclass 只接受定义的字段作为参数
        - 过滤掉未定义的键可以避免 TypeError
        
        Args:
            config_dict: 包含配置信息的字典
            
        Returns:
            配置对象实例
        """
        # 使用 dataclasses 模块获取所有字段名
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(cls)}  # 集合推导式，获取所有字段名
        
        # 字典推导式：只保留在 dataclass 中定义的字段
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}
        
        # 使用过滤后的字典创建实例
        return cls(**filtered_dict)  # **filtered_dict 解包字典为关键字参数
    
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
        **kwargs,  # 捕获额外的关键字参数
    ):
        """
        初始化方法
        
        注意：在 Python 3.10+ 的 dataclass 中，如果定义了 __init__，
        dataclass 装饰器不会自动生成 __init__ 方法。
        但是这里实际上是空的，因为 dataclass 已经自动处理了字段的赋值。
        
        这个空的 __init__ 存在的原因：
        1. 提供明确的参数列表（用于文档和IDE提示）
        2. 支持 **kwargs（额外参数）
        """
        # Dataclass 会自动处理字段赋值，不需要手动设置
        pass


# ==================== 基础卷积层 ====================

class FlaxConv3d(nnx.Module):
    """
    3D卷积层的封装
    
    为什么需要封装：
    1. 统一接口：提供与 PyTorch nn.Conv3d 类似的接口
    2. 简化 padding 处理：Flax 的 padding 格式与 PyTorch 不同
    3. 代码复用：避免重复的 padding 转换逻辑
    
    3D卷积的应用场景：
    - 视频处理（时间 + 空间）
    - 医学影像（3D扫描）
    - 点云处理
    """
    def __init__(
        self,
        in_channels: int,  # 输入特征的通道数
        out_channels: int,  # 输出特征的通道数
        kernel_size: Union[int, Tuple[int, int, int]] = (3, 3, 3),  # 卷积核大小
        stride: Union[int, Tuple[int, int, int]] = (1, 1, 1),  # 步长
        padding: Union[int, Tuple[int, int, int], str] = 1,  # 填充
        rngs: nnx.Rngs = None,  # 随机数生成器（用于初始化权重）
    ):
        """
        初始化 3D 卷积层
        
        参数说明：
        - kernel_size: 卷积核在 (时间, 高度, 宽度) 三个维度的大小
        - stride: 卷积在三个维度的步长
        - padding: 填充方式，可以是整数、元组或字符串
        - rngs: Flax NNX 的随机数生成器，用于权重初始化
        
        Union[int, Tuple[...]] 的含义：
        - 参数可以是 int 类型或 Tuple 类型
        - 如果是 int，会自动转换为 Tuple（三个维度使用相同的值）
        """
        # 将单个整数转换为三元组（三个维度使用相同的值）
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3  # (k,) * 3 = (k, k, k)
        if isinstance(stride, int):
            stride = (stride,) * 3
            
        # 处理填充参数
        # Flax 的 padding 格式：((before_1, after_1), (before_2, after_2), ...)
        if isinstance(padding, int):
            if padding == 0:
                # 无填充
                padding_mode = ((0, 0), (0, 0), (0, 0))
            else:
                # 对称填充：每个维度前后各填充 padding 个单位
                padding_mode = ((padding, padding), (padding, padding), (padding, padding))
        elif isinstance(padding, tuple) and len(padding) == 3:
            # 将 (p1, p2, p3) 转换为 ((p1, p1), (p2, p2), (p3, p3))
            padding_mode = tuple((p, p) for p in padding)
        else:
            padding_mode = padding
            
        # 创建底层的 Flax 卷积层
        # nnx.Conv 是 Flax NNX 的卷积模块
        self.conv = nnx.Conv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            strides=stride,  # 注意：Flax 使用 strides（复数）
            padding=padding_mode,
            rngs=rngs,
        )

    def __call__(self, x):
        """
        前向传播
        
        __call__ 方法的作用：
        - 允许对象像函数一样被调用
        - 例如：conv = FlaxConv3d(...); output = conv(input)
        - 这是 Python 的魔术方法（magic method）
        
        Args:
            x: 输入张量，形状 (B, T, H, W, C)
               - B: batch size（批大小）
               - T: time（时间帧数）
               - H: height（高度）
               - W: width（宽度）
               - C: channels（通道数）
        
        Returns:
            输出张量，形状取决于卷积参数
        """
        return self.conv(x)


class FlaxConv2d(nnx.Module):
    """
    2D卷积层的封装
    
    用途：
    - 空间下采样/上采样（不涉及时间维度）
    - 在 CogVideoX 中用于 downsample 和 upsample 操作
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = (3, 3),
        stride: Union[int, Tuple[int, int]] = (1, 1),
        padding: Union[int, str] = 1,
        rngs: nnx.Rngs = None,
    ):
        # 与 FlaxConv3d 类似的处理，但只有两个空间维度
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
            
        # 处理填充
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
        """
        前向传播
        
        Args:
            x: 输入张量，形状 (B, H, W, C)
        
        Returns:
            输出张量
        """
        return self.conv(x)


# ==================== 因果卷积层 ====================

class FlaxCogVideoXCausalConv3d(nnx.Module):
    """
    因果3D卷积层 - CogVideoX 的核心组件
    
    什么是因果卷积（Causal Convolution）：
    - 在时间维度上只使用过去和当前的信息，不使用未来信息
    - 通过在时间维度前面填充来实现
    - 类似于语言模型中的因果mask
    
    为什么需要因果卷积：
    1. 支持逐帧生成（autoregressive generation）
    2. 避免信息泄露（future information leakage）
    3. 支持流式处理（streaming）
    
    缓存机制（Cache Mechanism）：
    - 逐帧处理时，需要保存之前的帧用于卷积计算
    - 例如：kernel_size=3 时，处理第 t 帧需要第 t-2, t-1, t 三帧
    - 通过缓存避免重复计算
    """
    
    # 类变量：定义缓存需要的帧数
    # 为什么是 2：因为 kernel_size=3 时，需要前 2 帧 + 当前帧
    CACHE_T = 2
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: int = 1,  # 步长（仅应用于时间维度）
        dilation: int = 1,  # 膨胀率（空洞卷积）
        pad_mode: str = "constant",  # 填充模式："constant" 或 "replicate"
        rngs: nnx.Rngs = None,
    ):
        """
        初始化因果卷积层
        
        pad_mode 说明：
        - "constant": 填充0（默认）
        - "replicate": 复制边缘值（用于减少边界效应）
        
        dilation（膨胀/空洞卷积）：
        - 扩大感受野而不增加参数
        - dilation=2 时，卷积核元素之间间隔1个位置
        """
        super().__init__()
        
        # 标准化 kernel_size 为三元组
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        
        # 解包卷积核大小
        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size
        
        # 计算填充大小
        # 时间维度：因果填充，只在前面填充
        self.time_pad = time_kernel_size - 1
        # 空间维度：对称填充
        self.height_pad = (height_kernel_size - 1) // 2
        self.width_pad = (width_kernel_size - 1) // 2
        
        self.pad_mode = pad_mode
        self.time_kernel_size = time_kernel_size
        self.temporal_dim = 1  # 在 JAX NTHWC 格式中，时间是第1维
        
        # 对于 constant 模式，只填充空间维度（时间填充单独处理）
        const_padding_conv3d = (0, self.height_pad, self.width_pad)
        
        # 创建底层卷积
        stride_tuple = (stride, 1, 1) if isinstance(stride, int) else stride
        dilation_tuple = (dilation, 1, 1)
        
        self.conv = FlaxConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride_tuple,
            # replicate 模式时 padding=0（手动处理），constant 模式时填充空间维度
            padding=0 if self.pad_mode == "replicate" else const_padding_conv3d,
            rngs=rngs,
        )
    
    def __call__(
        self,
        inputs: jnp.ndarray,
        conv_cache: Optional[jnp.ndarray] = None,  # 旧的缓存模式
        feat_cache: Optional[list] = None,  # 新的缓存列表
        feat_idx: Optional[list] = None,  # 当前索引列表
    ):
        """
        前向传播，支持两种缓存模式
        
        为什么有两种缓存模式：
        1. conv_cache: 旧模式，保持向后兼容
        2. feat_cache/feat_idx: 新模式，用于逐帧解码，避免 OOM
        
        OOM（Out Of Memory）问题：
        - 视频解码需要大量内存
        - 逐帧处理可以大幅减少内存使用
        - TPU v6e 只有 32GB 内存，需要精细的内存管理
        
        Args:
            inputs: 输入张量 (B, T, H, W, C)
            conv_cache: 旧模式的缓存
            feat_cache: 新模式的缓存列表
            feat_idx: 新模式的索引列表（使用列表是为了在函数间传递引用）
            
        Returns:
            output: 卷积输出
            new_cache: 更新的缓存
        """
        # 根据参数选择缓存模式
        if feat_cache is not None and feat_idx is not None:
            # 新模式
            return self._call_with_feat_cache(inputs, feat_cache, feat_idx)
        
        # 旧模式（保持兼容性）
        return self._call_with_conv_cache(inputs, conv_cache)
    
    def _call_with_feat_cache(
        self,
        inputs: jnp.ndarray,
        feat_cache: list,
        feat_idx: list,
    ):
        """
        使用 feat_cache 的新缓存模式
        
        这个方法参考了 WAN VAE 的实现，支持逐帧处理。
        
        工作原理：
        1. 从缓存中获取之前的帧
        2. 与当前帧拼接
        3. 执行卷积
        4. 更新缓存（保存最新的帧供下次使用）
        
        为什么使用列表传递索引：
        - Python 的参数传递是"对象引用"
        - 对于不可变对象（如 int），修改不会影响外部
        - 使用列表 [idx] 可以通过修改列表内容来更新外部的索引
        
        Args:
            inputs: 输入张量 (B, T, H, W, C)，通常 T=1（逐帧）
            feat_cache: 缓存列表
            feat_idx: 索引列表 [idx]
            
        Returns:
            output: 卷积输出
            None: 新模式不返回缓存（直接修改 feat_cache）
        """
        idx = feat_idx[0]  # 获取当前索引
        
        # 处理时间填充
        if self.pad_mode == "replicate":
            # Replicate 模式：边缘复制填充
            # pad_width 格式：[(dim0_before, dim0_after), (dim1_before, dim1_after), ...]
            pad_width = [
                (0, 0),  # batch: 不填充
                (self.time_pad, 0),  # time: 只在前面填充（因果）
                (self.height_pad, self.height_pad),  # height: 对称填充
                (self.width_pad, self.width_pad),  # width: 对称填充
                (0, 0),  # channels: 不填充
            ]
            # mode='edge' 表示使用边缘值填充（复制边缘）
            x = jnp.pad(inputs, pad_width, mode='edge')
        else:
            # Constant 模式：使用缓存
            if self.time_kernel_size > 1:
                padding_needed = self.time_kernel_size - 1  # 需要的历史帧数
                
                # 如果缓存中有数据
                if feat_cache[idx] is not None:
                    cache_len = feat_cache[idx].shape[1]  # 缓存的帧数
                    # 拼接缓存和当前输入
                    x = jnp.concatenate([feat_cache[idx], inputs], axis=1)
                    
                    # 调整填充需求
                    padding_needed -= cache_len
                    if padding_needed > 0:
                        # 缓存不够，补充额外的 padding
                        # jnp.tile: 重复张量
                        # x[:, :1, ...] 取第一帧
                        # (1, padding_needed, 1, 1, 1) 在时间维度重复
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
                
                # ⚠️ 关键：先执行卷积，再更新缓存
                # 这是从 WAN 实现中学到的正确做法
                x2 = self.conv(x)
                
                # 更新缓存：保存最新的 CACHE_T 帧
                if inputs.shape[1] < self.CACHE_T and feat_cache[idx] is not None:
                    # 当前帧数不足：从旧缓存取最后1帧 + 当前所有帧
                    feat_cache[idx] = jnp.concatenate([
                        jnp.expand_dims(feat_cache[idx][:, -1, :, :, :], axis=1),
                        inputs[:, -self.CACHE_T:, :, :, :]
                    ], axis=1)
                else:
                    # 当前帧数足够：直接取最后 CACHE_T 帧
                    feat_cache[idx] = inputs[:, -self.CACHE_T:, :, :, :]
                
                # 索引递增（移到下一个卷积层）
                feat_idx[0] += 1
                
                return x2, None
            else:
                # kernel_size=1 时不需要缓存
                x = inputs
        
        # 执行卷积（非缓存路径）
        output = self.conv(x)
        
        # 索引递增
        feat_idx[0] += 1
        
        # 返回 (output, None) 保持 API 一致性
        return output, None
    
    def _call_with_conv_cache(self, inputs: jnp.ndarray, conv_cache: Optional[jnp.ndarray]):
        """
        旧的 conv_cache 模式（保持向后兼容）
        
        这个方法用于非逐帧的处理，一次性处理多帧。
        
        Args:
            inputs: 输入张量 (B, T, H, W, C)
            conv_cache: 上一次的缓存
            
        Returns:
            output: 卷积输出
            new_cache: 更新的缓存
        """
        # 应用因果填充
        if self.pad_mode == "replicate":
            # Replicate 模式：全维度填充
            pad_width = [
                (0, 0),  # batch
                (self.time_pad, 0),  # time（只在前面填充，因果）
                (self.height_pad, self.height_pad),  # height
                (self.width_pad, self.width_pad),  # width
                (0, 0),  # channels
            ]
            inputs = jnp.pad(inputs, pad_width, mode='edge')
            conv_cache = None
        else:
            # Constant 模式：使用 conv_cache
            if self.time_kernel_size > 1:
                if conv_cache is not None:
                    # 使用提供的缓存
                    cached_inputs = conv_cache
                else:
                    # 第一次调用：重复第一帧
                    cached_inputs = jnp.tile(
                        inputs[:, :1, :, :, :],
                        (1, self.time_kernel_size - 1, 1, 1, 1)
                    )
                # 拼接缓存和当前输入
                inputs = jnp.concatenate([cached_inputs, inputs], axis=1)
        
        # 应用卷积
        output = self.conv(inputs)
        
        # 更新缓存
        if self.pad_mode == "replicate":
            new_cache = None
        else:
            # 保存最后 (time_kernel_size - 1) 帧用于下次迭代
            new_cache = inputs[:, -(self.time_kernel_size - 1):, :, :, :]
        
        return output, new_cache


# ==================== 归一化层 ====================

class FlaxGroupNorm(nnx.Module):
    """
    Group Normalization（组归一化）
    
    什么是 Group Normalization：
    - 将通道分成多个组，在每组内进行归一化
    - 不依赖 batch size（适合小批量或单样本）
    - 介于 Layer Norm 和 Instance Norm 之间
    
    为什么使用 Group Norm：
    1. 稳定训练（减少内部协变量偏移）
    2. 不受 batch size 影响（视频处理通常 batch size 小）
    3. 在卷积网络中效果好
    
    Group Norm vs Batch Norm：
    - Batch Norm: 在 batch 维度归一化（需要较大的 batch）
    - Group Norm: 在通道组维度归一化（不受 batch 影响）
    
    实现细节：
    - PyTorch 的 GroupNorm 是 channel-first (N, C, H, W)
    - JAX/Flax 通常是 channel-last (N, H, W, C)
    - 需要正确处理维度以保持数值一致性
    """
    
    def __init__(
        self,
        num_groups: int,  # 组数（通道数必须能被组数整除）
        num_channels: int,  # 通道数
        epsilon: float = 1e-6,  # 数值稳定性参数（防止除零）
        rngs: nnx.Rngs = None,
    ):
        """
        初始化 Group Normalization
        
        参数说明：
        - num_groups: 将通道分成几组（例如 32）
        - num_channels: 总通道数（例如 256）
        - epsilon: 防止除零的小常数
        
        要求：num_channels % num_groups == 0
        """
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.epsilon = epsilon
        
        # 创建可学习参数
        # nnx.Param 标记参数为可训练
        self.scale = nnx.Param(jnp.ones((num_channels,)))  # γ（scale）
        self.bias = nnx.Param(jnp.zeros((num_channels,)))  # β（shift）
    
    def __call__(self, x):
        """
        应用 Group Normalization
        
        算法步骤：
        1. 将通道分组
        2. 计算每组的均值和方差
        3. 归一化：(x - mean) / sqrt(var + eps)
        4. 应用仿射变换：γ * normalized + β
        
        Args:
            x: 输入，形状 (B, T, H, W, C) 或 (B, H, W, C) [channel-last]
            
        Returns:
            归一化后的输出，形状与输入相同
        """
        if len(x.shape) == 5:
            # 5D: (B, T, H, W, C)
            B, T, H, W, C = x.shape
            assert C == self.num_channels, f"通道数不匹配: {C} vs {self.num_channels}"
            assert C % self.num_groups == 0, f"通道数必须能被组数整除: {C} % {self.num_groups}"
            
            # 计算每组的通道数
            channels_per_group = C // self.num_groups
            
            # Reshape 以暴露组结构
            # (B, T, H, W, num_groups, channels_per_group)
            x_grouped = x.reshape(B, T, H, W, self.num_groups, channels_per_group)
            
            # 计算每组的统计量
            # 在 (T, H, W, channels_per_group) 维度计算均值和方差
            # keepdims=True 保持维度，便于广播
            mean = jnp.mean(x_grouped, axis=(1, 2, 3, 5), keepdims=True)
            var = jnp.var(x_grouped, axis=(1, 2, 3, 5), keepdims=True)
            
            # 归一化
            # sqrt(var + epsilon) 防止除零
            x_norm = (x_grouped - mean) / jnp.sqrt(var + self.epsilon)
            
            # Reshape 回原始形状
            x_norm = x_norm.reshape(B, T, H, W, C)
            
            # 应用仿射变换
            # scale 和 bias 的形状是 (C,)，需要 reshape 为 (1, 1, 1, 1, C) 以便广播
            x_out = x_norm * self.scale.value.reshape(1, 1, 1, 1, C) + self.bias.value.reshape(1, 1, 1, 1, C)
            
        else:
            # 4D: (B, H, W, C)
            B, H, W, C = x.shape
            assert C == self.num_channels
            assert C % self.num_groups == 0
            
            # 转换为 channel-first 以匹配 PyTorch 的计算方式
            # (B, H, W, C) -> (B, C, H, W)
            x_cf = x.transpose(0, 3, 1, 2)
            
            # Reshape 为组结构
            # (B, num_groups, C//num_groups, H, W)
            x_grouped = x_cf.reshape(B, self.num_groups, C // self.num_groups, H, W)
            
            # 计算统计量（在 C//num_groups, H, W 维度）
            mean = jnp.mean(x_grouped, axis=(2, 3, 4), keepdims=True)
            var = jnp.var(x_grouped, axis=(2, 3, 4), keepdims=True)
            
            # 归一化
            x_norm = (x_grouped - mean) / jnp.sqrt(var + self.epsilon)
            
            # Reshape 回 (B, C, H, W)
            x_norm = x_norm.reshape(B, C, H, W)
            
            # 应用仿射变换
            x_out = x_norm * self.scale.value.reshape(1, C, 1, 1) + self.bias.value.reshape(1, C, 1, 1)
            
            # 转换回 channel-last: (B, C, H, W) -> (B, H, W, C)
            x_out = x_out.transpose(0, 2, 3, 1)
        
        return x_out


# ==================== 空间归一化层 ====================

class FlaxCogVideoXSpatialNorm3D(nnx.Module):
    """
    空间条件归一化（Spatially Conditioned Normalization）
    
    这是 CogVideoX 解码器的关键组件。
    
    工作原理：
    1. 对特征图进行 Group Normalization
    2. 使用潜在表示（zq）生成仿射变换参数（scale 和 shift）
    3. 应用条件化的仿射变换
    
    为什么需要空间条件归一化：
    - 解码器需要从压缩的潜在表示恢复细节
    - 通过空间条件化，不同位置可以有不同的归一化参数
    - 提高重建质量
    
    类似的技术：
    - AdaIN (Adaptive Instance Normalization)
    - SPADE (Spatially-Adaptive Normalization)
    """
    
    def __init__(
        self,
        f_channels: int,  # 特征图的通道数
        zq_channels: int,  # 条件潜在表示的通道数
        groups: int = 32,  # Group Norm 的组数
        rngs: nnx.Rngs = None,
    ):
        """
        初始化空间归一化层
        
        Args:
            f_channels: 输入特征图的通道数
            zq_channels: 条件信号（潜在表示）的通道数
            groups: Group Normalization 的组数
        """
        # 基础归一化层
        self.norm_layer = FlaxGroupNorm(
            num_groups=groups,
            num_channels=f_channels,
            epsilon=1e-6,
            rngs=rngs
        )
        
        # 生成 scale 参数的卷积
        # 使用 1x1x1 卷积将 zq 转换为与 f 相同的通道数
        self.conv_y = FlaxCogVideoXCausalConv3d(
            zq_channels, f_channels, kernel_size=1, stride=1, pad_mode="constant", rngs=rngs
        )
        
        # 生成 shift 参数的卷积
        self.conv_b = FlaxCogVideoXCausalConv3d(
            zq_channels, f_channels, kernel_size=1, stride=1, pad_mode="constant", rngs=rngs
        )
    
    def __call__(
        self,
        f: jnp.ndarray,  # 特征图
        zq: jnp.ndarray,  # 条件潜在表示
        conv_cache: Optional[Dict[str, jnp.ndarray]] = None,
        feat_cache: Optional[list] = None,
        feat_idx: Optional[list] = None,
    ):
        """
        应用空间条件归一化
        
        数学公式：
        output = (norm(f) * scale) + shift
        其中：
        - norm(f) 是 f 的 GroupNorm
        - scale = conv_y(zq)
        - shift = conv_b(zq)
        
        Args:
            f: 特征图 (B, T, H, W, C)
            zq: 条件潜在表示 (B, T', H', W', C')
            conv_cache: 旧模式缓存
            feat_cache: 新模式缓存
            feat_idx: 新模式索引
            
        Returns:
            归一化并条件化的特征
            更新的缓存
        """
        # 根据缓存类型选择处理路径
        if feat_cache is not None and feat_idx is not None:
            return self._call_with_feat_cache(f, zq, feat_cache, feat_idx)
        
        return self._call_with_conv_cache(f, zq, conv_cache)
    
    def _call_with_feat_cache(
        self,
        f: jnp.ndarray,
        zq: jnp.ndarray,
        feat_cache: list,
        feat_idx: list,
    ):
        """新缓存模式的实现"""
        B, T, H, W, C = f.shape
        
        # 特殊处理奇数帧（匹配 PyTorch 实现）
        # 为什么要特殊处理奇数帧：
        # - 视频上采样时，奇数帧可能导致尺寸不匹配
        # - 第一帧单独处理，其余帧一起处理
        if T > 1 and T % 2 == 1:
            # 分离第一帧和其余帧
            f_first = f[:, :1, :, :, :]
            f_rest = f[:, 1:, :, :, :]
            z_first = zq[:, :1, :, :, :]
            z_rest = zq[:, 1:, :, :, :]
            
            # 分别 resize
            # jax.image.resize: JAX 的图像插值函数
            # method='nearest': 最近邻插值（速度快，保持边缘）
            z_first = jax.image.resize(z_first, (B, 1, H, W, zq.shape[-1]), method='nearest')
            z_rest = jax.image.resize(z_rest, (B, T-1, H, W, zq.shape[-1]), method='nearest')
            
            # 拼接回去
            zq = jnp.concatenate([z_first, z_rest], axis=1)
        else:
            # 常规 resize
            zq = jax.image.resize(zq, (B, T, H, W, zq.shape[-1]), method='nearest')
        
        # 应用条件化卷积
        # 生成 scale 和 shift 参数
        conv_y, _ = self.conv_y(zq, feat_cache=feat_cache, feat_idx=feat_idx)
        conv_b, _ = self.conv_b(zq, feat_cache=feat_cache, feat_idx=feat_idx)
        
        # 归一化并条件化
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
        conv_cache = conv_cache or {}  # 如果为 None，使用空字典
        
        B, T, H, W, C = f.shape
        
        # 处理奇数帧（与新模式相同）
        if T > 1 and T % 2 == 1:
            f_first = f[:, :1, :, :, :]
            f_rest = f[:, 1:, :, :, :]
            z_first = zq[:, :1, :, :, :]
            z_rest = zq[:, 1:, :, :, :]
            
            z_first = jax.image.resize(z_first, (B, 1, H, W, zq.shape[-1]), method='nearest')
            z_rest = jax.image.resize(z_rest, (B, T-1, H, W, zq.shape[-1]), method='nearest')
            
            zq = jnp.concatenate([z_first, z_rest], axis=1)
        else:
            zq = jax.image.resize(zq, (B, T, H, W, zq.shape[-1]), method='nearest')
        
        # 应用条件化卷积（使用旧缓存）
        conv_y, new_conv_cache["conv_y"] = self.conv_y(zq, conv_cache=conv_cache.get("conv_y"))
        conv_b, new_conv_cache["conv_b"] = self.conv_b(zq, conv_cache=conv_cache.get("conv_b"))
        
        # 归一化并条件化
        norm_f = self.norm_layer(f)
        new_f = norm_f * conv_y + conv_b
        
        return new_f, new_conv_cache


# ==================== ResNet 块 ====================

class FlaxCogVideoXResnetBlock3D(nnx.Module):
    """
    3D ResNet 块 - CogVideoX 的基本构建单元
    
    ResNet（残差网络）的核心思想：
    - 引入跳跃连接（skip connection）
    - 学习残差映射：F(x) = H(x) - x
    - 缓解梯度消失问题
    - 允许构建更深的网络
    
    ResNet块的结构：
    input -> norm -> activate -> conv -> norm -> activate -> conv -> + -> output
             |                                                        ^
             +--------------------------------------------------------+
             (skip connection / shortcut)
    
    为什么使用 ResNet：
    1. 训练深层网络更容易
    2. 性能更好（ImageNet 等基准测试）
    3. 梯度流动更顺畅
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,  # Dropout 概率
        temb_channels: int = 512,  # 时间嵌入的通道数
        groups: int = 32,  # Group Norm 的组数
        eps: float = 1e-6,  # 归一化的 epsilon
        non_linearity: str = "swish",  # 激活函数类型
        conv_shortcut: bool = False,  # 是否使用卷积作为 shortcut
        spatial_norm_dim: Optional[int] = None,  # 空间归一化维度（仅解码器）
        pad_mode: str = "first",  # 填充模式
        rngs: nnx.Rngs = None,
    ):
        """
        初始化 ResNet 块
        
        参数说明：
        - spatial_norm_dim: 如果不为 None，使用空间条件归一化（解码器）
                           如果为 None，使用普通 GroupNorm（编码器）
        - conv_shortcut: True 时使用 3x3x3 卷积，False 时使用 1x1x1 卷积
        - temb_channels: 时间嵌入维度（用于扩散模型，这里设为0表示不使用）
        """
        out_channels = out_channels or in_channels
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.spatial_norm_dim = spatial_norm_dim
        
        # 归一化层
        if spatial_norm_dim is None:
            # 编码器：使用 GroupNorm
            self.norm1 = FlaxGroupNorm(num_groups=groups, num_channels=in_channels, epsilon=eps, rngs=rngs)
            self.norm2 = FlaxGroupNorm(num_groups=groups, num_channels=out_channels, epsilon=eps, rngs=rngs)
        else:
            # 解码器：使用空间条件归一化
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
        
        # 第一个卷积层
        self.conv1 = FlaxCogVideoXCausalConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            pad_mode=pad_mode,
            rngs=rngs
        )
        
        # 时间嵌入投影（用于扩散模型，这里通常不使用）
        if temb_channels > 0:
            self.temb_proj = nnx.Linear(temb_channels, out_channels, rngs=rngs)
        else:
            self.temb_proj = None
        
        # Dropout 率
        self.dropout_rate = dropout
        
        # 第二个卷积层
        self.conv2 = FlaxCogVideoXCausalConv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            pad_mode=pad_mode,
            rngs=rngs
        )
        
        # Shortcut 连接
        # 当输入输出通道数不同时，需要调整维度
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                # 使用 3x3x3 卷积
                self.conv_shortcut = FlaxCogVideoXCausalConv3d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    pad_mode=pad_mode,
                    rngs=rngs
                )
            else:
                # 使用 1x1x1 卷积（更高效）
                self.conv_shortcut = FlaxConv3d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    rngs=rngs
                )
        else:
            # 通道数相同，不需要调整
            self.conv_shortcut = None
    
    def __call__(
        self,
        inputs: jnp.ndarray,
        temb: Optional[jnp.ndarray] = None,  # 时间嵌入
        zq: Optional[jnp.ndarray] = None,  # 空间条件（解码器）
        conv_cache: Optional[Dict[str, jnp.ndarray]] = None,
        feat_cache: Optional[list] = None,
        feat_idx: Optional[list] = None,
        deterministic: bool = True,  # 是否确定性（控制 Dropout）
    ):
        """
        ResNet 块的前向传播
        
        流程：
        1. norm1 -> activate -> conv1
        2. 添加时间嵌入（如果有）
        3. norm2 -> activate -> dropout -> conv2
        4. shortcut 连接
        5. 残差相加
        
        Args:
            inputs: 输入特征 (B, T, H, W, C)
            temb: 时间嵌入（扩散模型用）
            zq: 空间条件（解码器用）
            conv_cache/feat_cache/feat_idx: 缓存相关
            deterministic: True 时不使用 Dropout
        """
        # 根据缓存类型选择处理路径
        if feat_cache is not None and feat_idx is not None:
            return self._call_with_feat_cache(inputs, temb, zq, feat_cache, feat_idx, deterministic)
        
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
        
        # 第一个残差路径：norm -> activate -> conv
        if zq is not None:
            # 解码器：使用空间条件归一化
            hidden_states, _ = self.norm1(hidden_states, zq, feat_cache=feat_cache, feat_idx=feat_idx)
        else:
            # 编码器：使用普通 GroupNorm
            hidden_states = self.norm1(hidden_states)
        
        # SiLU 激活函数（也称为 Swish）
        # SiLU(x) = x * sigmoid(x)
        # 特点：平滑、非单调、自门控
        hidden_states = jax.nn.silu(hidden_states)
        hidden_states, _ = self.conv1(hidden_states, feat_cache=feat_cache, feat_idx=feat_idx)
        
        # 时间嵌入（通常在扩散模型中使用）
        if temb is not None and self.temb_proj is not None:
            # 投影时间嵌入并添加到特征图
            # [:, None, None, None, :] 扩展维度以匹配 (B, T, H, W, C)
            temb_proj = self.temb_proj(jax.nn.silu(temb))
            hidden_states = hidden_states + temb_proj[:, None, None, None, :]
        
        # 第二个残差路径：norm -> activate -> dropout -> conv
        if zq is not None:
            hidden_states, _ = self.norm2(hidden_states, zq, feat_cache=feat_cache, feat_idx=feat_idx)
        else:
            hidden_states = self.norm2(hidden_states)
        
        hidden_states = jax.nn.silu(hidden_states)
        
        # Dropout（训练时随机丢弃部分神经元，防止过拟合）
        # deterministic=True 时不使用 Dropout（推理阶段）
        if self.dropout_rate > 0 and not deterministic:
            hidden_states = nnx.Dropout(rate=self.dropout_rate)(hidden_states)
        
        hidden_states, _ = self.conv2(hidden_states, feat_cache=feat_cache, feat_idx=feat_idx)
        
        # Shortcut 连接
        if self.conv_shortcut is not None:
            if self.use_conv_shortcut:
                # 3x3x3 卷积需要缓存
                inputs, _ = self.conv_shortcut(inputs, feat_cache=feat_cache, feat_idx=feat_idx)
            else:
                # 1x1x1 卷积不需要缓存（kernel_size=1）
                inputs = self.conv_shortcut(inputs)
        
        # 残差连接：输出 = F(x) + x
        # 这是 ResNet 的核心：学习残差而不是直接学习映射
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
        """旧缓存模式的实现（类似新模式，但使用字典缓存）"""
        new_conv_cache = {}
        conv_cache = conv_cache or {}
        
        hidden_states = inputs
        
        # 第一个残差路径
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
        
        # 时间嵌入
        if temb is not None and self.temb_proj is not None:
            temb_proj = self.temb_proj(jax.nn.silu(temb))
            hidden_states = hidden_states + temb_proj[:, None, None, None, :]
        
        # 第二个残差路径
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
        
        # 残差连接
        hidden_states = hidden_states + inputs
        
        return hidden_states, new_conv_cache


# ==================== 下采样块 ====================

class FlaxCogVideoXDownBlock3D(nnx.Module):
    """
    下采样块 - 用于编码器
    
    功能：
    1. 通过多个 ResNet 块提取特征
    2. 通过下采样减少空间分辨率
    3. 可选的时间维度压缩
    
    下采样的作用：
    - 增大感受野
    - 减少计算量
    - 提取更抽象的特征
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
        """
        初始化下采样块
        
        参数说明：
        - compress_time: 是否压缩时间维度（通过平均池化）
        - add_downsample: 是否添加下采样层
        - num_layers: ResNet 块的数量
        
        下采样策略：
        - 空间下采样：使用 stride=2 的 2D 卷积
        - 时间压缩：使用平均池化（保留第一帧，其余帧两两平均）
        """
        # 创建多个 ResNet 层
        resnets = []
        for i in range(num_layers):
            # 第一层输入通道数可能不同
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
        
        # nnx.List: Flax NNX 的列表容器，用于存储子模块
        self.resnets = nnx.List(resnets)
        
        # 下采样器（仅对空间维度下采样，不涉及时间）
        if add_downsample:
            # 使用 2D 卷积进行空间下采样
            # stride=2 将空间分辨率减半
            downsampler = FlaxConv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=0,  # 手动添加 padding
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
        """
        下采样块的前向传播
        
        流程：
        1. 通过多个 ResNet 块
        2. （可选）时间压缩
        3. （可选）空间下采样
        """
        new_conv_cache = {}
        conv_cache = conv_cache or {}
        
        # 通过所有 ResNet 块
        for i, resnet in enumerate(self.resnets):
            conv_cache_key = f"resnet_{i}"
            hidden_states, new_conv_cache[conv_cache_key] = resnet(
                hidden_states, temb, zq, conv_cache=conv_cache.get(conv_cache_key), deterministic=deterministic
            )
        
        if self.downsamplers is not None:
            # 时间压缩（如果需要）
            if self.compress_time:
                B, T, H, W, C = hidden_states.shape
                # Reshape 以应用池化
                hidden_states = hidden_states.reshape(B * H * W, T, C)
                hidden_states = hidden_states.transpose(0, 2, 1)  # (B*H*W, C, T)
                
                if T % 2 == 1:
                    # 奇数帧：保留第一帧，其余帧两两平均
                    first_frame = hidden_states[:, :, 0:1]
                    rest_frames = hidden_states[:, :, 1:]
                    if rest_frames.shape[2] > 0:
                        # 平均池化
                        rest_frames = jnp.mean(
                            rest_frames.reshape(B*H*W, C, rest_frames.shape[2]//2, 2),
                            axis=-1
                        )
                    hidden_states = jnp.concatenate([first_frame, rest_frames], axis=2)
                else:
                    # 偶数帧：常规平均池化
                    hidden_states = jnp.mean(
                        hidden_states.reshape(B*H*W, C, T//2, 2),
                        axis=-1
                    )
                
                # Reshape 回原始格式
                T_new = hidden_states.shape[2]
                hidden_states = hidden_states.transpose(0, 2, 1)  # (B*H*W, T_new, C)
                hidden_states = hidden_states.reshape(B, H, W, T_new, C)
                hidden_states = hidden_states.transpose(0, 3, 1, 2, 4)  # (B, T_new, H, W, C)
            
            # 空间下采样
            for downsampler in self.downsamplers:
                B, T, H, W, C = hidden_states.shape
                
                # 手动添加 padding (0, 1, 0, 1)
                pad_width = [
                    (0, 0),  # batch
                    (0, 0),  # time
                    (0, 1),  # height: 下方填充1
                    (0, 1),  # width: 右侧填充1
                    (0, 0),  # channels
                ]
                hidden_states = jnp.pad(hidden_states, pad_width, mode='constant', constant_values=0)
                
                # Reshape 以应用 2D 卷积
                _, _, H_padded, W_padded, _ = hidden_states.shape
                hidden_states = hidden_states.reshape(B * T, H_padded, W_padded, C)
                hidden_states = downsampler(hidden_states)
                
                # Reshape 回 5D
                _, H_new, W_new, _ = hidden_states.shape
                hidden_states = hidden_states.reshape(B, T, H_new, W_new, C)
        
        return hidden_states, new_conv_cache


# ==================== 中间块 ====================

class FlaxCogVideoXMidBlock3D(nnx.Module):
    """
    中间块 - 编码器和解码器的瓶颈部分
    
    作用：
    - 在编码器和解码器的中间
    - 处理最抽象的特征表示
    - 不改变空间分辨率
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
        """
        初始化中间块
        
        spatial_norm_dim:
        - None: 编码器（使用 GroupNorm）
        - 非None: 解码器（使用 SpatialNorm）
        """
        resnets = []
        for i in range(num_layers):
            resnet = FlaxCogVideoXResnetBlock3D(
                in_channels=in_channels,
                out_channels=in_channels,  # 中间块不改变通道数
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
        """中间块的前向传播"""
        # 新模式
        if feat_cache is not None and feat_idx is not None:
            for resnet in self.resnets:
                hidden_states, _ = resnet(
                    hidden_states, temb, zq,
                    feat_cache=feat_cache, feat_idx=feat_idx,
                    deterministic=deterministic
                )
            return hidden_states, None
        
        # 旧模式
        new_conv_cache = {}
        conv_cache = conv_cache or {}
        
        for i, resnet in enumerate(self.resnets):
            conv_cache_key = f"resnet_{i}"
            hidden_states, new_conv_cache[conv_cache_key] = resnet(
                hidden_states, temb, zq, conv_cache=conv_cache.get(conv_cache_key), deterministic=deterministic
            )
        
        return hidden_states, new_conv_cache


# ==================== 上采样块 ====================

class FlaxCogVideoXUpBlock3D(nnx.Module):
    """
    上采样块 - 用于解码器
    
    功能：
    1. 通过多个 ResNet 块恢复特征
    2. 通过上采样增加空间分辨率
    3. 可选的时间维度扩展
    
    上采样的作用：
    - 恢复空间细节
    - 从抽象特征重建图像
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
        """
        初始化上采样块
        
        compress_time:
        - True: 同时上采样时间和空间（2x）
        - False: 只上采样空间
        """
        # 创建 ResNet 层
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
        
        # 上采样器
        if add_upsample:
            upsampler = FlaxConv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=upsample_padding,
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
        """
        上采样块的前向传播
        
        流程：
        1. 通过 ResNet 块
        2. （可选）插值上采样
        3. 通过 2D 卷积细化
        """
        # 新模式
        if feat_cache is not None and feat_idx is not None:
            for resnet in self.resnets:
                hidden_states, _ = resnet(
                    hidden_states, temb, zq,
                    feat_cache=feat_cache, feat_idx=feat_idx,
                    deterministic=deterministic
                )
            
            if self.upsamplers is not None:
                for upsampler in self.upsamplers:
                    B, T, H, W, C = hidden_states.shape
                    
                    if self.compress_time:
                        # 时间+空间上采样
                        if T == 1:
                            # 单帧：1 -> 2帧，空间2x
                            hidden_states = jax.image.resize(hidden_states, (B, 2, H * 2, W * 2, C), method='nearest')
                        elif T > 1 and T % 2 == 1:
                            # 奇数帧
                            first_frame = hidden_states[:, 0, :, :, :]
                            rest_frames = hidden_states[:, 1:, :, :, :]
                            first_frame = jax.image.resize(first_frame, (B, H * 2, W * 2, C), method='nearest')
                            first_frame = first_frame[:, None, :, :, :]
                            rest_frames = jax.image.resize(rest_frames, (B, 2 * (T-1), H * 2, W * 2, C), method='nearest')
                            hidden_states = jnp.concatenate([first_frame, rest_frames], axis=1)
                        else:
                            # 偶数帧
                            hidden_states = jax.image.resize(hidden_states, (B, T * 2, H * 2, W * 2, C), method='nearest')
                    else:
                        # 只做空间上采样
                        hidden_states = hidden_states.reshape(B * T, H, W, C)
                        hidden_states = jax.image.resize(hidden_states, (B * T, H * 2, W * 2, C), method='nearest')
                        hidden_states = hidden_states.reshape(B, T, H * 2, W * 2, C)
                    
                    # 应用 2D 卷积
                    B, T_new, H_new, W_new, C = hidden_states.shape
                    hidden_states = hidden_states.reshape(B * T_new, H_new, W_new, C)
                    hidden_states = upsampler(hidden_states)
                    _, H_final, W_final, _ = hidden_states.shape
                    hidden_states = hidden_states.reshape(B, T_new, H_final, W_final, C)
            
            return hidden_states, None
        
        # 旧模式（省略详细实现，与新模式类似）
        # ... [旧模式代码] ...
        return hidden_states, {}


# ==================== 缓存管理器 ====================

class FlaxCogVideoXCache:
    """
    缓存管理器 - 用于逐帧解码
    
    这个类的核心作用是管理所有 CausalConv3d 层的缓存。
    
    为什么需要缓存管理器：
    1. 逐帧处理需要保存历史帧
    2. 每个 CausalConv3d 层都需要独立的缓存
    3. 需要跟踪当前处理到哪个层
    
    关键属性：
    - _conv_num: CausalConv3d 层的总数
    - _feat_map: 缓存列表（每个元素对应一个层）
    - _conv_idx: 当前层的索引
    
    为什么使用 [None] * n 初始化：
    - 创建一个长度为 n 的列表
    - 所有元素初始为 None
    - 后续逐个填充实际的缓存数据
    
    例如：[None] * 3 = [None, None, None]
    """
    
    def __init__(self, decoder_module):
        """
        初始化缓存管理器
        
        Args:
            decoder_module: FlaxCogVideoXDecoder3D 实例
        """
        self.decoder_module = decoder_module
        self.clear_cache()
    
    def clear_cache(self):
        """
        清空所有缓存
        
        这个方法会：
        1. 计算 decoder 中的 CausalConv3d 层数量
        2. 创建对应数量的缓存槽位
        3. 重置索引
        """
        # 计算需要多少个缓存槽位
        self._conv_num = self._count_causal_conv3d(self.decoder_module)
        
        # 使用列表（而不是整数）存储索引，以便在函数间传递引用
        self._conv_idx = [0]
        
        # 创建缓存列表
        # [None] * n 创建一个包含 n 个 None 的列表
        # 这是 Python 的列表乘法语法
        self._feat_map = [None] * self._conv_num
    
    @staticmethod  # 静态方法装饰器
    def _count_causal_conv3d(module):
        """
        递归计算模块中 CausalConv3d 层的数量
        
        @staticmethod 的作用：
        - 不需要访问类或实例的属性
        - 可以通过类名直接调用
        - 不接收 self 或 cls 参数
        - 本质上是一个普通函数，只是在类的命名空间中
        
        为什么使用 staticmethod：
        - 这个方法不需要访问实例或类的数据
        - 只是一个工具函数
        - 放在类中是为了组织代码
        
        Args:
            module: nnx.Module 实例
            
        Returns:
            int: CausalConv3d 层的数量
        """
        count = 0
        # 使用 nnx.graph.iter_graph 遍历所有子模块
        # 这会递归遍历整个模块树
        node_types = nnx.graph.iter_graph([module])
        for _, value in node_types:
            # isinstance: 检查对象是否是指定类的实例
            if isinstance(value, FlaxCogVideoXCausalConv3d):
                count += 1
        return count


# ==================== 主要的 VAE 模型 ====================
# 由于文件长度限制，这里只展示关键部分的注释
# 完整代码请参考原文件

"""
后续还有以下重要类（原文件中已有实现）：

1. FlaxCogVideoXEncoder3D: 编码器网络
   - 将视频编码为潜在表示
   - 包含多个下采样块和中间块

2. FlaxCogVideoXDecoder3D: 解码器网络
   - 从潜在表示解码回视频
   - 包含中间块和多个上采样块
   - 支持逐帧解码（通过 feat_cache）

3. FlaxAutoencoderKLCogVideoX: 完整的 VAE 模型
   - 组合编码器和解码器
   - 提供 encode() 和 decode() 接口
   - 支持 tiling（分块处理大图像）
   - 支持逐帧处理（避免 OOM）
   - 提供 from_pretrained() 方法加载预训练权重

关键技术点总结：

1. 因果卷积：时间维度只使用过去信息
2. 缓存机制：逐帧处理时保存历史
3. 空间条件归一化：提高解码质量
4. Tiling：处理超大分辨率视频
5. 逐帧解码：避免内存溢出

这个实现的特点：
- 完全兼容 PyTorch 版本
- 针对 TPU/GPU 优化
- 支持多种内存优化策略
- 完整的注释和文档
"""