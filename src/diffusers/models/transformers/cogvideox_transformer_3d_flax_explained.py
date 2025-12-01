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
======================================================================================
CogVideoX Transformer 3D 模型的 JAX/Flax 实现 - 详细注释版
======================================================================================

【文件概述】
这是 CogVideoX 视频生成模型的 Transformer 核心实现。CogVideoX 是一个基于 Diffusion 的
视频生成模型，使用 DiT（Diffusion Transformer）架构来处理视频数据。

【整体架构流程】

    输入视频latents (B, T, H, W, C)
            ↓
    ┌───────────────────────────────────────────────────────────┐
    │  FlaxCogVideoXPatchEmbed                                  │
    │  - 将视频切成patches，转换成token序列                       │
    │  - 投影文本嵌入到相同维度                                   │
    │  - 拼接文本tokens和视频tokens                              │
    └───────────────────────────────────────────────────────────┘
            ↓                           ↓
    拼接后的序列 (B, text_len+video_len, inner_dim)     时间步 t
            ↓                           ↓
            │                    ┌──────────────────────┐
            │                    │ FlaxTimesteps        │
            │                    │ 正弦编码: t → (B,inner_dim)│
            │                    └──────────────────────┘
            │                           ↓
            │                    ┌──────────────────────┐
            │                    │ FlaxTimestepEmbedding│
            │                    │ MLP: (B,inner_dim)→(B,512)│
            │                    └──────────────────────┘
            │                           ↓
            │                      时间嵌入 temb
            ↓                           ↓
    ┌───────────────────────────────────────────────────────────┐
    │  FlaxCogVideoXBlock × N (2B: 30层, 5B: 42层)              │
    │  ┌─────────────────────────────────────────────────────┐ │
    │  │ FlaxCogVideoXLayerNormZero                          │ │
    │  │ - LayerNorm + 时间步调制 (scale/shift/gate)          │ │
    │  └─────────────────────────────────────────────────────┘ │
    │                        ↓                                  │
    │  ┌─────────────────────────────────────────────────────┐ │
    │  │ FlaxAttention (联合注意力)                           │ │
    │  │ - 文本和视频tokens一起做自注意力                      │ │
    │  │ - 使用 Splash Attention (TPU优化)                    │ │
    │  └─────────────────────────────────────────────────────┘ │
    │                        ↓                                  │
    │  ┌─────────────────────────────────────────────────────┐ │
    │  │ FlaxCogVideoXLayerNormZero                          │ │
    │  └─────────────────────────────────────────────────────┘ │
    │                        ↓                                  │
    │  ┌─────────────────────────────────────────────────────┐ │
    │  │ FlaxFeedForward                                      │ │
    │  │ - Linear → GELU → Linear                             │ │
    │  └─────────────────────────────────────────────────────┘ │
    └───────────────────────────────────────────────────────────┘
            ↓
    ┌───────────────────────────────────────────────────────────┐
    │  Final LayerNorm + FlaxAdaLayerNorm                       │
    │  Linear投影回patch维度                                     │
    │  Unpatchify: tokens → (B, T, H, W, C)                     │
    └───────────────────────────────────────────────────────────┘
            ↓
    输出: 去噪后的视频latents (B, T, H, W, C)


【关键概念解释】

1. Timestep Embedding（时间步嵌入）- 为什么用正余弦编码 + MLP？
   
   【问题背景】
   Diffusion模型在每个去噪步骤都需要知道当前是第几步 (t=999, 998, ..., 0)。
   直接把数字t输入模型有两个问题：
   - 数值范围问题：t=0 和 t=999 差异太大，难以学习
   - 表达能力问题：单个数字包含的信息太少
   
   【解决方案：两步走】
   
   Step 1: FlaxTimesteps (正弦/余弦编码)
   ────────────────────────────────────
   把标量 t 转换成高维向量，类似 Transformer 的位置编码。
   
   为什么用正弦编码而不是直接 one-hot 或线性映射？
   
   ┌─────────────────────────────────────────────────────────────────────────┐
   │ 方案对比                                                                │
   ├─────────────────────────────────────────────────────────────────────────┤
   │ One-hot: 需要 1000 维向量，且相邻时间步完全正交，失去"接近性"信息        │
   │ 线性映射: t/1000 → 单个数字，信息量不足                                  │
   │ 正弦编码: ✓ 低维高效 ✓ 保持相对距离 ✓ 可泛化到未见过的时间步             │
   └─────────────────────────────────────────────────────────────────────────┘
   
   正弦编码的数学性质：
   - 不同频率的sin/cos组合 → 类似"傅里叶分解"
   - t=500 和 t=501 的编码向量很接近（高频成分略有变化）
   - t=0 和 t=999 的编码向量差异很大（所有频率成分都不同）
   - 可以外推到训练时没见过的时间步（比如 t=1500）
   
   Step 2: FlaxTimestepEmbedding (可学习的 MLP)
   ────────────────────────────────────────────
   为什么还需要 MLP？正弦编码不够吗？
   
   ┌─────────────────────────────────────────────────────────────────────────┐
   │ 正弦编码的局限性：                                                       │
   │ - 固定的数学变换，无法适应特定任务                                        │
   │ - 不同模型可能需要不同的时间步"语义"                                      │
   │ - 去噪过程不是线性的，早期/后期的行为差异很大                             │
   │                                                                         │
   │ MLP 的作用：                                                             │
   │ - 让模型自己学习"什么样的时间表示最有用"                                 │
   │ - 可以学到非线性的时间依赖关系                                            │
   │ - 不同的模型层可能需要不同的时间信息，MLP 可以提供更灵活的映射            │
   └─────────────────────────────────────────────────────────────────────────┘
   
   【类比理解】
   
   想象你在教一个人画画，告诉他"你在第500步"：
   - 正弦编码：把"500"这个数字翻译成一种"坐标系"，让AI容易理解
   - MLP：让AI自己学习"第500步意味着什么"——是应该画大轮廓还是细节？
   
   这就像：
   - 正弦编码 = 给你一张地图坐标 (经纬度)
   - MLP = 你自己学习这个坐标代表什么地方 (沙漠？城市？)

2. Patch Embedding（补丁嵌入）:
   - 将视频帧分割成小块（patches），类似于 ViT
   - CogVideoX 1.0: 只在空间维度分patch (2D Conv, patch_size=2)
   - CogVideoX 1.5: 在时间+空间维度分patch (3D, 用Linear)

3. 联合注意力（Joint Attention）:
   - 文本tokens和视频tokens拼接后一起做注意力
   - 让文本和视频特征相互交互，实现文本引导

4. 条件调制（Conditional Modulation）:
   - 使用时间步嵌入来调制LayerNorm的scale和shift
   - 公式: output = LayerNorm(x) * (1 + scale) + shift
   - 让模型在不同去噪阶段有不同的行为

5. Splash Attention:
   - Google为TPU优化的注意力实现
   - 使用分块计算减少内存占用
   - 支持多设备分片并行
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from flax import nnx
import math
import functools
from jax.experimental.pallas.ops.tpu import splash_attention
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P, Mesh


# ======================================================================================
# 配置类
# ======================================================================================

@dataclass
class FlaxCogVideoXTransformer3DConfig:
    """
    CogVideoX Transformer 3D 模型的配置类。
    
    【重要参数说明】
    
    ═══════════════════════════════════════════════════════════════════════════
    【inner_dim - Transformer的隐藏维度，极其重要！】
    ═══════════════════════════════════════════════════════════════════════════
    
    inner_dim = num_attention_heads × attention_head_dim
    
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ 模型版本      │ 注意力头数 │ 每头维度 │ inner_dim │ Transformer层数    │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ CogVideoX-2B │    30     │   64    │   1920    │       30层         │
    │ CogVideoX-5B │    48     │   64    │   3072    │       42层         │ ← 本文件主要示例
    └─────────────────────────────────────────────────────────────────────────┘
    
    这个 inner_dim 就是Transformer的隐藏维度，它贯穿整个模型的所有层：
    - Patch嵌入后的token维度 = inner_dim
    - 每个Transformer块的输入/输出维度 = inner_dim
    - 时间步正弦编码的维度 = inner_dim（注意不是time_embed_dim=512！）
    
    视频latent参数:
        - in_channels (16): VAE编码后的视频latent通道数
        - sample_frames (49): latent空间的帧数
        - sample_height (60): latent空间的高度
        - sample_width (90): latent空间的宽度
        
        实际视频尺寸 = latent尺寸 × VAE缩放因子
        例如: 60 × 8 = 480 像素高度
    
    Patch相关:
        - patch_size (2): 空间patch大小，每2×2 latent像素合成1个token
        - patch_size_t: 时间patch大小
          - None: CogVideoX 1.0，只做空间patch
          - 某整数: CogVideoX 1.5，时空联合patch
        
        视频tokens数量 = T × (H/patch_size) × (W/patch_size)
                       = 49 × 30 × 45 = 66,150 tokens (1.0模式)
    """
    
    config_name: str = "config.json"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Transformer架构参数
    # ─────────────────────────────────────────────────────────────────────────
    num_attention_heads: int = 30      # 多头注意力的头数 (2B=30, 5B=48)
    attention_head_dim: int = 64       # 每个头的维度 (inner_dim = heads × 64)
    in_channels: int = 16              # 输入视频latent的通道数 (VAE输出)
    out_channels: int = 16             # 输出通道数，与in_channels相同
    
    # ─────────────────────────────────────────────────────────────────────────
    # 时间步嵌入参数
    # ─────────────────────────────────────────────────────────────────────────
    flip_sin_to_cos: bool = True       # 正弦编码顺序: True=[cos,sin], False=[sin,cos]
    freq_shift: int = 0                # 频率偏移（通常为0）
    time_embed_dim: int = 512          # 时间嵌入最终维度，用于调制各层
    ofs_embed_dim: Optional[int] = None  # 光流尺度嵌入维度（I2V模型使用）
    
    # ─────────────────────────────────────────────────────────────────────────
    # 文本编码参数
    # ─────────────────────────────────────────────────────────────────────────
    text_embed_dim: int = 4096         # T5-XXL文本编码器的输出维度
    
    # ─────────────────────────────────────────────────────────────────────────
    # 模型深度和正则化
    # ─────────────────────────────────────────────────────────────────────────
    num_layers: int = 30               # Transformer块数量
    dropout: float = 0.0               # Dropout率
    attention_bias: bool = True        # 注意力层是否使用bias
    
    # ─────────────────────────────────────────────────────────────────────────
    # 视频尺寸（latent空间，非像素空间）
    # ─────────────────────────────────────────────────────────────────────────
    sample_width: int = 90             # latent宽度
    sample_height: int = 60            # latent高度
    sample_frames: int = 49            # latent帧数
    
    # ─────────────────────────────────────────────────────────────────────────
    # Patch相关参数
    # ─────────────────────────────────────────────────────────────────────────
    patch_size: int = 2                # 空间patch大小 (2×2)
    patch_size_t: Optional[int] = None # 时间patch大小（None=不做时间patch，即1.0模式）
    temporal_compression_ratio: int = 4 # VAE时间压缩比
    max_text_seq_length: int = 226     # 最大文本token数
    
    # ─────────────────────────────────────────────────────────────────────────
    # 激活函数和归一化
    # ─────────────────────────────────────────────────────────────────────────
    activation_fn: str = "gelu-approximate"      # FFN激活函数
    timestep_activation_fn: str = "silu"         # 时间嵌入MLP激活函数
    norm_elementwise_affine: bool = True         # LayerNorm是否有可学习参数
    norm_eps: float = 1e-5                       # LayerNorm的epsilon
    
    # ─────────────────────────────────────────────────────────────────────────
    # 位置编码
    # ─────────────────────────────────────────────────────────────────────────
    spatial_interpolation_scale: float = 1.875   # 空间位置编码插值尺度
    temporal_interpolation_scale: float = 1.0    # 时间位置编码插值尺度
    use_rotary_positional_embeddings: bool = False  # 是否使用RoPE旋转位置编码
    use_learned_positional_embeddings: bool = False # 是否使用可学习位置编码
    patch_bias: bool = True                      # patch投影层是否使用bias
    
    @classmethod
    def from_dict(cls, config_dict: Dict):
        """从字典创建配置，自动过滤不存在的字段"""
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(cls)}
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}
        return cls(**filtered_dict)


# ======================================================================================
# 时间步嵌入模块
# ======================================================================================

class FlaxTimesteps(nnx.Module):
    """
    时间步正弦嵌入模块 - 将标量时间步转换为向量表示。
    
    【这个类的作用】
    在 Diffusion 模型中，去噪过程有多个时间步（如 t=999, 998, ..., 0）。
    模型需要知道当前是第几步，才能正确地预测噪声。
    
    直接把数字 t=500 输入模型效果不好，需要转换成向量：
    1. FlaxTimesteps: 把标量 t=500 → 向量 [cos(...), sin(...)]
    2. FlaxTimestepEmbedding: 再通过MLP投影到更高维度
    
    ═══════════════════════════════════════════════════════════════════════════
    【维度变换】
    ═══════════════════════════════════════════════════════════════════════════
    
    输入: timestep，形状 (B,) 或标量
          例如: t = 500 (表示当前在第500个去噪步)
    
    输出: 正弦嵌入向量，形状 (B, num_channels)
          - CogVideoX-2B: (B, 1920)
          - CogVideoX-5B: (B, 3072)
    
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ 输入                      输出                                          │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ t = 500 (标量)     →     (1, 3072) 向量 [5B模型]                        │
    │ t = [500, 300]     →     (2, 3072) 矩阵 [batch=2, 5B模型]               │
    │     ↑ (B=2,)             ↑ 每行是一个时间步的编码                        │
    └─────────────────────────────────────────────────────────────────────────┘
    
    ═══════════════════════════════════════════════════════════════════════════
    【正弦编码的数学公式】
    ═══════════════════════════════════════════════════════════════════════════
    
    设 d = num_channels (如 3072), half_dim = d/2 (如 1536)
    
    对于第 i 个维度 (i = 0, 1, 2, ..., half_dim-1):
    
        频率因子:  ω_i = exp(-i × ln(10000) / half_dim)
                      = 10000^(-i / half_dim)
        
        编码公式:  emb[i]            = cos(t × ω_i)    ← 前半部分是 cos
                   emb[i + half_dim] = sin(t × ω_i)    ← 后半部分是 sin
    
    ═══════════════════════════════════════════════════════════════════════════
    【为什么是 10000？这个数字怎么来的？】
    ═══════════════════════════════════════════════════════════════════════════
    
    10000 来自 2017 年 Google 的 Transformer 原论文 "Attention Is All You Need"。
    这是一个**经验性选择**（empirical choice），而非严格的理论推导。
    
    【设计思路】
    
    作者希望频率因子 ω 形成一个几何级数，从 1 递减到一个很小的值：
    
        ω_0 = 1                    (最高频，周期 = 2π ≈ 6.28)
        ω_{d/2-1} = 1/10000       (最低频，周期 = 2π × 10000 ≈ 62832)
    
    这样设计的目的是：
    
    1. 【覆盖多个尺度】
       - 最高频 (ω=1): 周期约6，适合区分相邻位置 (如 t=500 vs t=501)
       - 最低频 (ω=1/10000): 周期约62832，适合区分整体范围
       
    2. 【匹配预期序列长度】
       - 原论文处理的序列长度约 512-1000
       - 选择 10000 确保最低频的周期远大于序列长度
       - 这样即使是最长序列，低频成分也不会"绕回"造成混淆
    
    【为什么不用其他数字？】
    
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ 数值选择     │ 效果                                                     │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ 太小 (如100) │ 最低频周期太短，长序列会出现周期性混淆                    │
    │ 太大 (如10^6)│ 低频成分变化太慢，浪费维度，表达能力下降                   │
    │ 10000       │ 平衡点：周期范围 [6, 62832]，覆盖大多数应用场景            │
    └─────────────────────────────────────────────────────────────────────────┘
    
    【在 Diffusion 模型中的适用性】
    
    Diffusion 模型的时间步通常是 0-1000，恰好在 10000 的覆盖范围内：
    - t=1000 时，最高频成分 cos(1000) 已经转了约 159 圈
    - t=1000 时，最低频成分 cos(0.1) 只转了约 0.016 圈
    
    这确保了：
    - 相邻时间步有可区分的高频差异
    - 整体时间范围有平滑的低频变化
    
    【其他模型的选择】
    
    有些模型使用不同的值：
    - BERT: 10000
    - GPT-2: 10000
    - 某些 Diffusion 模型: 10000 或可配置
    
    本质上，只要这个数字足够大以覆盖序列长度，具体值影响不大。
    10000 是一个被广泛验证有效的经验值。
    
    【频率因子的几何级数】
    
        i=0:    ω_0 = 10000^0 = 1           → 最高频 (t=500 → 500弧度)
        i=1:    ω_1 = 10000^(-1/1536) ≈ 0.994
        i=2:    ω_2 = 10000^(-2/1536) ≈ 0.988
        ...
        i=1535: ω_1535 = 10000^(-1535/1536) ≈ 0.0001  → 最低频 (t=500 → 0.05弧度)
    
    【具体计算示例】(以 d=8, t=100 为例，简化演示)
    
        half_dim = 4
        
        Step 1: 计算频率因子
        ────────────────────────────────────────────────────────────────
        i=0: ω_0 = 10000^(-0/4) = 10000^0 = 1
        i=1: ω_1 = 10000^(-1/4) = 10000^-0.25 ≈ 0.1
        i=2: ω_2 = 10000^(-2/4) = 10000^-0.5 = 0.01
        i=3: ω_3 = 10000^(-3/4) = 10000^-0.75 ≈ 0.001
        
        频率从1递减到0.001，是一个几何级数
        
        Step 2: 计算角度 (t × ω)
        ────────────────────────────────────────────────────────────────
        angle_0 = 100 × 1     = 100     弧度 (高频，变化剧烈)
        angle_1 = 100 × 0.1   = 10      弧度
        angle_2 = 100 × 0.01  = 1       弧度
        angle_3 = 100 × 0.001 = 0.1     弧度 (低频，变化平缓)
        
        Step 3: 应用三角函数
        ────────────────────────────────────────────────────────────────
        emb = [cos(100), cos(10), cos(1), cos(0.1),   ← 前半: cos
               sin(100), sin(10), sin(1), sin(0.1)]   ← 后半: sin
            ≈ [0.86,    -0.84,   0.54,   0.995,
               -0.51,    -0.54,   0.84,   0.0998]
        
        最终输出: 一个8维向量 (如果d=3072，则是3072维向量)
    
    【为什么这样设计？】
    
    1. 多尺度表示:
       - 高频成分 (ω大): 对相邻时间步敏感 (t=500 vs t=501 差异大)
       - 低频成分 (ω小): 对时间范围敏感 (t=0 vs t=500 差异大)
       
    2. 平滑过渡:
       - 相邻时间步 (如 t=500, t=501) 的编码向量接近
       - 远离时间步 (如 t=0, t=999) 的编码向量差异大
       
    3. 唯一性:
       - 不同时间步的编码是唯一的
       - 类似"指纹"，每个时间步有独特的模式
    
    ═══════════════════════════════════════════════════════════════════════════
    【与 FlaxTimestepEmbedding 的关系】
    ═══════════════════════════════════════════════════════════════════════════
    
        timestep (标量, 如 t=500)
              ↓
        FlaxTimesteps ←── 你在这里
        正弦编码，固定数学变换（无可学习参数）
        t=500 → (B, 3072) 向量 [5B模型]
              ↓
        FlaxTimestepEmbedding
        MLP投影，有可学习参数
        (B, 3072) → Linear → (B, 512) → SiLU → Linear → (B, 512)
              ↓
        (B, 512) 时间嵌入向量
              ↓
        用于调制Transformer各层的 LayerNorm
    
    【为什么分两步？】
    - FlaxTimesteps: 固定的数学变换（无参数），把标量变成"有意义"的向量
    - FlaxTimestepEmbedding: 可学习的MLP，让模型自己学习如何使用这个向量
    """
    
    def __init__(
        self,
        num_channels: int,           # 输出维度，等于 inner_dim (2B:1920, 5B:3072)
        flip_sin_to_cos: bool = True,
        freq_shift: float = 0,
    ):
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.freq_shift = freq_shift
    
    def __call__(self, timesteps: jnp.ndarray) -> jnp.ndarray:
        """
        生成时间步的正弦嵌入。
        
        Args:
            timesteps: 时间步值，形状 (B,) 或标量
                      例如: [500, 500] 表示batch中两个样本都在第500步
            
        Returns:
            正弦嵌入向量，形状 (B, num_channels)
            例如: (2, 1920)
        
        【计算过程示例】
        假设 num_channels=8, timestep=100:
        
        1. half_dim = 4
        2. 计算频率因子:
           exponent = [0, -2.3, -4.6, -6.9] (log(10000)的等差数列)
           freq = exp(exponent) = [1, 0.1, 0.01, 0.001] (几何级数下降)
        3. 乘以时间步:
           angles = 100 × [1, 0.1, 0.01, 0.001] = [100, 10, 1, 0.1]
        4. 应用三角函数:
           output = [cos(100), cos(10), cos(1), cos(0.1),
                     sin(100), sin(10), sin(1), sin(0.1)]
        
        注意: 高频成分（如cos(100)）变化很快，低频成分（如cos(0.1)）变化很慢
        """
        # 确保 timesteps 是1D数组
        if timesteps.ndim == 0:
            timesteps = timesteps[None]
        
        # 计算一半的维度（因为sin和cos各占一半）
        half_dim = self.num_channels // 2
        
        # 计算频率因子的指数
        # 频率从高到低：10000^0, 10000^(-1/half_dim), 10000^(-2/half_dim), ...
        exponent = -math.log(10000) * jnp.arange(0, half_dim, dtype=jnp.float32)
        exponent = exponent / (half_dim - self.freq_shift)
        
        # 计算频率因子 [1, 0.x, 0.0x, 0.00x, ...]
        emb = jnp.exp(exponent)  # (half_dim,)
        
        # 时间步 × 频率因子 = 三角函数的角度
        # timesteps[:, None]: (B, 1) × emb[None, :]: (1, half_dim) → (B, half_dim)
        emb = timesteps[:, None] * emb[None, :]
        
        # 应用 sin 和 cos，拼接成完整的嵌入
        if self.flip_sin_to_cos:
            emb = jnp.concatenate([jnp.cos(emb), jnp.sin(emb)], axis=-1)
        else:
            emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        
        return emb  # (B, num_channels)


class FlaxTimestepEmbedding(nnx.Module):
    """
    时间步嵌入的 MLP 投影模块。
    
    【这个类的作用】
    接收 FlaxTimesteps 生成的正弦嵌入，通过两层 MLP 投影到目标维度。
    这是一个**可学习**的映射，让模型自己学习如何使用时间信息。
    
    【网络结构】(以5B模型为例)
        input (B, 3072)
           ↓
        Linear(3072 → 512) ← linear_1  ← 降维!
           ↓
        SiLU激活函数
           ↓
        Linear(512 → 512) ← linear_2
           ↓
        output (B, 512)
    
    ═══════════════════════════════════════════════════════════════════════════
    【为什么从 3072 降维到 512？为什么不保持 3072？】
    ═══════════════════════════════════════════════════════════════════════════
    
    这是一个很好的问题！降维的原因有以下几点：
    
    1. 【信息瓶颈 (Information Bottleneck)】
       ─────────────────────────────────────────────────────────────────────
       时间步是一个简单的标量 (如 t=500)，本质上只包含很少的信息。
       3072维的正弦编码是冗余的——目的是让不同时间步有不同的"指纹"。
       
       真正需要传递给模型的"时间语义"远不需要3072维。
       512维足以编码：
       - "当前是早期/中期/晚期哪个阶段"
       - "应该保守还是激进地去噪"
       - 其他与时间相关的控制信号
       
       强制压缩到512维 → 模型必须学习提取最本质的时间信息
    
    2. 【参数效率】
       ─────────────────────────────────────────────────────────────────────
       时间嵌入要被用于调制每一层的LayerNorm。看看参数量对比：
       
       ┌─────────────────────────────────────────────────────────────────────┐
       │ 方案                                 │ 每层调制所需参数             │
       ├─────────────────────────────────────────────────────────────────────┤
       │ 保持 3072 维: Linear(3072 → 18432)  │ 3072 × 18432 ≈ 56.6M        │
       │ 降到 512 维:  Linear(512 → 18432)   │ 512 × 18432 ≈ 9.4M          │
       │                                      │ 节省 83% 参数!              │
       └─────────────────────────────────────────────────────────────────────┘
       
       (18432 = 6 × inner_dim = 6 × 3072，是6组调制参数)
       
       如果保持3072维，每层多用 47M 参数，42层 = 多用约 2B 参数！
       这会让5B模型变成7B模型，训练和推理成本大增。
    
    3. 【计算效率】
       ─────────────────────────────────────────────────────────────────────
       每次前向传播都要计算 temb → 调制参数：
       - 512维: 512 × 18432 = 9.4M 次乘加
       - 3072维: 3072 × 18432 = 56.6M 次乘加
       
       降维让每一步推理都更快。
    
    4. 【正则化效果】
       ─────────────────────────────────────────────────────────────────────
       信息瓶颈有正则化作用，防止模型过度依赖时间信息。
       如果时间嵌入太高维，模型可能会记住训练集的时间特定模式。
       
    5. 【经验验证】
       ─────────────────────────────────────────────────────────────────────
       CogVideoX 团队（以及之前的 DiT、SDXL 等）通过实验验证：
       512维足以编码时间信息，增加维度收益递减。
       
       不同模型的 time_embed_dim 选择：
       - Stable Diffusion: 1280 (因为 inner_dim=1280)
       - SDXL: 1280
       - DiT: 根据模型大小调整
       - CogVideoX: 固定512
    
    【总结】
    
    3072维正弦编码是"展开"时间步，让每个时间步有独特模式。
    512维MLP输出是"压缩"，只保留对去噪有用的信息。
    
    类比：
    - 正弦编码3072维 = 把一个人的身份证号展开成二维码
    - MLP降到512维 = 从二维码提取出"姓名、年龄、地址"等有用字段
    
    ═══════════════════════════════════════════════════════════════════════════
    【为什么 in_channels 是 3072 (或1920) 而不是配置里的 in_channels (16)？】
    ═══════════════════════════════════════════════════════════════════════════
    
    这是容易混淆的地方！
    
    - 配置里的 in_channels = 16：是视频VAE latent的通道数
    - 这里的 in_channels 参数：是 FlaxTimesteps 的输出维度 = inner_dim
      - CogVideoX-2B: inner_dim = 30 × 64 = 1920
      - CogVideoX-5B: inner_dim = 48 × 64 = 3072
    
    看主模型的初始化代码就明白了：
    
        inner_dim = config.num_attention_heads * config.attention_head_dim  # 5B: 48×64=3072
        self.time_proj = FlaxTimesteps(inner_dim, ...)  # 输出3072维
        self.time_embedding = FlaxTimestepEmbedding(
            inner_dim,              # ← 这里传入3072，不是16！
            config.time_embed_dim,  # ← 512
            ...
        )
    
    【完整的时间步处理流程】(以5B模型为例)
    
        timestep = 500 (标量)
              ↓
        FlaxTimesteps(num_channels=3072)
        正弦编码，把标量"展开"成高维向量
              ↓
        (B, 3072) 正弦编码向量
              ↓
        FlaxTimestepEmbedding.linear_1(3072 → 512)
        降维，提取本质时间信息
              ↓
        SiLU激活
              ↓
        FlaxTimestepEmbedding.linear_2(512 → 512)
        非线性变换，学习时间语义
              ↓
        (B, 512) 时间嵌入 temb
              ↓
        传入每个 FlaxCogVideoXBlock，用于调制LayerNorm
    """
    
    def __init__(
        self,
        in_channels: int,         # 输入维度 = inner_dim (2B:1920, 5B:3072)
        time_embed_dim: int,      # 输出维度 = 512（用于调制）
        act_fn: str = "silu",     # 激活函数
        rngs: nnx.Rngs = None,
    ):
        # 第一层: inner_dim → 512（降维）
        self.linear_1 = nnx.Linear(in_channels, time_embed_dim, rngs=rngs)
        # 第二层: 512 → 512（保持维度，增加非线性表达能力）
        self.linear_2 = nnx.Linear(time_embed_dim, time_embed_dim, rngs=rngs)
        
        # 选择激活函数
        # SiLU(x) = x × sigmoid(x)，比ReLU更平滑，在Diffusion模型中常用
        if act_fn == "silu":
            self.act = jax.nn.silu
        elif act_fn == "gelu":
            self.act = jax.nn.gelu
        else:
            raise ValueError(f"Unsupported activation: {act_fn}")
    
    def __call__(
        self,
        sample: jnp.ndarray,
        condition: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Args:
            sample: 来自FlaxTimesteps的正弦嵌入，形状 (B, in_channels)
                   例如 5B模型: (2, 3072), 2B模型: (2, 1920)
            condition: 可选的额外条件（CogVideoX中未使用）
            
        Returns:
            投影后的时间嵌入，形状 (B, time_embed_dim)
            例如 (2, 512)
        """
        sample = self.linear_1(sample)  # (B, inner_dim) → (B, 512)
        sample = self.act(sample)        # SiLU非线性激活
        sample = self.linear_2(sample)  # (B, 512) → (B, 512)
        return sample


# ======================================================================================
# Patch 嵌入模块
# ======================================================================================

class FlaxCogVideoXPatchEmbed(nnx.Module):
    """
    CogVideoX 的 Patch 嵌入模块 - 将视频转换为token序列。
    
    【这个类的作用 - 为什么需要它？】
    
    Transformer 处理的是**序列**（一串token），但视频是**5D张量** (B, T, H, W, C)。
    我们需要把视频转换成序列形式，就像 ViT 处理图像一样。
    
    这个模块的工作：
    1. 把视频切成小块（patches），每个patch变成一个token
    2. 投影文本嵌入到相同维度
    3. 拼接文本和视频tokens
    4. 可选地添加位置编码
    
    【输入输出示例】
    
    输入:
        text_embeds: (B, 226, 4096) ← T5文本编码器输出
        image_embeds: (B, 49, 60, 90, 16) ← VAE编码的视频latents
    
    输出: (以5B模型为例)
        (B, 226 + 66150, 3072) = (B, 66376, 3072) ← 拼接后的token序列
    
    ═══════════════════════════════════════════════════════════════════════════
    【Patch操作详解 - 为什么这里的代码这么复杂？】
    ═══════════════════════════════════════════════════════════════════════════
    
    CogVideoX有两种patch模式：
    
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ 模式1: CogVideoX 1.0 (patch_size_t = None)                              │
    │ 只在空间维度做patch，每帧独立处理                                        │
    └─────────────────────────────────────────────────────────────────────────┘
    
    输入: (B, T, H, W, C) = (1, 49, 60, 90, 16)
    
    步骤:
    1. 重塑为 (B×T, H, W, C) = (49, 60, 90, 16)  ← 把帧堆叠成batch
    2. 2D卷积 (kernel=2×2, stride=2)
       → (49, 30, 45, inner_dim)  ← 空间尺寸减半，通道变成inner_dim
    3. 重塑回 (B, T×H'×W', embed_dim) = (1, 49×30×45, inner_dim) = (1, 66150, 3072) [5B]
    
    Token数量 = T × (H/2) × (W/2) = 49 × 30 × 45 = 66,150
    
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ 模式2: CogVideoX 1.5 (patch_size_t = 某整数，如2)                        │
    │ 在时间+空间维度同时做patch                                               │
    └─────────────────────────────────────────────────────────────────────────┘
    
    输入: (B, T, H, W, C) = (1, 48, 60, 90, 16)
    假设 patch_size=2, patch_size_t=2
    
    这个更复杂，用reshape+transpose手动提取3D patches：
    
    【复杂reshape操作的详细解释】
    
    Step 1: 分离patch边界
        (B, T, H, W, C)
        ↓ reshape
        (B, T/p_t, p_t, H/p, p, W/p, p, C)
        (1, 24,    2,   30,  2, 45,  2, 16)
        
        解读：
        - T方向有24个patch，每个包含2帧
        - H方向有30个patch，每个包含2像素
        - W方向有45个patch，每个包含2像素

    Step 2: 把patch内部的维度放一起
        (B, T/p_t, p_t, H/p, p, W/p, p, C)
        ↓ transpose(0,1,3,5,2,4,6,7)
        (B, T/p_t, H/p, W/p, p_t, p, p, C)
        (1, 24,    30,  45,  2,   2, 2, 16)
        
        前面是patch的位置，后面是patch内部的内容

    Step 3: 展平每个patch
        → (B, T', H', W', p_t×p×p×C)
        → (1, 24, 30, 45, 128)  # 每个patch变成128维向量

    Step 4: 展平所有patches成序列
        → (B, T'×H'×W', patch_dim) = (1, 32400, 128)

    Step 5: Linear投影
        → (B, 32400, inner_dim)  # 5B: 3072, 2B: 1920
    
    Token数量 = (T/2) × (H/2) × (W/2) = 24 × 30 × 45 = 32,400
    比1.0模式少很多！（66150 vs 32400）→ 计算更快
    
    【为什么用Conv vs Linear？】
    - CogVideoX 1.0用Conv2D：天然适合提取2D patches
    - CogVideoX 1.5用Linear：需要3D patch，手动reshape更灵活
    """
    
    def __init__(
        self,
        patch_size: int = 2,
        patch_size_t: Optional[int] = None,
        in_channels: int = 16,
        embed_dim: int = 1920,
        text_embed_dim: int = 4096,
        bias: bool = True,
        sample_width: int = 90,
        sample_height: int = 60,
        sample_frames: int = 49,
        temporal_compression_ratio: int = 4,
        max_text_seq_length: int = 226,
        spatial_interpolation_scale: float = 1.875,
        temporal_interpolation_scale: float = 1.0,
        use_positional_embeddings: bool = True,
        use_learned_positional_embeddings: bool = False,
        rngs: nnx.Rngs = None,
    ):
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.embed_dim = embed_dim
        self.sample_height = sample_height
        self.sample_width = sample_width
        self.sample_frames = sample_frames
        self.temporal_compression_ratio = temporal_compression_ratio
        self.max_text_seq_length = max_text_seq_length
        self.spatial_interpolation_scale = spatial_interpolation_scale
        self.temporal_interpolation_scale = temporal_interpolation_scale
        self.use_positional_embeddings = use_positional_embeddings
        self.use_learned_positional_embeddings = use_learned_positional_embeddings
        
        # Text projection
        self.text_proj = nnx.Linear(text_embed_dim, embed_dim, rngs=rngs)
        
        # Video patch projection
        if patch_size_t is None:
            # CogVideoX 1.0: 2D spatial patching with Conv2d
            kernel_size = (patch_size, patch_size)
            stride = (patch_size, patch_size)
            self.proj = nnx.Conv(
                in_channels,
                embed_dim,
                kernel_size=kernel_size,
                strides=stride,
                padding="VALID",
                use_bias=bias,
                rngs=rngs,
            )
            self.use_linear_proj = False
        else:
            # CogVideoX 1.5: 3D spatio-temporal patching with Linear projection
            # The input patches are flattened and projected through a Linear layer
            # patch_dim = in_channels * patch_size_t * patch_size * patch_size
            patch_dim = in_channels * patch_size_t * patch_size * patch_size
            self.proj = nnx.Linear(patch_dim, embed_dim, use_bias=bias, rngs=rngs)
            self.patch_size_t = patch_size_t
            self.use_linear_proj = True
        
        # ══════════════════════════════════════════════════════════════════════
        # 位置编码 (Positional Embeddings) - 详细解释
        # ══════════════════════════════════════════════════════════════════════
        #
        # 【为什么需要位置编码？】
        #
        # 视频被切成 patches 后变成一个 token 序列，但 Transformer 的自注意力
        # 本身是"位置无关"的——它不知道 token 的顺序。位置编码告诉模型：
        # "这是第1个 token，那是第66150个 token"
        #
        # 【三种模式】
        #
        # ┌─────────────────────────────────────────────────────────────────────┐
        # │ 模式                          │ 说明                                │
        # ├─────────────────────────────────────────────────────────────────────┤
        # │ use_positional_embeddings=True │                                    │
        # │ use_learned=True               │ 创建可学习的位置参数，训练时更新    │
        # ├─────────────────────────────────────────────────────────────────────┤
        # │ use_positional_embeddings=True │                                    │
        # │ use_learned=False              │ 用正弦余弦公式计算，不需要训练      │
        # ├─────────────────────────────────────────────────────────────────────┤
        # │ use_positional_embeddings=False│ 不在这里加位置编码                  │
        # │                                │ （可能用 RoPE 在 attention 里加）   │
        # └─────────────────────────────────────────────────────────────────────┘
        #
        # 【可学习位置编码的初始化】
        #
        # max_num_patches = 帧数 × (H/patch_size) × (W/patch_size)
        #                 = 49 × 30 × 45 = 66150  (以5B模型默认参数为例)
        #
        # pos_embedding: (1, 66150, 3072)
        #                ↑  ↑       ↑
        #                │  │       └── embed_dim，每个位置一个向量
        #                │  └────────── 最大 token 数量
        #                └───────────── batch=1，会广播到实际 batch_size
        #
        # * 0.02：初始化时用很小的值，避免干扰 token 嵌入（见前面的解释）
        #
        if use_positional_embeddings:
            if use_learned_positional_embeddings:
                # 可学习位置编码：创建一个可训练的参数矩阵
                max_num_patches = (
                    (sample_frames // temporal_compression_ratio) *
                    (sample_height // patch_size) *
                    (sample_width // patch_size)
                )
                self.pos_embedding = nnx.Param(
                    jax.random.normal(rngs(), (1, max_num_patches, embed_dim)) * 0.02
                )
            else:
                # 正弦位置编码：运行时用公式计算，不需要存储参数
                # 见下方 _get_sinusoidal_pos_embed() 的详细解释
                self.pos_embedding = None
        else:
            # 不使用位置编码（当使用 RoPE 旋转位置编码时）
            # RoPE 在 FlaxAttention 的 _apply_rotary_emb 中注入位置信息
            self.pos_embedding = None
    
    def __call__(
        self,
        text_embeds: jnp.ndarray,
        image_embeds: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Args:
            text_embeds: Text embeddings (B, max_text_seq_length, text_embed_dim)
            image_embeds: Video latents (B, T, H, W, C) in channel-last format
            
        Returns:
            Combined embeddings (B, text_seq_len + num_patches, embed_dim)
        """
        batch_size = image_embeds.shape[0]
        text_embeds = self.text_proj(text_embeds)
        
        # Patchify video latents
        if not self.use_linear_proj:
            # CogVideoX 1.0: 2D patching with Conv2d
            b, t, h, w, c = image_embeds.shape
            # Reshape to (B*T, H, W, C) for 2D conv
            image_embeds = image_embeds.reshape(b * t, h, w, c)
            image_embeds = self.proj(image_embeds)
            # Reshape back to (B, T, H', W', embed_dim)
            _, h_new, w_new, _ = image_embeds.shape
            image_embeds = image_embeds.reshape(b, t, h_new, w_new, self.embed_dim)
            # Flatten spatial dimensions: (B, T*H'*W', embed_dim)
            image_embeds = image_embeds.reshape(b, t * h_new * w_new, self.embed_dim)
        else:
            # CogVideoX 1.5: 3D patching with Linear projection
            b, t, h, w, c = image_embeds.shape
            p_t = self.patch_size_t
            p = self.patch_size
            
            # Calculate output dimensions
            t_new = t // p_t
            h_new = h // p
            w_new = w // p
            
            # Reshape to extract patches: (B, T/p_t, p_t, H/p, p, W/p, p, C)
            image_embeds = image_embeds.reshape(b, t_new, p_t, h_new, p, w_new, p, c)
            # Rearrange: (B, T/p_t, H/p, W/p, p_t, p, p, C)
            image_embeds = image_embeds.transpose(0, 1, 3, 5, 2, 4, 6, 7)
            # Flatten patches: (B, T', H', W', p_t*p*p*C)
            image_embeds = image_embeds.reshape(b, t_new, h_new, w_new, p_t * p * p * c)
            # Flatten spatial dimensions for linear: (B, T'*H'*W', p_t*p*p*C)
            image_embeds = image_embeds.reshape(b, t_new * h_new * w_new, p_t * p * p * c)
            # Apply linear projection: (B, T'*H'*W', embed_dim)
            image_embeds = self.proj(image_embeds)
        
        # ══════════════════════════════════════════════════════════════════════
        # 添加位置编码
        # ══════════════════════════════════════════════════════════════════════
        #
        # 【核心操作】
        #
        # image_embeds = image_embeds + pos_embed
        #
        # 位置编码是**加到** token 嵌入上的，不是拼接！
        # 这样每个 token 的表示 = 内容信息 + 位置信息
        #
        # 【为什么用加法而不是拼接？】
        #
        # 1. 维度不变：(B, 66150, 3072) + (1, 66150, 3072) = (B, 66150, 3072)
        #    如果拼接，维度会变成 (B, 66150, 6144)，增加计算量
        #
        # 2. 研究表明加法效果相当，且更高效
        #
        # 3. 这是 Transformer 原论文的做法，已被广泛验证
        #
        if self.use_positional_embeddings:
            if self.use_learned_positional_embeddings and self.pos_embedding is not None:
                # 可学习位置编码
                # pos_embedding: (1, max_patches, embed_dim)
                # 截取到实际 token 数量（可能比最大值小）
                num_patches = image_embeds.shape[1]
                pos_embed = self.pos_embedding.value[:, :num_patches, :]
                image_embeds = image_embeds + pos_embed  # 广播加法
            elif not self.use_learned_positional_embeddings:
                # 正弦位置编码：运行时计算
                pos_embed = self._get_sinusoidal_pos_embed(image_embeds.shape[1], self.embed_dim)
                # pos_embed: (num_patches, embed_dim)
                # 添加 batch 维度后相加: (1, num_patches, embed_dim)
                image_embeds = image_embeds + pos_embed[None, :, :]
        
        # Concatenate text and image embeddings
        embeds = jnp.concatenate([text_embeds, image_embeds], axis=1)
        
        return embeds
    
    def _get_sinusoidal_pos_embed(self, num_patches: int, embed_dim: int) -> jnp.ndarray:
        """
        生成正弦余弦位置编码。
        
        ═══════════════════════════════════════════════════════════════════════
        【正弦位置编码的数学原理】
        ═══════════════════════════════════════════════════════════════════════
        
        对于位置 pos 和维度 i：
        
            PE(pos, 2i)   = sin(pos / 10000^(2i/d))
            PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
        
        其中 d = embed_dim（如 3072）
        
        ═══════════════════════════════════════════════════════════════════════
        【计算步骤详解】(以 num_patches=66150, embed_dim=3072 为例)
        ═══════════════════════════════════════════════════════════════════════
        
        Step 1: 位置索引
        ─────────────────────────────────────────────────────────────────────
            position = [0, 1, 2, ..., 66149]
            shape: (66150, 1)  # 加一维便于广播
        
        Step 2: 频率因子 (div_term)
        ─────────────────────────────────────────────────────────────────────
            i = [0, 2, 4, ..., 3070]  # 步长为2，共1536个
            
            div_term = exp(-i × ln(10000) / 3072)
                     = 10000^(-i/3072)
            
            div_term[0]    = 10000^0        = 1       (最高频)
            div_term[1]    = 10000^(-2/3072) ≈ 0.997
            div_term[2]    = 10000^(-4/3072) ≈ 0.994
            ...
            div_term[1535] = 10000^(-3070/3072) ≈ 0.0001 (最低频)
            
            shape: (1536,)
        
        Step 3: 计算角度
        ─────────────────────────────────────────────────────────────────────
            angles = position × div_term
            
            通过广播:
            (66150, 1) × (1536,) → (66150, 1536)
            
            例如第100个位置:
            angles[100] = [100×1, 100×0.997, 100×0.994, ..., 100×0.0001]
                        = [100,   99.7,      99.4,      ..., 0.01]
        
        Step 4: 应用三角函数，交替填充
        ─────────────────────────────────────────────────────────────────────
            pe[:, 0::2] = sin(angles)  # 第0,2,4,...列用sin
            pe[:, 1::2] = cos(angles)  # 第1,3,5,...列用cos
            
            最终每一行形如:
            pe[pos] = [sin(pos×ω₀), cos(pos×ω₀),
                       sin(pos×ω₁), cos(pos×ω₁),
                       sin(pos×ω₂), cos(pos×ω₂), ...]
            
            其中 ω₀=1, ω₁≈0.997, ω₂≈0.994, ...
        
        ═══════════════════════════════════════════════════════════════════════
        【为什么 sin 和 cos 交替使用？】
        ═══════════════════════════════════════════════════════════════════════
        
        1. 唯一性保证:
           - 只用 sin: sin(0)=0, sin(π)=0, sin(2π)=0... 不同位置会重复
           - sin + cos 配合: 形成唯一的二维"坐标"
           
        2. 相对位置可学习:
           - PE(pos+k) 可以表示为 PE(pos) 的线性变换
           - 这让模型更容易学习"相隔k个位置"的关系
           
        3. 平滑过渡:
           - 相邻位置的编码向量相似
           - 相距远的位置编码差异大
        
        ═══════════════════════════════════════════════════════════════════════
        【具体数值示例】(简化为 embed_dim=8)
        ═══════════════════════════════════════════════════════════════════════
        
        设 num_patches=5, embed_dim=8
        
        频率因子: ω = [1, 0.1, 0.01, 0.001]  (embed_dim//2 = 4个)
        
        位置0: [sin(0), cos(0), sin(0),   cos(0),   sin(0),    cos(0),    sin(0),     cos(0)]
              = [0,      1,      0,        1,        0,         1,         0,          1]
        
        位置1: [sin(1), cos(1), sin(0.1), cos(0.1), sin(0.01), cos(0.01), sin(0.001), cos(0.001)]
              ≈ [0.84,  0.54,   0.10,     0.995,    0.01,      1.0,       0.001,      1.0]
        
        位置2: [sin(2), cos(2), sin(0.2), cos(0.2), sin(0.02), cos(0.02), sin(0.002), cos(0.002)]
              ≈ [0.91, -0.42,   0.20,     0.98,     0.02,      1.0,       0.002,      1.0]
        
        观察:
        - 高频成分 (左边): 变化快，区分相邻位置
        - 低频成分 (右边): 变化慢，编码整体位置范围
        
        Args:
            num_patches: token 数量，如 66150
            embed_dim: 嵌入维度，如 3072
            
        Returns:
            位置编码矩阵 (num_patches, embed_dim)
        """
        # Step 1: 位置索引 (66150, 1)
        position = jnp.arange(num_patches, dtype=jnp.float32)[:, None]
        
        # Step 2: 频率因子 (1536,)
        # exp(-i × ln(10000) / d) = 10000^(-i/d)
        div_term = jnp.exp(
            jnp.arange(0, embed_dim, 2, dtype=jnp.float32) *
            -(math.log(10000.0) / embed_dim)
        )
        
        # Step 3 & 4: 计算并填充 sin/cos
        pe = jnp.zeros((num_patches, embed_dim))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))  # 偶数列: sin
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))  # 奇数列: cos
        
        return pe


class FlaxCogVideoXLayerNormZero(nnx.Module):
    """
    CogVideoX 的条件调制LayerNorm模块 - 核心的条件注入机制。
    
    【这个类的作用 - Diffusion模型中的条件调制】
    
    在Diffusion模型中，我们希望模型在不同的去噪阶段有不同的行为：
    - 早期阶段（t大）：噪声多，需要更激进的去噪
    - 后期阶段（t小）：接近原图，需要精细调整
    
    这个模块使用时间步嵌入来**调制**（modulate）LayerNorm的输出：
    
        调制公式: output = LayerNorm(x) × (1 + scale) + shift
    
    - scale: 缩放因子，控制特征的"强度"
    - shift: 偏移量，调整特征的"中心点"
    - gate: 门控因子，控制残差连接的强度
    
    【为什么输出6个参数？】
    
    因为这个模块同时处理注意力层和FFN层的两个分支：
    (以5B模型 inner_dim=3072 为例)
    
        temb → Linear(512 → 6 × 3072) → split成6份
                     ↓
        ┌────────────┴────────────┐
        │    Attention分支         │    FFN分支
        ├─ shift_msa (3072)       ├─ shift_mlp (3072)
        ├─ scale_msa (3072)       ├─ scale_mlp (3072)
        └─ gate_msa  (3072)       └─ gate_mlp  (3072)
    
    msa = Multi-head Self-Attention
    mlp = Multi-Layer Perceptron (即FFN)
    
    【在CogVideoXBlock中的使用方式】
    
        # 1. 生成调制参数
        norm_hidden, norm_enc, gate_attn, gate_enc = self.norm1(hidden, encoder, temb)
        
        # 2. 注意力计算（使用调制后的特征）
        attn_out = self.attn1(norm_hidden, norm_enc)
        
        # 3. 残差连接（使用gate控制强度）
        hidden = hidden + gate_attn * attn_out  ← gate决定"采纳多少新信息"
    
    【直觉理解】
    
    想象你在画画：
    - scale: 控制画笔的粗细
    - shift: 控制画布的位置
    - gate: 控制这一笔要画多用力（0=不画，1=正常力度）
    
    在去噪早期，gate可能较大（大刀阔斧）；后期gate较小（精雕细琢）。
    模型通过学习自动决定每个时间步的最优策略。
    """
    
    def __init__(
        self,
        time_embed_dim: int,    # 时间嵌入维度 = 512
        dim: int,               # 特征维度 = inner_dim (2B:1920, 5B:3072)
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        bias: bool = True,
        rngs: nnx.Rngs = None,
    ):
        # 标准LayerNorm，用于归一化
        self.norm = nnx.LayerNorm(dim, epsilon=eps, use_bias=bias, use_scale=elementwise_affine, rngs=rngs)
        
        # ──────────────────────────────────────────────────────────────
        # 调制参数生成器
        # 输入: 时间嵌入 (B, 512)
        # 输出: 6组调制参数 (B, 6 × inner_dim)
        #       5B模型: (B, 6 × 3072 = 18432)
        # ──────────────────────────────────────────────────────────────
        self.linear = nnx.Linear(time_embed_dim, 6 * dim, use_bias=True, rngs=rngs)
    
    def __call__(
        self,
        hidden_states: jnp.ndarray,           # 视频特征 (B, num_patches, inner_dim)
        encoder_hidden_states: jnp.ndarray,   # 文本特征 (B, text_len, inner_dim)
        temb: jnp.ndarray,                    # 时间嵌入 (B, 512)
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        调制LayerNorm前向传播。
        
        【计算流程】(以5B模型 inner_dim=3072 为例)
        
        Step 1: 从时间嵌入生成6组调制参数
            temb (B, 512)
                ↓ SiLU激活
            (B, 512)
                ↓ Linear(512 → 18432)  # 6 × 3072
            (B, 18432)
                ↓ split成6份
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
            每个都是 (B, 3072)
        
        Step 2: 对视频特征应用LayerNorm + 调制
            hidden_states (B, num_patches, 3072)
                ↓ LayerNorm
            normalized (B, num_patches, 3072)
                ↓ × (1 + scale_msa) + shift_msa
            modulated_hidden (B, num_patches, 3072)
        
        Step 3: 对文本特征应用相同的LayerNorm + 不同的调制参数
            encoder_hidden_states (B, text_len, 3072)
                ↓ LayerNorm  (共享同一个LayerNorm!)
            normalized (B, text_len, 3072)
                ↓ × (1 + scale_mlp) + shift_mlp
            modulated_encoder (B, text_len, 3072)
        
        Returns:
            - norm_hidden_states: 调制后的视频特征
            - norm_encoder_hidden_states: 调制后的文本特征
            - gate_msa: 视频的残差门控 (用于注意力输出)
            - gate_mlp: 文本的残差门控 (用于FFN输出)
        """
        # ──────────────────────────────────────────────────────────────
        # Step 1: 生成6组调制参数
        # ──────────────────────────────────────────────────────────────
        # SiLU激活 + Linear投影 + 切分
        # 为什么用SiLU？因为它是平滑的非线性，在Diffusion模型中效果好
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(
            self.linear(jax.nn.silu(temb)),  # (B, 512) → (B, 6 × inner_dim)
            6,                                # 切分成6份
            axis=-1                           # 沿最后一个维度切
        )
        # 每个参数现在是 (B, inner_dim)，例如5B模型: (B, 3072)
        
        # ──────────────────────────────────────────────────────────────
        # Step 2: 视频特征的LayerNorm + 调制
        # ──────────────────────────────────────────────────────────────
        norm_hidden_states = self.norm(hidden_states)  # (B, num_patches, inner_dim)
        # 调制公式: x_new = x × (1 + scale) + shift
        # scale和shift需要广播: (B, inner_dim) → (B, 1, inner_dim) 以匹配 (B, seq_len, inner_dim)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :]
        
        # ──────────────────────────────────────────────────────────────
        # Step 3: 文本特征的LayerNorm + 调制
        # ──────────────────────────────────────────────────────────────
        # 注意：文本和视频共享同一个LayerNorm，但使用不同的调制参数
        # 这是一种参数共享策略，减少参数量同时保持灵活性
        norm_encoder_hidden_states = self.norm(encoder_hidden_states)
        norm_encoder_hidden_states = (
            norm_encoder_hidden_states * (1 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
        )
        
        # 返回调制后的特征和门控参数
        # 门控也需要广播到序列维度: (B, inner_dim) → (B, 1, inner_dim)
        return norm_hidden_states, norm_encoder_hidden_states, gate_msa[:, None, :], gate_mlp[:, None, :]


class FlaxAdaLayerNorm(nnx.Module):
    """
    自适应LayerNorm - 用于模型输出之前的最终调制。
    
    【这个类的作用】
    
    这是一个简化版的调制LayerNorm，只输出2个参数（shift和scale）。
    用在模型输出之前，对最终特征进行调制。
    
    与FlaxCogVideoXLayerNormZero的区别：
    - LayerNormZero: 输出6个参数，用于Transformer块内部
    - AdaLayerNorm: 输出2个参数，用于最终输出
    
    【在模型中的位置】
    
        Transformer Blocks × N (2B: 30层, 5B: 42层)
              ↓
        norm_final (普通LayerNorm)
              ↓
        norm_out (FlaxAdaLayerNorm) ← 你在这里
              ↓
        proj_out (Linear投影回patch维度)
              ↓
        Unpatchify
    
    【为什么最后还需要调制？】
    
    即使经过了30层Transformer块的处理，最终输出仍然需要根据
    时间步进行调整。这确保了模型在所有阶段都能利用时间信息。
    """
    
    def __init__(
        self,
        embedding_dim: int,      # 时间嵌入维度 = 512
        output_dim: int,         # 输出维度 = 2 × inner_dim (5B: 2×3072=6144)
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        rngs: nnx.Rngs = None,
    ):
        self.silu = jax.nn.silu
        # 投影到 output_dim = 2 × inner_dim
        # 这样可以切分成 shift 和 scale 两部分
        self.linear = nnx.Linear(embedding_dim, output_dim, rngs=rngs)
        # LayerNorm的维度是 output_dim // 2 = inner_dim
        self.norm = nnx.LayerNorm(output_dim // 2, epsilon=eps, use_scale=elementwise_affine, rngs=rngs)
    
    def __call__(
        self,
        hidden_states: jnp.ndarray,  # (B, seq_len, inner_dim)
        temb: jnp.ndarray,           # (B, 512)
    ) -> jnp.ndarray:
        """
        最终输出的调制。
        
        【计算流程】(以5B模型 inner_dim=3072 为例)
        
            temb (B, 512)
              ↓ SiLU
            (B, 512)
              ↓ Linear(512 → 6144)  # 2 × 3072
            (B, 6144)
              ↓ split成2份
            shift (B, 3072), scale (B, 3072)
        
            hidden_states (B, seq_len, 3072)
              ↓ LayerNorm
            normalized (B, seq_len, 3072)
              ↓ × (1 + scale) + shift
            output (B, seq_len, 3072)
        """
        # 生成shift和scale
        emb = self.linear(self.silu(temb))  # (B, 512) → (B, 2 × inner_dim)
        shift, scale = jnp.split(emb, 2, axis=-1)  # 各 (B, inner_dim)
        
        # 归一化 + 调制
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states * (1 + scale[:, None, :]) + shift[:, None, :]
        
        return hidden_states


# ======================================================================================
# Splash Attention - TPU 优化的注意力实现
# ======================================================================================
"""
【Splash Attention 是什么？】

Splash Attention 是 Google 开发的、专门为 TPU 优化的注意力计算实现。
它解决了长序列注意力计算的内存和计算效率问题。

【为什么需要它？】

标准注意力的问题：
- 内存占用: O(seq_len²) - 对于 66000+ tokens 的视频，需要 ~17GB 的注意力矩阵
- HBM带宽瓶颈: 需要反复读写巨大的注意力权重矩阵

Splash Attention 的解决方案：
- 分块计算: 把大矩阵分成小块，逐块计算
- 融合内核: 把 QK^T → softmax → V 融合成一个操作
- 永不具体化完整注意力矩阵，内存占用降至 O(seq_len)

【性能提升】

在 CogVideoX 的视频生成场景中：
- 序列长度 ~66,000 tokens
- 标准注意力：OOM 或极慢
- Splash Attention：可以正常运行，速度提升 2-3x

【关键配置参数】
"""

# Splash Attention 分块大小配置
BQSIZE = 2048          # Query 分块大小 - 每次处理2048个query tokens
BKVSIZE = 2048         # Key/Value 分块大小 - 每次处理2048个kv tokens
BKVCOMPUTESIZE = 1024  # 计算块大小 - 内部计算时的细分粒度
USE_K_SMOOTH = True    # 是否使用 K-smooth 优化（见下方解释）

"""
【K-smooth 优化是什么？】

在注意力计算中，softmax(QK^T/√d) 可能因为数值范围大而导致精度问题。
K-smooth 通过减去 key 的均值来稳定数值：

    key_smooth = key - mean(key)
    
这不改变最终结果（因为softmax对常数偏移不敏感），但能提高数值稳定性。
"""

# 全局 mesh 变量，用于多设备分片
# Mesh 定义了设备的逻辑拓扑，例如 2x4 的 TPU pod
_GLOBAL_MESH = None

def set_global_mesh(mesh: Mesh):
    """
    设置全局 mesh，用于多设备分片。
    
    在使用多个 TPU 设备时，需要定义设备的逻辑布局。
    例如，8个 TPU 核心可以组织成 2x4 的网格：
    
        mesh = Mesh(devices, axis_names=('dp', 'tp'))
        set_global_mesh(mesh)
    
    其中：
    - 'dp' = data parallel（数据并行维度）
    - 'tp' = tensor parallel（张量并行维度）
    """
    global _GLOBAL_MESH
    _GLOBAL_MESH = mesh

def get_global_mesh() -> Mesh:
    """获取全局 mesh"""
    return _GLOBAL_MESH


def _create_splash_attention_kernel(padded_q_seq_len, padded_kv_seq_len, num_heads_on_device, window_size=None):
    """
    创建 Splash Attention 的计算内核。
    
    【参数说明】
    - padded_q_seq_len: 填充后的query序列长度（需要是块大小的倍数）
    - padded_kv_seq_len: 填充后的key/value序列长度
    - num_heads_on_device: 当前设备上的注意力头数
    - window_size: 局部注意力的窗口大小（None表示全局注意力）
    
    【Mask类型】
    - FullMask: 全局注意力，每个位置可以看到所有位置
    - LocalMask: 局部注意力，每个位置只能看到窗口内的位置
    
    CogVideoX使用FullMask，因为视频生成需要全局上下文。
    """
    if window_size is not None:
        mask_class = functools.partial(splash_attention.LocalMask, window_size=window_size, offset=0)
    else:
        mask_class = splash_attention.FullMask

    mask = splash_attention.MultiHeadMask(
        [mask_class((padded_q_seq_len, padded_kv_seq_len)) for _ in range(num_heads_on_device)]
    )

    block_sizes = splash_attention.BlockSizes(
        block_q=min(BQSIZE, padded_q_seq_len),
        block_kv=min(BKVSIZE, padded_kv_seq_len),
        block_kv_compute=min(BKVCOMPUTESIZE, padded_kv_seq_len),
    )
    
    return splash_attention.make_splash_mha(
        mask=mask, block_sizes=block_sizes, head_shards=1, q_seq_shards=1
    )


def _tpu_splash_attention_single_device(query, key, value, scale=None, use_k_smooth=True):
    """TPU Splash Attention 实现（单设备版本）"""
    
    # 可选的 K-smooth 处理
    if use_k_smooth:
        key_mean = jnp.mean(key, axis=2, keepdims=True)
        key = key - key_mean

    # 缩放 query 张量
    scale_factor = 1.0 / math.sqrt(query.shape[-1]) if scale is None else scale
    query = query * scale_factor

    def pad_to_multiple(x, multiple, axis):
        seq_len = x.shape[axis]
        pad_len = (multiple - seq_len % multiple) % multiple
        if pad_len == 0:
            return x, seq_len
        pad_width = [(0, 0)] * x.ndim
        pad_width[axis] = (0, pad_len)
        return jnp.pad(x, pad_width), seq_len

    def kernel_3d(q_3d, k_3d, v_3d):
        num_heads_on_device = q_3d.shape[0]

        # 填充到块大小的倍数
        q_3d_padded, q_orig_len = pad_to_multiple(q_3d, BQSIZE, axis=1)
        k_3d_padded, _ = pad_to_multiple(k_3d, BKVSIZE, axis=1)
        v_3d_padded, _ = pad_to_multiple(v_3d, BKVSIZE, axis=1)

        padded_q_seq_len = int(q_3d_padded.shape[1])
        padded_kv_seq_len = int(k_3d_padded.shape[1])

        # 创建并执行 Splash attention kernel
        splash_kernel = _create_splash_attention_kernel(
            padded_q_seq_len, padded_kv_seq_len, num_heads_on_device, window_size=None
        )
        out = splash_kernel(q_3d_padded, k_3d_padded, v_3d_padded)
        
        # 移除填充
        return out[:, :q_orig_len, ...]

    # 在批次维度上映射 kernel
    vmapped_kernel = jax.vmap(kernel_3d, in_axes=(0, 0, 0), out_axes=0)
    return vmapped_kernel(query, key, value)


def _tpu_splash_attention_sharded(query, key, value, mesh, scale=None, use_k_smooth=True):
    """TPU Splash Attention 实现（多设备分片版本）
    
    使用 shard_map 在多设备间分片执行 Splash Attention。
    注意：不使用 lru_cache 缓存 kernel，以避免 tracer leak。
    """
    
    # 可选的 K-smooth 处理（在 shard_map 外部执行）
    if use_k_smooth:
        key_mean = jnp.mean(key, axis=2, keepdims=True)
        key = key - key_mean

    # 缩放 query 张量（在 shard_map 外部执行）
    scale_factor = 1.0 / math.sqrt(query.shape[-1]) if scale is None else scale
    query = query * scale_factor

    num_heads = query.shape[1]

    def _attention_on_slices(q, k, v):
        """在单个设备切片上执行 attention"""
        
        def pad_to_multiple(x, multiple, axis):
            seq_len = x.shape[axis]
            pad_len = (multiple - seq_len % multiple) % multiple
            if pad_len == 0:
                return x, seq_len
            pad_width = [(0, 0)] * x.ndim
            pad_width[axis] = (0, pad_len)
            return jnp.pad(x, pad_width), seq_len

        def kernel_3d(q_3d, k_3d, v_3d):
            num_heads_on_device = q_3d.shape[0]

            # 填充到块大小的倍数
            q_3d_padded, q_orig_len = pad_to_multiple(q_3d, BQSIZE, axis=1)
            k_3d_padded, _ = pad_to_multiple(k_3d, BKVSIZE, axis=1)
            v_3d_padded, _ = pad_to_multiple(v_3d, BKVSIZE, axis=1)

            padded_q_seq_len = q_3d_padded.shape[1]
            padded_kv_seq_len = k_3d_padded.shape[1]

            # 直接创建 mask 和 kernel（不使用缓存）
            mask = splash_attention.MultiHeadMask(
                [splash_attention.FullMask((padded_q_seq_len, padded_kv_seq_len))
                 for _ in range(num_heads_on_device)]
            )
            
            block_sizes = splash_attention.BlockSizes(
                block_q=min(BQSIZE, padded_q_seq_len),
                block_kv=min(BKVSIZE, padded_kv_seq_len),
                block_kv_compute=min(BKVCOMPUTESIZE, padded_kv_seq_len),
            )
            
            splash_kernel = splash_attention.make_splash_mha(
                mask=mask, block_sizes=block_sizes, head_shards=1, q_seq_shards=1
            )
            out = splash_kernel(q_3d_padded, k_3d_padded, v_3d_padded)
            
            # 移除填充
            return out[:, :q_orig_len, ...]

        # 在批次维度上映射 kernel
        vmapped_kernel = jax.vmap(kernel_3d, in_axes=(0, 0, 0), out_axes=0)
        return vmapped_kernel(q, k, v)

    # 根据设备数量和头数确定分片策略
    if num_heads < mesh.size:
        q_partition_spec = P()
        kv_partition_spec = P()
    else:
        if query.shape[2] == key.shape[2]:  # 自注意力
            q_partition_spec = P('dp', 'tp', 'sp', None)
            kv_partition_spec = P('dp', 'tp', None, None)
        else:  # 交叉注意力
            q_partition_spec = P('dp', None, ('tp', 'sp'), None)
            kv_partition_spec = P('dp', None, None, None)

    # 使用 shard_map 在设备间分片执行
    sharded_fn = shard_map(
        _attention_on_slices,
        mesh=mesh,
        in_specs=(q_partition_spec, kv_partition_spec, kv_partition_spec),
        out_specs=q_partition_spec,
        check_rep=False,
    )
    out = sharded_fn(query, key, value)
    
    return out


def _tpu_splash_attention(query, key, value, scale=None, use_k_smooth=True):
    """TPU Splash Attention 统一入口
    
    自动根据全局 mesh 配置选择单设备或多设备分片版本。
    """
    mesh = get_global_mesh()
    
    if mesh is not None and mesh.size > 1:
        # 多设备分片版本
        return _tpu_splash_attention_sharded(query, key, value, mesh, scale=scale, use_k_smooth=use_k_smooth)
    else:
        # 单设备版本
        return _tpu_splash_attention_single_device(query, key, value, scale=scale, use_k_smooth=use_k_smooth)


class FlaxAttention(nnx.Module):
    """
    CogVideoX 的多头注意力模块 - 实现文本和视频的联合自注意力。
    
    【这个类的作用】
    
    这是 Transformer 的核心组件。在 CogVideoX 中，它有一个特殊的设计：
    **联合注意力**（Joint Attention）—— 文本和视频 tokens 拼接在一起做自注意力。
    
    【联合注意力的工作原理】(以5B模型 inner_dim=3072 为例)
    
        文本tokens (B, 226, 3072)    视频tokens (B, 66150, 3072)
                    ↘                     ↙
                 拼接: (B, 66376, 3072)
                           ↓
              ┌─────────────────────────────────┐
              │ Q = Linear(hidden_states)       │
              │ K = Linear(hidden_states)       │
              │ V = Linear(hidden_states)       │
              └─────────────────────────────────┘
                           ↓
              ┌─────────────────────────────────┐
              │ Attention(Q, K, V)              │
              │ 使用 Splash Attention (TPU优化) │
              │                                 │
              │ 每个token可以attend到所有tokens  │
              │ 包括文本和视频的交互              │
              └─────────────────────────────────┘
                           ↓
                 分割回文本和视频部分
                    ↙                     ↘
        文本输出 (B, 226, 3072)    视频输出 (B, 66150, 3072)
    
    【为什么用联合注意力而不是交叉注意力？】
    
    交叉注意力（Cross-Attention）:
        - 视频做Q，文本做K/V
        - 信息单向流动：文本 → 视频
    
    联合注意力（Joint Attention）:
        - 文本和视频都做Q/K/V
        - 信息双向流动：文本 ↔ 视频
        - 更强的建模能力，但计算量更大
    
    【QK Normalization】
    
    对 Query 和 Key 应用 LayerNorm，提高训练稳定性：
        Q_norm = LayerNorm(Q)
        K_norm = LayerNorm(K)
        attention = softmax(Q_norm @ K_norm^T / √d)
    
    【旋转位置编码 (RoPE)】
    
    可选地对视频 tokens 应用 RoPE，编码空间位置信息。
    注意：只对视频 tokens 应用，不对文本 tokens 应用。
    """
    
    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        out_bias: bool = True,
        qk_norm: Optional[str] = None,
        eps: float = 1e-6,
        use_splash_attention: bool = True,
        rngs: nnx.Rngs = None,
    ):
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head
        self.scale = dim_head ** -0.5
        self.use_splash_attention = use_splash_attention
        
        # Query, Key, Value projections
        self.to_q = nnx.Linear(query_dim, inner_dim, use_bias=bias, rngs=rngs)
        self.to_k = nnx.Linear(query_dim, inner_dim, use_bias=bias, rngs=rngs)
        self.to_v = nnx.Linear(query_dim, inner_dim, use_bias=bias, rngs=rngs)
        
        # QK normalization
        self.qk_norm = qk_norm
        if qk_norm == "layer_norm":
            self.q_norm = nnx.LayerNorm(dim_head, epsilon=eps, rngs=rngs)
            self.k_norm = nnx.LayerNorm(dim_head, epsilon=eps, rngs=rngs)
        else:
            self.q_norm = None
            self.k_norm = None
        
        # Output projection
        self.to_out = nnx.Linear(inner_dim, query_dim, use_bias=out_bias, rngs=rngs)
        self.dropout_rate = dropout
    
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        image_rotary_emb: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Args:
            hidden_states: Video features (B, num_patches, dim)
            encoder_hidden_states: Text features (B, text_seq_len, dim)
            image_rotary_emb: Rotary embeddings (cos, sin)
            deterministic: Whether to use dropout
            
        Returns:
            Tuple of (attn_hidden_states, attn_encoder_hidden_states)
        """
        batch_size = hidden_states.shape[0]
        
        # Concatenate text and video for joint attention
        if encoder_hidden_states is not None:
            text_seq_length = encoder_hidden_states.shape[1]
            hidden_states = jnp.concatenate([encoder_hidden_states, hidden_states], axis=1)
        else:
            text_seq_length = 0
        
        # Project to Q, K, V
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)
        
        # Reshape to multi-head
        # (B, seq_len, heads, dim_head)
        query = query.reshape(batch_size, -1, self.heads, self.dim_head)
        key = key.reshape(batch_size, -1, self.heads, self.dim_head)
        value = value.reshape(batch_size, -1, self.heads, self.dim_head)
        
        # Apply QK normalization
        if self.q_norm is not None:
            query = self.q_norm(query)
        if self.k_norm is not None:
            key = self.k_norm(key)
        
        # Apply rotary embeddings if provided
        if image_rotary_emb is not None:
            # Only apply to video tokens (after text tokens)
            cos, sin = image_rotary_emb
            
            # Split into text and video parts
            if text_seq_length > 0:
                query_text = query[:, :text_seq_length]
                query_video = query[:, text_seq_length:]
                key_text = key[:, :text_seq_length]
                key_video = key[:, text_seq_length:]
                
                # Apply rotary to video tokens
                query_video = self._apply_rotary_emb(query_video, cos, sin)
                key_video = self._apply_rotary_emb(key_video, cos, sin)
                
                # Concatenate back
                query = jnp.concatenate([query_text, query_video], axis=1)
                key = jnp.concatenate([key_text, key_video], axis=1)
            else:
                query = self._apply_rotary_emb(query, cos, sin)
                key = self._apply_rotary_emb(key, cos, sin)
        
        # Attention computation
        if self.use_splash_attention:
            # Use Splash Attention (TPU-optimized)
            # Transpose to (B, heads, seq_len, dim_head)
            query = query.transpose(0, 2, 1, 3)
            key = key.transpose(0, 2, 1, 3)
            value = value.transpose(0, 2, 1, 3)
            
            # Apply Splash Attention
            attn_output = _tpu_splash_attention(
                query, key, value,
                scale=self.scale,
                use_k_smooth=USE_K_SMOOTH
            )
            
            # Reshape back: (B, seq_len, heads * dim_head)
            attn_output = attn_output.transpose(0, 2, 1, 3)
            attn_output = attn_output.reshape(batch_size, -1, self.heads * self.dim_head)
        else:
            # Standard attention
            # Transpose for attention: (B, heads, seq_len, dim_head)
            query = query.transpose(0, 2, 1, 3)
            key = key.transpose(0, 2, 1, 3)
            value = value.transpose(0, 2, 1, 3)
            
            # Attention
            attn_weights = jnp.matmul(query, key.transpose(0, 1, 3, 2)) * self.scale
            attn_weights = jax.nn.softmax(attn_weights, axis=-1)
            
            # Dropout
            if self.dropout_rate > 0 and not deterministic:
                attn_weights = nnx.Dropout(rate=self.dropout_rate)(attn_weights)
            
            # Attend to values
            attn_output = jnp.matmul(attn_weights, value)
            
            # Reshape back: (B, seq_len, heads * dim_head)
            attn_output = attn_output.transpose(0, 2, 1, 3)
            attn_output = attn_output.reshape(batch_size, -1, self.heads * self.dim_head)
        
        # Output projection
        attn_output = self.to_out(attn_output)
        
        # Split back into text and video
        if text_seq_length > 0:
            attn_encoder_hidden_states = attn_output[:, :text_seq_length]
            attn_hidden_states = attn_output[:, text_seq_length:]
            return attn_hidden_states, attn_encoder_hidden_states
        else:
            return attn_output, None
    
    def _apply_rotary_emb(
        self,
        x: jnp.ndarray,
        cos: jnp.ndarray,
        sin: jnp.ndarray,
    ) -> jnp.ndarray:
        """Apply rotary position embeddings."""
        # x: (B, seq_len, heads, dim_head)
        # cos, sin: (seq_len, dim_head) or similar
        
        # Ensure cos/sin have correct shape
        if cos.ndim == 2:
            cos = cos[None, :, None, :]  # (1, seq_len, 1, dim_head)
            sin = sin[None, :, None, :]
        
        # Rotate
        x_rot = jnp.concatenate([-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]], axis=-1)
        x_out = x * cos + x_rot * sin
        
        return x_out


class FlaxFeedForward(nnx.Module):
    """
    前馈网络 (FFN) - Transformer 中的另一个核心组件。
    
    【这个类的作用】
    
    FFN 是一个简单的两层 MLP，用于对每个 token 独立地进行非线性变换。
    它通常先升维再降维，增加模型的表达能力。
    
    【网络结构】(以5B模型 inner_dim=3072 为例)
    
        输入 (B, seq_len, 3072)
              ↓
        Linear(3072 → 12288)  ← 升维4倍
              ↓
        GELU激活函数
              ↓
        (可选 Dropout)
              ↓
        Linear(12288 → 3072)  ← 降回原维度
              ↓
        (可选 Dropout)
              ↓
        输出 (B, seq_len, 3072)
    
    【为什么需要 FFN？】
    
    Attention 层主要负责 tokens 之间的信息交换，
    FFN 则负责对每个 token 的特征进行非线性变换。
    
    两者配合：
    - Attention: "我要从其他token那里获取什么信息？"
    - FFN: "得到信息后，我要怎么处理它？"
    
    【GELU vs ReLU】
    
    GELU(x) = x * Φ(x)，其中 Φ 是标准正态分布的CDF
    
    比 ReLU 更平滑，在 Transformer 模型中效果更好。
    """
    
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        final_dropout: bool = False,
        inner_dim: Optional[int] = None,
        bias: bool = True,
        rngs: nnx.Rngs = None,
    ):
        if inner_dim is None:
            inner_dim = int(dim * mult)
        
        dim_out = dim_out or dim
        
        # Activation function
        if activation_fn == "gelu" or activation_fn == "gelu-approximate":
            self.act = jax.nn.gelu
        elif activation_fn == "silu":
            self.act = jax.nn.silu
        else:
            raise ValueError(f"Unsupported activation: {activation_fn}")
        
        # Layers - use direct attributes instead of list
        self.linear1 = nnx.Linear(dim, inner_dim, use_bias=bias, rngs=rngs)
        self.linear2 = nnx.Linear(inner_dim, dim_out, use_bias=bias, rngs=rngs)
        
        if dropout > 0:
            self.dropout1 = nnx.Dropout(rate=dropout)
        else:
            self.dropout1 = None
        
        if final_dropout and dropout > 0:
            self.dropout2 = nnx.Dropout(rate=dropout)
        else:
            self.dropout2 = None
    
    def __call__(self, hidden_states: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """
        Args:
            hidden_states: Input features (B, seq_len, dim)
            deterministic: Whether to use dropout
            
        Returns:
            Output features (B, seq_len, dim_out)
        """
        # First linear layer
        hidden_states = self.linear1(hidden_states)
        hidden_states = self.act(hidden_states)
        
        if self.dropout1 is not None and not deterministic:
            hidden_states = self.dropout1(hidden_states)
        
        # Second linear layer
        hidden_states = self.linear2(hidden_states)
        
        if self.dropout2 is not None and not deterministic:
            hidden_states = self.dropout2(hidden_states)
        
        return hidden_states


class FlaxCogVideoXBlock(nnx.Module):
    """
    CogVideoX 的 Transformer 块 - 模型的核心重复单元。
    
    【这个类的作用】
    
    这是 CogVideoX 的基本构建块。
    - CogVideoX-2B: 30 个块
    - CogVideoX-5B: 42 个块
    
    每个块包含：
    1. 带时间调制的 LayerNorm + 联合注意力
    2. 带时间调制的 LayerNorm + FFN
    
    【块的结构图】
    
        hidden_states (视频)          encoder_hidden_states (文本)
              ↓                                    ↓
        ╔════════════════════════════════════════════════════════════╗
        ║                   norm1 (LayerNormZero)                    ║
        ║    用时间嵌入生成 scale, shift, gate 调制参数               ║
        ╚════════════════════════════════════════════════════════════╝
              ↓ (调制后)                           ↓ (调制后)
              └──────────────────┬─────────────────┘
                                 ↓
        ╔════════════════════════════════════════════════════════════╗
        ║                   attn1 (联合注意力)                        ║
        ║    文本和视频拼接后一起做自注意力                            ║
        ╚════════════════════════════════════════════════════════════╝
                                 ↓
              ┌──────────────────┴─────────────────┐
              ↓ (分割)                             ↓ (分割)
        hidden += gate * attn_out     encoder += gate * attn_out
              ↓                                    ↓
        ╔════════════════════════════════════════════════════════════╗
        ║                   norm2 (LayerNormZero)                    ║
        ║    再次用时间嵌入调制                                       ║
        ╚════════════════════════════════════════════════════════════╝
              ↓                                    ↓
              └──────────────────┬─────────────────┘
                       拼接后送入 FFN
                                 ↓
        ╔════════════════════════════════════════════════════════════╗
        ║                   ff (前馈网络)                             ║
        ║    对拼接后的序列进行非线性变换                              ║
        ╚════════════════════════════════════════════════════════════╝
                                 ↓
              ┌──────────────────┴─────────────────┐
              ↓ (分割)                             ↓ (分割)
        hidden += gate * ff_out       encoder += gate * ff_out
              ↓                                    ↓
        输出 hidden_states                 输出 encoder_hidden_states
    
    【门控残差连接】
    
    普通残差连接: output = input + sublayer(input)
    门控残差连接: output = input + gate * sublayer(input)
    
    gate 由时间嵌入控制，让模型可以动态调整每个块的"贡献程度"。
    在去噪早期可能 gate 较大，后期 gate 较小。
    """
    
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = False,
        qk_norm: bool = True,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = True,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
        rngs: nnx.Rngs = None,
    ):
        # 1. Self Attention
        self.norm1 = FlaxCogVideoXLayerNormZero(
            time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True, rngs=rngs
        )
        
        self.attn1 = FlaxAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
            rngs=rngs,
        )
        
        # 2. Feed Forward
        self.norm2 = FlaxCogVideoXLayerNormZero(
            time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True, rngs=rngs
        )
        
        self.ff = FlaxFeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
            rngs=rngs,
        )
    
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        encoder_hidden_states: jnp.ndarray,
        temb: jnp.ndarray,
        image_rotary_emb: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Args:
            hidden_states: Video features (B, num_patches, dim)
            encoder_hidden_states: Text features (B, text_seq_len, dim)
            temb: Time embeddings (B, time_embed_dim)
            image_rotary_emb: Rotary position embeddings
            deterministic: Whether to use dropout
            
        Returns:
            Tuple of (hidden_states, encoder_hidden_states)
        """
        text_seq_length = encoder_hidden_states.shape[1]
        
        # Norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
            hidden_states, encoder_hidden_states, temb
        )
        
        # Attention
        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            deterministic=deterministic,
        )
        
        hidden_states = hidden_states + gate_msa * attn_hidden_states
        encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states
        
        # Norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
            hidden_states, encoder_hidden_states, temb
        )
        
        # Feed-forward
        norm_hidden_states = jnp.concatenate([norm_encoder_hidden_states, norm_hidden_states], axis=1)
        ff_output = self.ff(norm_hidden_states, deterministic=deterministic)
        
        hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:]
        encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]
        
        return hidden_states, encoder_hidden_states


class FlaxCogVideoXTransformer3DModel(nnx.Module):
    """
    CogVideoX 3D Transformer 主模型 - 视频生成的核心。
    
    ═══════════════════════════════════════════════════════════════════════════
    【模型概述】
    ═══════════════════════════════════════════════════════════════════════════
    
    这是 CogVideoX 视频生成模型的 Transformer 核心，负责在 Diffusion 过程中
    预测噪声。模型接收：
    - 带噪声的视频 latents
    - 文本编码
    - 当前时间步
    
    输出去噪后的视频 latents。
    
    ═══════════════════════════════════════════════════════════════════════════
    【完整前向传播流程】
    ═══════════════════════════════════════════════════════════════════════════
    
    输入:
        hidden_states: (B, T, H, W, C) = (1, 49, 60, 90, 16) ← 视频latents
        encoder_hidden_states: (B, 226, 4096) ← T5文本编码
        timestep: 500 ← 当前去噪步骤
    
    Step 1: 时间步嵌入 (以5B模型 inner_dim=3072 为例)
        ┌─────────────────────────────────────────────────────────────────┐
        │ timestep = 500                                                  │
        │     ↓                                                           │
        │ time_proj (FlaxTimesteps): 正弦编码 → (B, 3072)                 │
        │     ↓                                  ↑ inner_dim，不是512！    │
        │ time_embedding (FlaxTimestepEmbedding): MLP → (B, 512)          │
        │     ↓                                  ↑ 这才是time_embed_dim    │
        │ emb = 时间嵌入，用于调制所有层                                   │
        └─────────────────────────────────────────────────────────────────┘
    
    Step 2: Patch 嵌入 (以5B模型 inner_dim=3072 为例)
        ┌─────────────────────────────────────────────────────────────────┐
        │ patch_embed (FlaxCogVideoXPatchEmbed):                          │
        │                                                                 │
        │ 视频: (B, 49, 60, 90, 16) → patchify → (B, 66150, 3072)        │
        │ 文本: (B, 226, 4096) → Linear → (B, 226, 3072)                  │
        │                                                                 │
        │ 拼接: (B, 226 + 66150, 3072) = (B, 66376, 3072)                 │
        │                                                                 │
        │ 然后分开:                                                       │
        │   encoder_hidden_states = 前226个tokens（文本）                 │
        │   hidden_states = 后66150个tokens（视频）                        │
        └─────────────────────────────────────────────────────────────────┘
    
    Step 3: Transformer 块 × N (2B: 30层, 5B: 42层)
        ┌─────────────────────────────────────────────────────────────────┐
        │ for block in transformer_blocks:                                │
        │     hidden_states, encoder_hidden_states = block(               │
        │         hidden_states,        # 视频特征                        │
        │         encoder_hidden_states, # 文本特征                        │
        │         temb=emb,             # 时间嵌入                         │
        │     )                                                           │
        │                                                                 │
        │ 每个块内部：                                                     │
        │   - LayerNormZero + 联合注意力 + 门控残差                        │
        │   - LayerNormZero + FFN + 门控残差                               │
        └─────────────────────────────────────────────────────────────────┘
    
    Step 4: 最终输出 (以5B模型 inner_dim=3072 为例)
        ┌─────────────────────────────────────────────────────────────────┐
        │ hidden_states: (B, 66150, 3072)                                 │
        │     ↓                                                           │
        │ norm_final (LayerNorm)                                          │
        │     ↓                                                           │
        │ norm_out (AdaLayerNorm): 最后一次时间调制                        │
        │     ↓                                                           │
        │ proj_out (Linear): (B, 66150, 3072) → (B, 66150, 64)            │
        │     ↓               ↑ inner_dim           ↑ patch_size²×out_ch   │
        │ unpatchify: 把tokens重组回视频格式                               │
        │     ↓                                                           │
        │ output: (B, 49, 60, 90, 16)                                     │
        └─────────────────────────────────────────────────────────────────┘
    
    ═══════════════════════════════════════════════════════════════════════════
    【模型规模对比】
    ═══════════════════════════════════════════════════════════════════════════
    
    ┌─────────────────────────────────────────────────────────────────────────┐
    │              │ CogVideoX-2B          │ CogVideoX-5B (本文件主要示例)    │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ Transformer层数 │ 30层                │ 42层                           │
    │ 注意力头数      │ 30                   │ 48                             │
    │ 每头维度        │ 64                   │ 64                             │
    │ inner_dim      │ 30×64 = 1920        │ 48×64 = 3072                   │
    │ 参数量         │ 约20亿               │ 约50亿                          │
    └─────────────────────────────────────────────────────────────────────────┘
    
    ═══════════════════════════════════════════════════════════════════════════
    【与 PyTorch 版本的对应】
    ═══════════════════════════════════════════════════════════════════════════
    
    这个 Flax 实现与 HuggingFace diffusers 库中的 PyTorch 版本功能完全对应。
    可以通过 from_pretrained() 方法加载预训练的 PyTorch 权重。
    """
    
    config_class = FlaxCogVideoXTransformer3DConfig
    
    def __init__(
        self,
        config: FlaxCogVideoXTransformer3DConfig,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
    ):
        super().__init__()
        self.config = config
        self.dtype = dtype
        
        inner_dim = config.num_attention_heads * config.attention_head_dim
        
        if not config.use_rotary_positional_embeddings and config.use_learned_positional_embeddings:
            raise ValueError(
                "There are no CogVideoX checkpoints available with disabled rotary embeddings and learned positional "
                "embeddings."
            )
        
        # 1. Patch embedding
        self.patch_embed = FlaxCogVideoXPatchEmbed(
            patch_size=config.patch_size,
            patch_size_t=config.patch_size_t,
            in_channels=config.in_channels,
            embed_dim=inner_dim,
            text_embed_dim=config.text_embed_dim,
            bias=config.patch_bias,
            sample_width=config.sample_width,
            sample_height=config.sample_height,
            sample_frames=config.sample_frames,
            temporal_compression_ratio=config.temporal_compression_ratio,
            max_text_seq_length=config.max_text_seq_length,
            spatial_interpolation_scale=config.spatial_interpolation_scale,
            temporal_interpolation_scale=config.temporal_interpolation_scale,
            use_positional_embeddings=not config.use_rotary_positional_embeddings,
            use_learned_positional_embeddings=config.use_learned_positional_embeddings,
            rngs=rngs,
        )
        self.dropout_rate = config.dropout
        
        # 2. Time embeddings
        self.time_proj = FlaxTimesteps(inner_dim, config.flip_sin_to_cos, config.freq_shift)
        self.time_embedding = FlaxTimestepEmbedding(
            inner_dim, config.time_embed_dim, config.timestep_activation_fn, rngs=rngs
        )
        
        # Optional ofs embedding (for CogVideoX1.5-5B I2V)
        self.ofs_proj = None
        self.ofs_embedding = None
        if config.ofs_embed_dim:
            self.ofs_proj = FlaxTimesteps(config.ofs_embed_dim, config.flip_sin_to_cos, config.freq_shift)
            self.ofs_embedding = FlaxTimestepEmbedding(
                config.ofs_embed_dim, config.ofs_embed_dim, config.timestep_activation_fn, rngs=rngs
            )
        
        # 3. Transformer blocks
        transformer_blocks = []
        for _ in range(config.num_layers):
            block = FlaxCogVideoXBlock(
                dim=inner_dim,
                num_attention_heads=config.num_attention_heads,
                attention_head_dim=config.attention_head_dim,
                time_embed_dim=config.time_embed_dim,
                dropout=config.dropout,
                activation_fn=config.activation_fn,
                attention_bias=config.attention_bias,
                norm_elementwise_affine=config.norm_elementwise_affine,
                norm_eps=config.norm_eps,
                rngs=rngs,
            )
            transformer_blocks.append(block)
        self.transformer_blocks = nnx.List(transformer_blocks)
        
        self.norm_final = nnx.LayerNorm(
            inner_dim, epsilon=config.norm_eps, use_scale=config.norm_elementwise_affine, rngs=rngs
        )
        
        # 4. Output blocks
        self.norm_out = FlaxAdaLayerNorm(
            embedding_dim=config.time_embed_dim,
            output_dim=2 * inner_dim,
            elementwise_affine=config.norm_elementwise_affine,
            eps=config.norm_eps,
            rngs=rngs,
        )
        
        if config.patch_size_t is None:
            output_dim = config.patch_size * config.patch_size * config.out_channels
        else:
            output_dim = config.patch_size * config.patch_size * config.patch_size_t * config.out_channels
        
        self.proj_out = nnx.Linear(inner_dim, output_dim, rngs=rngs)
    
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        encoder_hidden_states: jnp.ndarray,
        timestep: Union[int, float, jnp.ndarray],
        timestep_cond: Optional[jnp.ndarray] = None,
        ofs: Optional[Union[int, float, jnp.ndarray]] = None,
        image_rotary_emb: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
        deterministic: bool = True,
        return_dict: bool = True,
    ) -> Union[Tuple[jnp.ndarray], jnp.ndarray]:
        """
        Forward pass through the transformer.
        
        Args:
            hidden_states: Video latents (B, T, H, W, C) in channel-last format
            encoder_hidden_states: Text embeddings (B, text_seq_len, text_embed_dim)
            timestep: Diffusion timestep
            timestep_cond: Additional timestep conditioning (not used in CogVideoX)
            ofs: Optical flow scale (for CogVideoX1.5-5B I2V)
            image_rotary_emb: Rotary position embeddings
            deterministic: Whether to use dropout
            return_dict: Whether to return a dict or tuple
            
        Returns:
            Denoised latents (B, T, H, W, C)
        """
        batch_size, num_frames, height, width, channels = hidden_states.shape
        
        # 1. Time embedding
        if isinstance(timestep, (int, float)):
            timestep = jnp.array([timestep] * batch_size, dtype=jnp.float32)
        elif isinstance(timestep, jnp.ndarray) and timestep.ndim == 0:
            timestep = jnp.broadcast_to(timestep, (batch_size,))
        
        t_emb = self.time_proj(timestep)
        t_emb = t_emb.astype(hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)
        
        if self.ofs_embedding is not None and ofs is not None:
            if isinstance(ofs, (int, float)):
                ofs = jnp.array([ofs] * batch_size, dtype=jnp.float32)
            ofs_emb = self.ofs_proj(ofs)
            ofs_emb = ofs_emb.astype(hidden_states.dtype)
            ofs_emb = self.ofs_embedding(ofs_emb)
            emb = emb + ofs_emb
        
        # 2. Patch embedding
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        
        if self.dropout_rate > 0 and not deterministic:
            hidden_states = nnx.Dropout(rate=self.dropout_rate)(hidden_states)
        
        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]
        
        # 3. Transformer blocks
        for block in self.transformer_blocks:
            hidden_states, encoder_hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=emb,
                image_rotary_emb=image_rotary_emb,
                deterministic=deterministic,
            )
        
        hidden_states = self.norm_final(hidden_states)
        
        # 4. Final block
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)
        
        # 5. Unpatchify
        p = self.config.patch_size
        p_t = self.config.patch_size_t
        
        if p_t is None:
            # CogVideoX 1.0: 2D spatial patching
            # hidden_states: (B, num_patches, patch_size^2 * out_channels)
            # Reshape to (B, T, H//p, W//p, out_channels, p, p)
            num_patches = hidden_states.shape[1]
            num_patches_per_frame = (height // p) * (width // p)
            
            output = hidden_states.reshape(
                batch_size, num_frames, height // p, width // p, -1, p, p
            )
            # Permute: (B, T, H//p, W//p, C, p, p) -> (B, T, C, H//p, p, W//p, p)
            output = output.transpose(0, 1, 4, 2, 5, 3, 6)
            # Flatten: -> (B, T, C, H, W)
            output = output.reshape(batch_size, num_frames, -1, height, width)
            # Transpose to channel-last: (B, T, H, W, C)
            output = output.transpose(0, 1, 3, 4, 2)
        else:
            # CogVideoX 1.5: 3D spatio-temporal patching
            output = hidden_states.reshape(
                batch_size, (num_frames + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p
            )
            output = output.transpose(0, 1, 5, 4, 2, 6, 3, 7)
            output = output.reshape(
                batch_size, -1, self.config.out_channels, height, width
            )
            # Transpose to channel-last
            output = output.transpose(0, 1, 3, 4, 2)
        
        if not return_dict:
            return (output,)
        return output
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        subfolder: Optional[str] = "transformer",
        dtype: jnp.dtype = jnp.float32,
        **kwargs,
    ):
        """
        Load pre-trained CogVideoX Transformer weights from HuggingFace.
        
        This method downloads PyTorch weights and converts them to JAX/Flax format.
        For large models (5B+), weights are kept on CPU during conversion to avoid OOM.
        
        Args:
            pretrained_model_name_or_path: Model ID (e.g., "THUDM/CogVideoX-2b")
            subfolder: Subfolder containing transformer weights
            dtype: Target dtype for weights
            **kwargs: Additional config overrides
            
        Returns:
            FlaxCogVideoXTransformer3DModel: Initialized model with loaded weights
        """
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open
        import json
        import re
        import os
        import numpy as np
        
        print(f"[1/5] 加载配置: {pretrained_model_name_or_path}")
        
        # Download config
        config_path = hf_hub_download(
            pretrained_model_name_or_path,
            subfolder=subfolder,
            filename="config.json"
        )
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        config_dict.update(kwargs)
        config = cls.config_class.from_dict(config_dict)
        
        print(f"[2/5] 下载 PyTorch 权重...")
        
        # Check for sharded weights first (index file)
        sharded_index_files = [
            "diffusion_pytorch_model.safetensors.index.json",
            "model.safetensors.index.json",
        ]
        
        # Try to find sharded weights
        index_path = None
        for index_filename in sharded_index_files:
            try:
                index_path = hf_hub_download(
                    pretrained_model_name_or_path,
                    subfolder=subfolder,
                    filename=index_filename
                )
                print(f"  ✓ 发现分片权重索引: {index_filename}")
                break
            except Exception:
                continue
        
        # Load weights as numpy arrays (stays on CPU)
        pytorch_weights = {}
        
        if index_path is not None:
            # Load sharded weights
            with open(index_path, 'r') as f:
                index_data = json.load(f)
            
            weight_map = index_data.get("weight_map", {})
            
            # Get unique shard files
            shard_files = sorted(set(weight_map.values()))
            print(f"  ✓ 需要下载 {len(shard_files)} 个分片文件")
            
            # Download and load each shard
            for i, shard_filename in enumerate(shard_files):
                print(f"  [{i+1}/{len(shard_files)}] 下载: {shard_filename}")
                shard_path = hf_hub_download(
                    pretrained_model_name_or_path,
                    subfolder=subfolder,
                    filename=shard_filename
                )
                
                # Load weights from this shard (as numpy, stays on CPU)
                with safe_open(shard_path, framework="np") as f:
                    for key in f.keys():
                        pytorch_weights[key] = f.get_tensor(key)
        else:
            # Try single file weights
            possible_filenames = [
                "diffusion_pytorch_model.safetensors",
                "diffusion_pytorch_model.bin",
                "pytorch_model.safetensors",
                "pytorch_model.bin",
            ]
            
            ckpt_path = None
            for filename in possible_filenames:
                try:
                    ckpt_path = hf_hub_download(
                        pretrained_model_name_or_path,
                        subfolder=subfolder,
                        filename=filename
                    )
                    print(f"  ✓ 找到权重文件: {filename}")
                    break
                except Exception:
                    continue
            
            if ckpt_path is None:
                raise FileNotFoundError(
                    f"无法在 {pretrained_model_name_or_path}/{subfolder} 中找到权重文件。"
                    f"尝试过的文件名: {possible_filenames}"
                )
            
            # Load PyTorch weights
            if ckpt_path.endswith(".safetensors"):
                with safe_open(ckpt_path, framework="np") as f:
                    for key in f.keys():
                        pytorch_weights[key] = f.get_tensor(key)
            else:
                # .bin file - need torch
                import torch
                pt_state = torch.load(ckpt_path, map_location="cpu")
                for key, value in pt_state.items():
                    pytorch_weights[key] = value.numpy()
        
        print(f"  ✓ 加载了 {len(pytorch_weights)} 个 PyTorch 权重张量 (CPU)")
        
        print(f"[3/5] 转换权重格式 (保持在 CPU)...")
        
        # Convert weight format but keep as numpy (CPU)
        # This avoids allocating TPU memory during conversion
        numpy_weights = {}
        
        for pt_key, pt_tensor in pytorch_weights.items():
            # Remove _orig_mod prefix if present
            if pt_key.startswith("_orig_mod."):
                pt_key = pt_key[len("_orig_mod."):]
            
            jax_key = pt_key
            np_tensor = pt_tensor
            
            # ==== Name mapping from PyTorch to Flax ====
            
            # FeedForward network: ff.net.0.proj -> ff.linear1, ff.net.2 -> ff.linear2
            # The PyTorch FeedForward has a GEGLU which splits into two linear layers
            jax_key = jax_key.replace(".ff.net.0.proj.", ".ff.linear1.")
            jax_key = jax_key.replace(".ff.net.2.", ".ff.linear2.")
            
            # Attention output: to_out is a Sequential with [Linear, Dropout]
            # PyTorch: to_out.0 -> Flax: to_out
            jax_key = jax_key.replace(".to_out.0.", ".to_out.")
            
            # Attention QK norm: norm_k -> k_norm, norm_q -> q_norm
            jax_key = jax_key.replace(".norm_k.", ".k_norm.")
            jax_key = jax_key.replace(".norm_q.", ".q_norm.")
            
            # ==== Weight format conversion ====
            
            # Convert Linear layers
            if ".weight" in jax_key and ("linear" in jax_key or "proj" in jax_key or "to_" in jax_key):
                jax_key = jax_key.replace(".weight", ".kernel")
                # Transpose: (out, in) -> (in, out)
                if len(np_tensor.shape) == 2:
                    np_tensor = np_tensor.T
            
            # Convert Conv layers for patch_embed.proj
            if "patch_embed.proj" in jax_key and ".weight" in jax_key:
                jax_key = jax_key.replace(".weight", ".kernel")
                # Handle both 2D and 3D convolutions
                if len(np_tensor.shape) == 5:
                    # Conv3d: (out, in, t, h, w) -> (t, h, w, in, out)
                    np_tensor = np_tensor.transpose(2, 3, 4, 1, 0)
                elif len(np_tensor.shape) == 4:
                    # Conv2d: (out, in, h, w) -> (h, w, in, out)
                    np_tensor = np_tensor.transpose(2, 3, 1, 0)
            
            # LayerNorm
            if ".weight" in jax_key and "norm" in jax_key:
                jax_key = jax_key.replace(".weight", ".scale")
            
            # Add .value for Param types
            if "pos_embedding" in jax_key:
                jax_key = jax_key + ".value"
            
            # Skip weights that don't exist in Flax model
            # patch_embed.proj uses Conv without bias in our Flax implementation
            if jax_key == "patch_embed.proj.bias":
                continue
            
            # Keep as numpy array for now (CPU)
            # Note: numpy doesn't have bfloat16, so we keep as float32 and convert when moving to JAX
            numpy_weights[jax_key] = np_tensor.astype(np.float32)
        
        # Clear original pytorch_weights to free CPU memory
        del pytorch_weights
        
        print(f"  ✓ 转换了 {len(numpy_weights)} 个权重格式")
        
        print(f"[4/5] 在 CPU 上初始化模型结构...")
        
        # Initialize model on CPU to avoid TPU memory allocation for random weights
        # We use jax.default_device to force CPU initialization
        cpu_device = jax.devices('cpu')[0]
        
        with jax.default_device(cpu_device):
            key = jax.random.key(0)
            rngs = nnx.Rngs(key)
            model = cls(config=config, rngs=rngs, dtype=dtype)
        
        print(f"  ✓ 模型结构初始化完成 (CPU)")
        
        print(f"[5/5] 加载权重并移至目标设备...")
        
        # Convert numpy weights to JAX arrays and unflatten
        from flax.traverse_util import unflatten_dict
        
        # Convert to jax arrays (this will be placed on default device)
        jax_weights = {k: jnp.array(v, dtype=dtype) for k, v in numpy_weights.items()}
        
        # Clear numpy weights
        del numpy_weights
        
        # Unflatten and merge
        nested_weights = unflatten_dict(jax_weights, sep=".")
        
        graphdef, _ = nnx.split(model)
        model = nnx.merge(graphdef, nested_weights)
        
        print(f"✓ 模型加载完成!")
        
        return model