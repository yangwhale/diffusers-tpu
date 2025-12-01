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
CogVideoX Transformer 3D 模型的 PyTorch 实现 - 详细注释版
======================================================================================

【文件概述】

这是 CogVideoX 视频生成模型的 Transformer 核心实现（PyTorch 版本）。
CogVideoX 是一个基于 Diffusion 的视频生成模型，使用 DiT（Diffusion Transformer）
架构来处理视频数据。

【与 JAX/Flax 版本的对应关系】

本文件是 PyTorch 官方实现，对应的 JAX/Flax 实现在：
    cogvideox_transformer_3d_flax_explained.py

两个版本的功能完全一致，主要区别在于：

┌─────────────────────────────────────────────────────────────────────────────────┐
│ 特性              │ PyTorch 版本              │ JAX/Flax 版本                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│ 框架              │ torch.nn                  │ flax.nnx                        │
│ 注意力实现        │ F.scaled_dot_product_attention │ Splash Attention (TPU优化) │
│ 梯度检查点        │ torch.utils.checkpoint    │ JAX 原生支持                    │
│ LoRA 支持         │ PeftAdapterMixin          │ 需额外实现                      │
│ 张量格式          │ (B, C, T, H, W) 通道优先  │ (B, T, H, W, C) 通道最后        │
│ 设备              │ GPU (CUDA)                │ TPU (XLA)                       │
└─────────────────────────────────────────────────────────────────────────────────┘

【为什么需要两个版本？】

1. PyTorch 版本：
   - 官方参考实现，用于模型发布和权重共享
   - 支持 NVIDIA GPU 推理和训练
   - 更好的调试体验和工具链支持
   - 支持 LoRA、量化等高级功能

2. JAX/Flax 版本：
   - 针对 Google TPU 优化
   - 使用 Splash Attention 减少内存占用
   - 支持大规模分布式训练
   - 更好的 XLA 编译优化

【整体架构流程】（与 Flax 版本相同）

    输入视频latents (B, C, T, H, W)   ← PyTorch 使用通道优先格式
            ↓
    ┌───────────────────────────────────────────────────────────┐
    │  CogVideoXPatchEmbed                                      │
    │  - 将视频切成patches，转换成token序列                       │
    │  - 投影文本嵌入到相同维度                                   │
    │  - 拼接文本tokens和视频tokens                              │
    └───────────────────────────────────────────────────────────┘
            ↓
    ┌───────────────────────────────────────────────────────────┐
    │  CogVideoXBlock × N (2B: 30层, 5B: 42层)                  │
    │  ┌─────────────────────────────────────────────────────┐ │
    │  │ CogVideoXLayerNormZero                              │ │
    │  │ - LayerNorm + 时间步调制 (scale/shift/gate)          │ │
    │  └─────────────────────────────────────────────────────┘ │
    │                        ↓                                  │
    │  ┌─────────────────────────────────────────────────────┐ │
    │  │ Attention (联合注意力)                               │ │
    │  │ - 使用 PyTorch F.scaled_dot_product_attention       │ │
    │  │ - 支持 Flash Attention 2 (GPU优化)                  │ │
    │  └─────────────────────────────────────────────────────┘ │
    │                        ↓                                  │
    │  ┌─────────────────────────────────────────────────────┐ │
    │  │ FeedForward                                          │ │
    │  │ - Linear → GELU → Linear                             │ │
    │  └─────────────────────────────────────────────────────┘ │
    └───────────────────────────────────────────────────────────┘
            ↓
    ┌───────────────────────────────────────────────────────────┐
    │  Final LayerNorm + AdaLayerNorm                           │
    │  Linear投影回patch维度                                     │
    │  Unpatchify: tokens → (B, C, T, H, W)                     │
    └───────────────────────────────────────────────────────────┘
            ↓
    输出: 去噪后的视频latents (B, C, T, H, W)

【模型规模】

┌─────────────────────────────────────────────────────────────────────────────────┐
│ 参数            │ CogVideoX-2B        │ CogVideoX-5B                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│ Transformer层数 │ 30                  │ 42                                      │
│ 注意力头数      │ 30                  │ 48                                      │
│ 每头维度        │ 64                  │ 64                                      │
│ inner_dim      │ 1920 (30×64)        │ 3072 (48×64)                            │
│ 总参数量        │ ~2B                 │ ~5B                                     │
└─────────────────────────────────────────────────────────────────────────────────┘
"""

from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn

# ════════════════════════════════════════════════════════════════════════════════
# Python 相对导入语法说明
# ════════════════════════════════════════════════════════════════════════════════
#
# 【什么是相对导入？】
#
# Python 的相对导入使用 `.` 符号来表示包的层级关系：
#   - `.`   = 当前目录（当前包）
#   - `..`  = 上一级目录（父包）
#   - `...` = 上两级目录（祖父包）
#
# 【本文件的位置】
#
# 假设 diffusers 包的目录结构如下：
#
# diffusers/                              ← 根包
# ├── __init__.py
# ├── configuration_utils.py              ← ... 导入这里
# ├── loaders/                            ← ... 导入这里
# │   └── peft.py
# ├── utils/                              ← ... 导入这里
# │   ├── __init__.py
# │   └── torch_utils.py
# └── models/                             ← .. 的父目录
#     ├── __init__.py
#     ├── attention.py                    ← .. 导入这里
#     ├── attention_processor.py          ← .. 导入这里
#     ├── cache_utils.py                  ← .. 导入这里
#     ├── embeddings.py                   ← .. 导入这里
#     ├── modeling_outputs.py             ← .. 导入这里
#     ├── modeling_utils.py               ← .. 导入这里
#     ├── normalization.py                ← .. 导入这里
#     └── transformers/                   ← 当前目录
#         ├── __init__.py
#         └── cogvideox_transformer_3d.py ← 本文件所在位置
#
# 【导入层级说明】
#
# 从本文件的角度：
#   - `from ..attention import ...`    → 从 models/attention.py 导入
#   - `from ...utils import ...`       → 从 diffusers/utils/ 导入
#   - `from ...configuration_utils`    → 从 diffusers/configuration_utils.py 导入
#
# ════════════════════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────────────────
# 从 diffusers 根包导入 (... = 上两级)
# ──────────────────────────────────────────────────────────────────────────────

# ConfigMixin: HuggingFace 配置管理基类
#   - 提供 from_config() 和 save_config() 方法
#   - 让模型可以从 JSON 配置文件初始化
#
# register_to_config: 装饰器，自动将 __init__ 参数保存到 config
#   - 用法: @register_to_config
#   - 效果: 调用 model.config.xxx 可以访问初始化参数
#
from ...configuration_utils import ConfigMixin, register_to_config

# PeftAdapterMixin: LoRA/PEFT 适配器支持
#   - 提供 load_lora_weights(), set_lora_device() 等方法
#   - 让模型支持低秩适配（Low-Rank Adaptation）微调
#   - 【Flax 版本没有这个】Flax 需要单独实现 LoRA 支持
#
from ...loaders import PeftAdapterMixin

# USE_PEFT_BACKEND: 布尔值，是否启用 PEFT 后端
# logging: HuggingFace 统一日志模块
# scale_lora_layers / unscale_lora_layers: LoRA 权重缩放函数
#   - 在前向传播前后调整 LoRA 层的权重
#
from ...utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers

# maybe_allow_in_graph: PyTorch 编译装饰器
#   - 用于 torch.compile() 兼容性
#   - 标记类可以被包含在计算图中
#   - 避免某些动态行为导致编译失败
#
from ...utils.torch_utils import maybe_allow_in_graph

# ──────────────────────────────────────────────────────────────────────────────
# 从 models 包导入 (.. = 上一级)
# ──────────────────────────────────────────────────────────────────────────────

# Attention: 多头注意力基类
#   - 实现标准的 Multi-Head Attention
#   - 支持自定义 processor（如 Flash Attention）
#
# FeedForward: 前馈网络（FFN）
#   - 实现 Linear → Activation → Linear 结构
#   - 通常升维 4 倍再降回来
#
from ..attention import Attention, FeedForward

# AttentionProcessor: 注意力计算的抽象基类
#   - 定义 __call__ 接口
#
# CogVideoXAttnProcessor2_0: CogVideoX 专用注意力处理器
#   - 使用 F.scaled_dot_product_attention（支持 Flash Attention 2）
#   - 实现联合注意力（文本+视频一起）
#
# FusedCogVideoXAttnProcessor2_0: 融合版本
#   - QKV 投影融合成一个矩阵，减少内存访问
#
from ..attention_processor import AttentionProcessor, CogVideoXAttnProcessor2_0, FusedCogVideoXAttnProcessor2_0

# CacheMixin: KV Cache 支持
#   - 用于推理优化
#   - 缓存之前的 Key 和 Value，避免重复计算
#
from ..cache_utils import CacheMixin

# CogVideoXPatchEmbed: Patch 嵌入模块
#   - 将视频分割成 patches，转换成 token 序列
#   - 同时处理文本嵌入的投影
#
# TimestepEmbedding: 时间步嵌入 MLP
#   - 将正弦编码通过 MLP 投影到目标维度
#   - 结构: Linear → SiLU → Linear
#
# Timesteps: 时间步正弦编码
#   - 将标量时间步转换成高维向量
#   - 使用正弦/余弦编码（类似 Transformer 位置编码）
#
from ..embeddings import CogVideoXPatchEmbed, TimestepEmbedding, Timesteps

# Transformer2DModelOutput: 模型输出数据类
#   - 包装输出张量，提供统一的接口
#   - 主要字段: sample（输出张量）
#
from ..modeling_outputs import Transformer2DModelOutput

# ModelMixin: HuggingFace 模型基类
#   - 提供 save_pretrained(), from_pretrained() 方法
#   - 统一的权重加载/保存接口
#
from ..modeling_utils import ModelMixin

# AdaLayerNorm: 自适应 LayerNorm
#   - 用时间嵌入调制 LayerNorm 的 scale 和 shift
#   - 用于模型输出前的最终调制
#
# CogVideoXLayerNormZero: CogVideoX 专用条件调制 LayerNorm
#   - 输出 6 组参数（shift, scale, gate × 2）
#   - 用于 Transformer 块内部
#
from ..normalization import AdaLayerNorm, CogVideoXLayerNormZero


logger = logging.get_logger(__name__)  # 获取模块级日志器


# ════════════════════════════════════════════════════════════════════════════════
# Python 装饰器语法说明
# ════════════════════════════════════════════════════════════════════════════════
#
# 【什么是装饰器？】
#
# 装饰器是 Python 的一种语法糖，用于在不修改原函数/类代码的情况下，
# 给函数或类添加额外功能。
#
# 语法:
#     @decorator_name
#     class MyClass:
#         ...
#
# 等价于:
#     class MyClass:
#         ...
#     MyClass = decorator_name(MyClass)
#
# 【@maybe_allow_in_graph 的作用】
#
# 这是 HuggingFace diffusers 定义的装饰器，用于 PyTorch 2.0+ 的编译功能。
#
# 背景：PyTorch 2.0 引入了 torch.compile()，可以将 Python 代码编译成
# 高效的内核，显著提升性能（通常 1.5-2x 加速）。
#
# 问题：torch.compile() 需要分析代码的计算图（graph），但某些 Python
# 动态特性会导致编译失败，比如：
# - 数据相关的控制流 (if x.sum() > 0: ...)
# - 动态形状变化
# - 某些复杂的类结构
#
# @maybe_allow_in_graph 的作用：
# 1. 标记这个类可以被安全地包含在计算图中
# 2. 避免 torch.compile() 在这个类处"打断"计算图
# 3. 如果编译失败，会优雅地回退到普通执行
#
# 【其他常见装饰器】
#
# @register_to_config (在 __init__ 上)
# ────────────────────────────────────────────────────────────────────
# 作用: 自动将 __init__ 的参数保存到 self.config
#
# 例如:
#     @register_to_config
#     def __init__(self, hidden_size=768, num_layers=12):
#         ...
#
# 效果: model.config.hidden_size 和 model.config.num_layers 可访问
# 这使得模型可以轻松保存/加载配置（save_pretrained/from_pretrained）
#
# @property (Python 内置)
# ────────────────────────────────────────────────────────────────────
# 作用: 让方法像属性一样访问
#
# 例如:
#     @property
#     def attn_processors(self):
#         return self._get_processors()
#
# 使用: model.attn_processors（不需要括号）
#
# @staticmethod / @classmethod (Python 内置)
# ────────────────────────────────────────────────────────────────────
# @staticmethod: 静态方法，不接收 self 或 cls
# @classmethod: 类方法，第一个参数是 cls（类本身）
#
# 【Flax 版本的对比】
#
# PyTorch 使用装饰器来扩展类功能
# Flax/JAX 更多使用函数式风格，通过组合实现类似功能
#
# ════════════════════════════════════════════════════════════════════════════════

@maybe_allow_in_graph
class CogVideoXBlock(nn.Module):
    """
    CogVideoX 的 Transformer 块 - 模型的核心重复单元。
    
    【与 Flax 版本的对应】
    对应 Flax 版本中的 FlaxCogVideoXBlock 类。
    
    【块的结构图】（与 Flax 版本相同）
    
        hidden_states (视频)          encoder_hidden_states (文本)
              ↓                                    ↓
        ╔════════════════════════════════════════════════════════════╗
        ║                   norm1 (CogVideoXLayerNormZero)           ║
        ║    用时间嵌入生成 scale, shift, gate 调制参数               ║
        ║                                                            ║
        ║    【关键差异】PyTorch 版本:                                ║
        ║    norm1 直接输出4个值: norm_hidden, norm_encoder,         ║
        ║                        gate_msa, enc_gate_msa              ║
        ╚════════════════════════════════════════════════════════════╝
              ↓ (调制后)                           ↓ (调制后)
              └──────────────────┬─────────────────┘
                                 ↓
        ╔════════════════════════════════════════════════════════════╗
        ║                   attn1 (联合注意力)                        ║
        ║    使用 CogVideoXAttnProcessor2_0                          ║
        ║    底层调用 F.scaled_dot_product_attention                 ║
        ║                                                            ║
        ║    【与 Flax 版本的区别】                                   ║
        ║    PyTorch: F.scaled_dot_product_attention (Flash Attn 2)  ║
        ║    Flax:    Splash Attention (TPU 优化)                    ║
        ╚════════════════════════════════════════════════════════════╝
                                 ↓
              ┌──────────────────┴─────────────────┐
              ↓ (分割)                             ↓ (分割)
        hidden += gate * attn_out     encoder += gate * attn_out
              ↓                                    ↓
        ╔════════════════════════════════════════════════════════════╗
        ║                   norm2 (CogVideoXLayerNormZero)           ║
        ╚════════════════════════════════════════════════════════════╝
              ↓                                    ↓
              └──────────────────┬─────────────────┘
                       拼接后送入 FFN
                                 ↓
        ╔════════════════════════════════════════════════════════════╗
        ║                   ff (FeedForward)                          ║
        ║    Linear(dim → 4*dim) → GELU → Linear(4*dim → dim)        ║
        ╚════════════════════════════════════════════════════════════╝
                                 ↓
              ┌──────────────────┴─────────────────┐
              ↓ (分割)                             ↓ (分割)
        hidden += gate * ff_out       encoder += gate * ff_out
              ↓                                    ↓
        输出 hidden_states                 输出 encoder_hidden_states
    
    【门控残差连接的意义】
    
    普通残差: output = input + sublayer(input)
    门控残差: output = input + gate * sublayer(input)
    
    gate 由时间步嵌入控制，让模型可以动态调整每个块的"贡献程度"：
    - 去噪早期（噪声大）: gate 较大，大刀阔斧地去噪
    - 去噪后期（接近原图）: gate 较小，精细调整
    
    Parameters:
        dim (`int`):
            输入/输出的通道数，等于 inner_dim = num_attention_heads × attention_head_dim
            CogVideoX-5B: 48 × 64 = 3072
        num_attention_heads (`int`):
            多头注意力的头数 (2B=30, 5B=48)
        attention_head_dim (`int`):
            每个注意力头的维度 (64)
        time_embed_dim (`int`):
            时间嵌入的维度，用于调制 LayerNorm (512)
        dropout (`float`, defaults to `0.0`):
            Dropout 概率
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            FFN 中使用的激活函数
        attention_bias (`bool`, defaults to `False`):
            注意力投影层是否使用 bias
        qk_norm (`bool`, defaults to `True`):
            是否对 Query 和 Key 做 LayerNorm（提高训练稳定性）
        norm_elementwise_affine (`bool`, defaults to `True`):
            LayerNorm 是否有可学习的 scale 和 bias
        norm_eps (`float`, defaults to `1e-5`):
            LayerNorm 的 epsilon，防止除零
        final_dropout (`bool` defaults to `False`):
            FFN 最后是否加 Dropout
        ff_inner_dim (`int`, *optional*, defaults to `None`):
            FFN 中间层维度，默认为 4 × dim
        ff_bias (`bool`, defaults to `True`):
            FFN 的 Linear 层是否使用 bias
        attention_out_bias (`bool`, defaults to `True`):
            注意力输出投影是否使用 bias
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
    ):
        super().__init__()

        # ══════════════════════════════════════════════════════════════════════
        # 1. Self Attention 分支
        # ══════════════════════════════════════════════════════════════════════
        #
        # norm1: CogVideoXLayerNormZero
        # ────────────────────────────────────────────────────────────────────
        # 这是一个条件调制的 LayerNorm，接收时间嵌入 temb 来生成：
        # - shift, scale: 用于调制 LayerNorm 输出
        # - gate: 用于控制残差连接的强度
        #
        # 输入:  hidden_states (视频), encoder_hidden_states (文本), temb (时间)
        # 输出:  norm_hidden, norm_encoder, gate_msa, enc_gate_msa (4个张量)
        #
        # 【与 Flax 版本的对应】
        # Flax 版本: FlaxCogVideoXLayerNormZero
        # 功能完全相同，只是框架不同
        #
        self.norm1 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        # attn1: Attention (联合注意力)
        # ────────────────────────────────────────────────────────────────────
        # 使用 CogVideoXAttnProcessor2_0 处理器
        # 底层调用 F.scaled_dot_product_attention (支持 Flash Attention 2)
        #
        # 【关键特性】
        # - 联合注意力：文本和视频 tokens 拼接后一起做自注意力
        # - QK Norm：对 Query 和 Key 做 LayerNorm，提高训练稳定性
        # - RoPE 支持：可选的旋转位置编码
        #
        # 【与 Flax 版本的区别】
        # PyTorch: F.scaled_dot_product_attention → 使用 Flash Attention 2 (GPU)
        # Flax:    Splash Attention → TPU 优化的分块注意力
        #
        self.attn1 = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
            processor=CogVideoXAttnProcessor2_0(),
        )

        # ══════════════════════════════════════════════════════════════════════
        # 2. Feed Forward 分支
        # ══════════════════════════════════════════════════════════════════════
        #
        # norm2: 第二个条件调制 LayerNorm
        # 与 norm1 结构相同，但生成不同的调制参数 (gate_ff, enc_gate_ff)
        #
        self.norm2 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        # ff: FeedForward (前馈网络)
        # ────────────────────────────────────────────────────────────────────
        # 结构: Linear(dim → 4*dim) → GELU → Dropout → Linear(4*dim → dim)
        #
        # 以 5B 模型为例:
        #   Linear(3072 → 12288) → GELU → Linear(12288 → 3072)
        #
        # 【与 Flax 版本的对应】
        # Flax 版本: FlaxFeedForward
        # 结构完全相同
        #
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        CogVideoXBlock 前向传播。
        
        【维度说明】(以 5B 模型为例)
        
            hidden_states:          (B, 66150, 3072)  ← 视频 tokens
            encoder_hidden_states:  (B, 226, 3072)    ← 文本 tokens
            temb:                   (B, 512)          ← 时间嵌入
        
        【计算流程】
        
        Step 1: 第一次 LayerNorm + 调制
            (hidden, encoder, temb) → norm1 → (norm_hidden, norm_encoder, gate_msa, enc_gate_msa)
        
        Step 2: 联合注意力
            (norm_hidden, norm_encoder) → attn1 → (attn_hidden, attn_encoder)
            
            内部会将 hidden 和 encoder 拼接成 (B, 66376, 3072) 做自注意力
            然后再分割回两个张量
        
        Step 3: 门控残差连接 (注意力)
            hidden = hidden + gate_msa * attn_hidden
            encoder = encoder + enc_gate_msa * attn_encoder
        
        Step 4: 第二次 LayerNorm + 调制
            (hidden, encoder, temb) → norm2 → (norm_hidden, norm_encoder, gate_ff, enc_gate_ff)
        
        Step 5: 前馈网络
            concat(norm_encoder, norm_hidden) → ff → ff_output
            
            【注意】这里先拼接再 FFN，然后再分割
            这样文本和视频共享同一个 FFN 权重
        
        Step 6: 门控残差连接 (FFN)
            hidden = hidden + gate_ff * ff_output[视频部分]
            encoder = encoder + enc_gate_ff * ff_output[文本部分]
        
        Args:
            hidden_states: 视频特征 (B, num_patches, dim)
            encoder_hidden_states: 文本特征 (B, text_seq_len, dim)
            temb: 时间嵌入 (B, time_embed_dim)
            image_rotary_emb: 可选的 RoPE 位置编码
            attention_kwargs: 注意力层的额外参数（如 LoRA scale）
            
        Returns:
            Tuple of (hidden_states, encoder_hidden_states)
        """
        text_seq_length = encoder_hidden_states.size(1)  # 通常是 226
        attention_kwargs = attention_kwargs or {}

        # ──────────────────────────────────────────────────────────────────────
        # Step 1: 第一次 LayerNorm + 时间步调制
        # ──────────────────────────────────────────────────────────────────────
        # norm1 返回4个值:
        # - norm_hidden_states: 调制后的视频特征
        # - norm_encoder_hidden_states: 调制后的文本特征
        # - gate_msa: 视频的注意力门控
        # - enc_gate_msa: 文本的注意力门控
        #
        norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
            hidden_states, encoder_hidden_states, temb
        )

        # ──────────────────────────────────────────────────────────────────────
        # Step 2: 联合注意力
        # ──────────────────────────────────────────────────────────────────────
        # CogVideoXAttnProcessor2_0 内部会:
        # 1. 拼接 norm_hidden 和 norm_encoder
        # 2. 计算 Q, K, V
        # 3. 应用 RoPE (如果提供)
        # 4. 调用 F.scaled_dot_product_attention
        # 5. 分割回文本和视频部分
        #
        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **attention_kwargs,
        )

        # ──────────────────────────────────────────────────────────────────────
        # Step 3: 门控残差连接 (注意力输出)
        # ──────────────────────────────────────────────────────────────────────
        # gate 控制新信息的注入程度
        # gate 接近 0: 几乎不更新（保守）
        # gate 接近 1: 正常更新
        # gate 大于 1: 增强更新（激进）
        #
        hidden_states = hidden_states + gate_msa * attn_hidden_states
        encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states

        # ──────────────────────────────────────────────────────────────────────
        # Step 4: 第二次 LayerNorm + 时间步调制
        # ──────────────────────────────────────────────────────────────────────
        norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
            hidden_states, encoder_hidden_states, temb
        )

        # ──────────────────────────────────────────────────────────────────────
        # Step 5: 前馈网络
        # ──────────────────────────────────────────────────────────────────────
        # 【注意】这里的设计：先拼接，再 FFN，再分割
        #
        # 为什么这样做？
        # - 文本和视频共享同一个 FFN 权重
        # - 减少参数量（不需要两个独立的 FFN）
        # - 保持一致的特征变换
        #
        # 【Flax 版本的"先拼接再分开"问题】
        # 你之前问的 patch_embed 的问题也类似
        # 这里的拼接是有意义的（共享 FFN），不是多余的
        #
        norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
        ff_output = self.ff(norm_hidden_states)

        # ──────────────────────────────────────────────────────────────────────
        # Step 6: 门控残差连接 (FFN 输出)
        # ──────────────────────────────────────────────────────────────────────
        # 分割 ff_output：
        # - 前 text_seq_length 个 tokens 是文本部分
        # - 后面的是视频部分
        #
        hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:]
        encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]

        return hidden_states, encoder_hidden_states


# ════════════════════════════════════════════════════════════════════════════════
# Mixin 类详解 - Python 多重继承的设计模式
# ════════════════════════════════════════════════════════════════════════════════
#
# 【什么是 Mixin？】
#
# Mixin 是一种 Python 设计模式，用于为类添加「可复用的功能模块」。
#
# 与普通继承的区别：
# ┌─────────────────────────────────────────────────────────────────────────────────┐
# │ 普通继承                              │ Mixin                                   │
# ├─────────────────────────────────────────────────────────────────────────────────┤
# │ class Dog(Animal):                   │ class Dog(Animal, Runnable, Trainable): │
# │   - Dog「是」一种 Animal              │   - Dog「是」Animal                     │
# │   - 单一继承，一条继承链              │   - Dog「能」Run（Mixin 功能）           │
# │                                       │   - Dog「能」被 Train（Mixin 功能）      │
# │                                       │   - 多重继承，组合多个功能               │
# └─────────────────────────────────────────────────────────────────────────────────┘
#
# 【Mixin 的特点】
#
# 1. 功能单一：每个 Mixin 只提供一种特定功能
# 2. 不独立使用：Mixin 不是设计来单独实例化的
# 3. 无状态或少状态：主要提供方法，避免复杂的 __init__
# 4. 可组合：多个 Mixin 可以自由组合
# 5. 命名约定：类名以 Mixin 结尾（如 ModelMixin, ConfigMixin）
#
# 【为什么用 Mixin 而不是单继承？】
#
# 假设我们有 100 个不同的模型类：
#
# 方案 1：单继承（❌ 不好）
# ────────────────────────────────────────────────────────────────────
#     class BaseModel:
#         def save_pretrained(self): ...
#         def load_pretrained(self): ...
#         def get_config(self): ...
#         def save_config(self): ...
#         def load_lora(self): ...
#         def enable_cache(self): ...
#         # ... 几十个方法
#
#     # 问题：所有模型都必须有所有功能，无法按需选择
#
# 方案 2：Mixin（✓ 好）
# ────────────────────────────────────────────────────────────────────
#     class ModelMixin:      # 只负责保存/加载模型
#         def save_pretrained(self): ...
#         def load_pretrained(self): ...
#
#     class ConfigMixin:     # 只负责配置管理
#         def get_config(self): ...
#         def save_config(self): ...
#
#     class PeftAdapterMixin:  # 只负责 LoRA
#         def load_lora(self): ...
#
#     class CacheMixin:      # 只负责缓存
#         def enable_cache(self): ...
#
#     # 按需组合！
#     class Model1(ModelMixin, ConfigMixin):  # 只要保存和配置
#         pass
#
#     class Model2(ModelMixin, ConfigMixin, PeftAdapterMixin):  # 还要 LoRA
#         pass
#
# ════════════════════════════════════════════════════════════════════════════════

class CogVideoXTransformer3DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, CacheMixin):
    """
    CogVideoX 3D Transformer 主模型 - 视频生成的核心。
    
    【与 Flax 版本的对应】
    对应 Flax 版本中的 FlaxCogVideoXTransformer3DModel 类。
    
    ═══════════════════════════════════════════════════════════════════════════
    【继承的 Mixin 类详解】
    ═══════════════════════════════════════════════════════════════════════════
    
    本类继承了 4 个 Mixin，下面逐一解释：
    
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │ Mixin 类                │ 功能                                                  │
    ├─────────────────────────────────────────────────────────────────────────────────┤
    │ ModelMixin             │ HuggingFace 模型基类，提供 save/load 功能              │
    │ ConfigMixin            │ 配置管理，支持 from_pretrained 和 save_pretrained     │
    │ PeftAdapterMixin       │ LoRA/PEFT 适配器支持 (Flax 版本没有)                   │
    │ CacheMixin             │ KV Cache 支持，用于推理优化                            │
    └─────────────────────────────────────────────────────────────────────────────────┘
    
    ───────────────────────────────────────────────────────────────────────────
    【1. ModelMixin】 - 模型保存/加载功能
    ───────────────────────────────────────────────────────────────────────────
    
    提供的主要方法：
    
        model.save_pretrained("./my_model/")
        ─────────────────────────────────────────────────────────────────
        将模型权重保存到指定目录
        会生成：
        - diffusion_pytorch_model.safetensors (权重文件)
        - 或 diffusion_pytorch_model.bin (旧格式)
        
        model = Model.from_pretrained("THUDM/CogVideoX-5b")
        ─────────────────────────────────────────────────────────────────
        从 HuggingFace Hub 或本地目录加载模型
        自动处理：
        - 权重下载
        - 分片权重合并
        - 设备放置
        - dtype 转换
        
        model.to(device="cuda", dtype=torch.float16)
        ─────────────────────────────────────────────────────────────────
        移动模型到指定设备和数据类型
        
        model.num_parameters()
        ─────────────────────────────────────────────────────────────────
        返回模型参数量
    
    【为什么需要 ModelMixin？】
    让所有 diffusers 模型有统一的保存/加载接口，
    用户不需要学习每个模型不同的 API。
    
    ───────────────────────────────────────────────────────────────────────────
    【2. ConfigMixin】 - 配置管理功能
    ───────────────────────────────────────────────────────────────────────────
    
    提供的主要功能：
    
        @register_to_config 装饰器
        ─────────────────────────────────────────────────────────────────
        用在 __init__ 方法上，自动将所有参数保存到 self.config
        
        例如：
            @register_to_config
            def __init__(self, num_layers=30, num_heads=48):
                ...
        
        之后可以通过 model.config.num_layers 访问参数值
        
        config = model.config
        ─────────────────────────────────────────────────────────────────
        获取模型配置，返回一个类似字典的对象
        
        model.save_config("./my_model/")
        ─────────────────────────────────────────────────────────────────
        保存配置到 config.json
        
        config = Model.load_config("./my_model/")
        ─────────────────────────────────────────────────────────────────
        加载配置，不加载权重
    
    【为什么需要 ConfigMixin？】
    - 保证模型可复现：配置和权重一起保存
    - 便于修改超参数：加载配置后可以调整再实例化
    - 统一的配置格式：所有模型都用 JSON 配置
    
    ───────────────────────────────────────────────────────────────────────────
    【3. PeftAdapterMixin】 - LoRA/PEFT 适配器支持
    ───────────────────────────────────────────────────────────────────────────
    
    PEFT = Parameter-Efficient Fine-Tuning（参数高效微调）
    LoRA = Low-Rank Adaptation（低秩适配）
    
    提供的主要方法：
    
        model.load_lora_weights("path/to/lora")
        ─────────────────────────────────────────────────────────────────
        加载预训练的 LoRA 权重
        LoRA 只有几 MB，而原模型有几 GB
        
        model.set_adapters(["lora1", "lora2"], weights=[0.7, 0.3])
        ─────────────────────────────────────────────────────────────────
        同时使用多个 LoRA，并设置混合权重
        
        model.disable_lora()
        model.enable_lora()
        ─────────────────────────────────────────────────────────────────
        临时禁用/启用 LoRA
        
        model.delete_adapters(["lora1"])
        ─────────────────────────────────────────────────────────────────
        删除已加载的 LoRA
    
    【LoRA 的原理简述】
    
    原始权重矩阵 W (如 4096 × 4096 = 16M 参数)
    
    LoRA 思路：
    - 不修改原始 W
    - 添加一个低秩矩阵 ΔW = A × B
    - A: 4096 × 8 = 32K 参数
    - B: 8 × 4096 = 32K 参数
    - 总共只需要 64K 参数（原来的 0.4%）
    
    推理时：W_new = W + α × (A × B)
    
    【为什么 Flax 版本没有 PeftAdapterMixin？】
    - PEFT 库主要为 PyTorch 设计
    - TPU 生态中 LoRA 支持不如 GPU 成熟
    - Flax 需要另外实现 LoRA 支持
    
    ───────────────────────────────────────────────────────────────────────────
    【4. CacheMixin】 - KV Cache 缓存功能
    ───────────────────────────────────────────────────────────────────────────
    
    KV Cache = Key-Value Cache（键值缓存）
    
    主要用于**自回归推理**（一次生成一个 token）：
    
    【没有 Cache 的问题】
    
        第 1 步：处理 "Hello"        → 计算 K, V
        第 2 步：处理 "Hello world"  → 重新计算所有 K, V（包括 Hello 的）
        第 3 步：处理 "Hello world !" → 又重新计算...
        
        浪费！"Hello" 的 K, V 被重复计算
    
    【有 Cache 的优化】
    
        第 1 步：处理 "Hello"        → 计算并缓存 K, V
        第 2 步：处理 "world"        → 只计算新 token，复用缓存
        第 3 步：处理 "!"           → 只计算新 token，复用缓存
        
        效率大增！
    
    提供的主要方法：
    
        model.enable_cache()
        ─────────────────────────────────────────────────────────────────
        启用 KV Cache
        
        model.disable_cache()
        ─────────────────────────────────────────────────────────────────
        禁用 KV Cache（训练时通常禁用）
        
        model.reset_cache()
        ─────────────────────────────────────────────────────────────────
        清空缓存（开始新序列时调用）
    
    【CogVideoX 中的 Cache 使用场景】
    
    虽然 CogVideoX 是 Diffusion 模型（不是自回归），
    但 CacheMixin 仍然有用：
    - 缓存不变的中间结果
    - 支持未来可能的自回归视频生成扩展
    
    ═══════════════════════════════════════════════════════════════════════════
    【Mixin 的继承顺序】
    ═══════════════════════════════════════════════════════════════════════════
    
    class MyModel(ModelMixin, ConfigMixin, PeftAdapterMixin, CacheMixin):
                    ↑           ↑            ↑              ↑
                   第1个        第2个         第3个          第4个
    
    Python 的 MRO (Method Resolution Order) 决定方法查找顺序：
    1. 先查找 MyModel 自己的方法
    2. 然后按继承列表从左到右查找 Mixin
    3. ModelMixin → ConfigMixin → PeftAdapterMixin → CacheMixin
    
    如果多个 Mixin 有同名方法，左边的优先。
    
    可以用 MyModel.__mro__ 查看完整的方法解析顺序。
    
    【完整前向传播流程】
    
    输入:
        hidden_states: (B, T, C, H, W) = (1, 49, 16, 60, 90) ← 视频 latents
                       注意: PyTorch 使用 (B, T, C, H, W)，Flax 使用 (B, T, H, W, C)
        encoder_hidden_states: (B, 226, 4096) ← T5 文本编码
        timestep: 500 ← 当前去噪步骤
    
    Step 1: 时间步嵌入
        ┌─────────────────────────────────────────────────────────────────┐
        │ timestep = 500                                                  │
        │     ↓                                                           │
        │ time_proj (Timesteps): 正弦编码 → (B, inner_dim)                │
        │     ↓                           5B模型: (B, 3072)               │
        │ time_embedding (TimestepEmbedding): MLP → (B, 512)              │
        │     ↓                                                           │
        │ emb = 时间嵌入，用于调制所有 Transformer 块                      │
        └─────────────────────────────────────────────────────────────────┘
    
    Step 2: Patch 嵌入
        ┌─────────────────────────────────────────────────────────────────┐
        │ patch_embed (CogVideoXPatchEmbed):                              │
        │                                                                 │
        │ 视频: (B, 49, 16, 60, 90) → patchify → (B, 66150, 3072)        │
        │ 文本: (B, 226, 4096) → Linear → (B, 226, 3072)                  │
        │                                                                 │
        │ 拼接后分开（见前面关于拼接的讨论）                               │
        └─────────────────────────────────────────────────────────────────┘
    
    Step 3: Transformer 块 × N
        ┌─────────────────────────────────────────────────────────────────┐
        │ for block in transformer_blocks:  # 2B: 30层, 5B: 42层         │
        │     hidden_states, encoder_hidden_states = block(...)           │
        │                                                                 │
        │ 支持 gradient_checkpointing 节省显存                            │
        └─────────────────────────────────────────────────────────────────┘
    
    Step 4: 最终输出
        ┌─────────────────────────────────────────────────────────────────┐
        │ hidden_states: (B, 66150, 3072)                                 │
        │     ↓                                                           │
        │ norm_final (LayerNorm)                                          │
        │     ↓                                                           │
        │ norm_out (AdaLayerNorm): 最后一次时间调制                        │
        │     ↓                                                           │
        │ proj_out (Linear): → (B, 66150, 64)                             │
        │     ↓                   64 = patch_size² × out_channels         │
        │ unpatchify: 重组回视频格式                                       │
        │     ↓                                                           │
        │ output: (B, 49, 16, 60, 90)                                     │
        └─────────────────────────────────────────────────────────────────┘

    Parameters:
        num_attention_heads (`int`, defaults to `30`):
            多头注意力的头数 (2B=30, 5B=48)
        attention_head_dim (`int`, defaults to `64`):
            每个注意力头的维度，inner_dim = heads × head_dim
        in_channels (`int`, defaults to `16`):
            输入视频 latent 的通道数 (VAE 编码后)
        out_channels (`int`, *optional*, defaults to `16`):
            输出通道数，通常与 in_channels 相同
        flip_sin_to_cos (`bool`, defaults to `True`):
            时间嵌入中 sin/cos 的顺序
        time_embed_dim (`int`, defaults to `512`):
            时间嵌入的最终维度，用于调制各层
        ofs_embed_dim (`int`, defaults to `512`):
            光流尺度嵌入维度（仅 CogVideoX1.5-5B I2V 使用）
        text_embed_dim (`int`, defaults to `4096`):
            T5 文本编码器输出维度
        num_layers (`int`, defaults to `30`):
            Transformer 块数量 (2B=30, 5B=42)
        dropout (`float`, defaults to `0.0`):
            Dropout 概率
        attention_bias (`bool`, defaults to `True`):
            注意力投影是否使用 bias
        sample_width (`int`, defaults to `90`):
            latent 空间宽度
        sample_height (`int`, defaults to `60`):
            latent 空间高度
        sample_frames (`int`, defaults to `49`):
            latent 空间帧数
        patch_size (`int`, defaults to `2`):
            空间 patch 大小
        patch_size_t (`int`, *optional*):
            时间 patch 大小（CogVideoX 1.5 使用）
        temporal_compression_ratio (`int`, defaults to `4`):
            VAE 时间压缩比
        max_text_seq_length (`int`, defaults to `226`):
            最大文本 token 数
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            FFN 激活函数
        timestep_activation_fn (`str`, defaults to `"silu"`):
            时间嵌入 MLP 激活函数
        norm_elementwise_affine (`bool`, defaults to `True`):
            LayerNorm 是否有可学习参数
        norm_eps (`float`, defaults to `1e-5`):
            LayerNorm epsilon
        spatial_interpolation_scale (`float`, defaults to `1.875`):
            空间位置编码插值尺度
        temporal_interpolation_scale (`float`, defaults to `1.0`):
            时间位置编码插值尺度
        use_rotary_positional_embeddings (`bool`, defaults to `False`):
            是否使用 RoPE 旋转位置编码
        use_learned_positional_embeddings (`bool`, defaults to `False`):
            是否使用可学习位置编码
        patch_bias (`bool`, defaults to `True`):
            Patch 投影是否使用 bias
    """

    # ══════════════════════════════════════════════════════════════════════════
    # 类属性 - 用于 HuggingFace 工具链
    # ══════════════════════════════════════════════════════════════════════════
    
    # 在层级混合精度时跳过这些模块（保持高精度）
    _skip_layerwise_casting_patterns = ["patch_embed", "norm"]
    
    # 支持梯度检查点（节省显存）
    _supports_gradient_checkpointing = True
    
    # 模型并行时不分割这些模块
    _no_split_modules = ["CogVideoXBlock", "CogVideoXPatchEmbed"]

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 30,
        attention_head_dim: int = 64,
        in_channels: int = 16,
        out_channels: Optional[int] = 16,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        time_embed_dim: int = 512,
        ofs_embed_dim: Optional[int] = None,
        text_embed_dim: int = 4096,
        num_layers: int = 30,
        dropout: float = 0.0,
        attention_bias: bool = True,
        sample_width: int = 90,
        sample_height: int = 60,
        sample_frames: int = 49,
        patch_size: int = 2,
        patch_size_t: Optional[int] = None,
        temporal_compression_ratio: int = 4,
        max_text_seq_length: int = 226,
        activation_fn: str = "gelu-approximate",
        timestep_activation_fn: str = "silu",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        spatial_interpolation_scale: float = 1.875,
        temporal_interpolation_scale: float = 1.0,
        use_rotary_positional_embeddings: bool = False,
        use_learned_positional_embeddings: bool = False,
        patch_bias: bool = True,
    ):
        """
        初始化 CogVideoX Transformer 3D 模型。
        
        【inner_dim 计算】
        
        inner_dim = num_attention_heads × attention_head_dim
        
        ┌─────────────────────────────────────────────────────────────────────────┐
        │ 模型            │ heads │ head_dim │ inner_dim │ layers                 │
        ├─────────────────────────────────────────────────────────────────────────┤
        │ CogVideoX-2B   │ 30    │ 64       │ 1920      │ 30                     │
        │ CogVideoX-5B   │ 48    │ 64       │ 3072      │ 42                     │
        └─────────────────────────────────────────────────────────────────────────┘
        
        【与 Flax 版本的对应】
        结构完全相同，只是用 PyTorch 实现
        """
        super().__init__()
        
        # ══════════════════════════════════════════════════════════════════════
        # 计算核心维度
        # ══════════════════════════════════════════════════════════════════════
        inner_dim = num_attention_heads * attention_head_dim
        # 5B模型: 48 × 64 = 3072
        # 2B模型: 30 × 64 = 1920

        # 位置编码组合检查
        if not use_rotary_positional_embeddings and use_learned_positional_embeddings:
            raise ValueError(
                "There are no CogVideoX checkpoints available with disable rotary embeddings and learned positional "
                "embeddings. If you're using a custom model and/or believe this should be supported, please open an "
                "issue at https://github.com/huggingface/diffusers/issues."
            )

        # ══════════════════════════════════════════════════════════════════════
        # 1. Patch Embedding 模块
        # ══════════════════════════════════════════════════════════════════════
        #
        # 将视频 latents 和文本 embeddings 转换成统一的 token 序列
        #
        # 视频处理流程:
        #   (B, T, C, H, W) → patchify → (B, num_patches, inner_dim)
        #   (1, 49, 16, 60, 90) → (1, 66150, 3072) [5B模型]
        #
        #   num_patches = T × (H/patch_size) × (W/patch_size)
        #               = 49 × 30 × 45 = 66150
        #
        # 文本处理流程:
        #   (B, text_len, text_dim) → Linear → (B, text_len, inner_dim)
        #   (1, 226, 4096) → (1, 226, 3072) [5B模型]
        #
        # 【与 Flax 版本的对应】
        # Flax: FlaxCogVideoXPatchEmbed
        #
        self.patch_embed = CogVideoXPatchEmbed(
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            in_channels=in_channels,
            embed_dim=inner_dim,
            text_embed_dim=text_embed_dim,
            bias=patch_bias,
            sample_width=sample_width,
            sample_height=sample_height,
            sample_frames=sample_frames,
            temporal_compression_ratio=temporal_compression_ratio,
            max_text_seq_length=max_text_seq_length,
            spatial_interpolation_scale=spatial_interpolation_scale,
            temporal_interpolation_scale=temporal_interpolation_scale,
            use_positional_embeddings=not use_rotary_positional_embeddings,
            use_learned_positional_embeddings=use_learned_positional_embeddings,
        )
        self.embedding_dropout = nn.Dropout(dropout)

        # ══════════════════════════════════════════════════════════════════════
        # 2. 时间步嵌入模块
        # ══════════════════════════════════════════════════════════════════════
        #
        # 两步处理时间步:
        #
        # time_proj (Timesteps): 标量 → 正弦编码向量
        #   timestep = 500 → (B, inner_dim) = (B, 3072) [5B模型]
        #
        # time_embedding (TimestepEmbedding): 正弦编码 → MLP → 时间嵌入
        #   (B, inner_dim) → Linear → SiLU → Linear → (B, time_embed_dim)
        #   (B, 3072) → (B, 512)
        #
        # 【为什么要降维？】
        # 详见 Flax 版本的 FlaxTimestepEmbedding 注释
        # 简单说: 512 维足够，且节省参数量
        #
        # 【与 Flax 版本的对应】
        # Flax: FlaxTimesteps + FlaxTimestepEmbedding
        #
        self.time_proj = Timesteps(inner_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(inner_dim, time_embed_dim, timestep_activation_fn)

        # ──────────────────────────────────────────────────────────────────────
        # 可选: OFS (Optical Flow Scale) 嵌入
        # ──────────────────────────────────────────────────────────────────────
        # 仅 CogVideoX1.5-5B I2V (图生视频) 模型使用
        # 用于控制生成视频与参考图像之间的运动幅度
        #
        self.ofs_proj = None
        self.ofs_embedding = None
        if ofs_embed_dim:
            self.ofs_proj = Timesteps(ofs_embed_dim, flip_sin_to_cos, freq_shift)
            self.ofs_embedding = TimestepEmbedding(
                ofs_embed_dim, ofs_embed_dim, timestep_activation_fn
            )

        # ══════════════════════════════════════════════════════════════════════
        # 3. Transformer 块堆叠
        # ══════════════════════════════════════════════════════════════════════
        #
        # 核心计算单元，重复 num_layers 次
        # - CogVideoX-2B: 30 层
        # - CogVideoX-5B: 42 层
        #
        # 每个块包含:
        # - LayerNormZero + 联合注意力 + 门控残差
        # - LayerNormZero + FFN + 门控残差
        #
        # 【nn.ModuleList vs Flax nnx.List】
        # PyTorch: nn.ModuleList 自动注册子模块参数
        # Flax:    nnx.List 是 Flax NNX 的等价物
        #
        # ══════════════════════════════════════════════════════════════════════
        # 【Python 列表推导式语法详解】
        # ══════════════════════════════════════════════════════════════════════
        #
        # 下面这段代码使用了 Python 的「列表推导式」(List Comprehension)：
        #
        #     [CogVideoXBlock(...) for _ in range(num_layers)]
        #
        # 这相当于以下普通循环的简洁写法：
        #
        #     blocks = []
        #     for _ in range(num_layers):  # 循环 num_layers 次（如 30 或 42 次）
        #         block = CogVideoXBlock(...)  # 每次创建一个新的块
        #         blocks.append(block)         # 添加到列表中
        #
        # ──────────────────────────────────────────────────────────────────────
        # 【`for _ in range(num_layers)` 各部分解释】
        # ──────────────────────────────────────────────────────────────────────
        #
        # 1. `range(num_layers)`:
        #    ─────────────────────
        #    - range() 是 Python 内置函数，生成一个整数序列
        #    - range(30) 生成: 0, 1, 2, 3, ..., 29 （共 30 个数）
        #    - range(42) 生成: 0, 1, 2, 3, ..., 41 （共 42 个数）
        #
        #    相当于告诉 Python："我要循环 num_layers 次"
        #
        # 2. `_` (下划线变量):
        #    ─────────────────────
        #    - 这是一个约定俗成的「丢弃变量」命名
        #    - 表示：我们不关心循环变量的值，只需要循环指定次数
        #
        #    对比两种写法：
        #
        #    【使用 _】（本代码的写法）
        #    for _ in range(3):    # 循环 3 次，不关心当前是第几次
        #        print("hello")    # 输出: hello hello hello
        #
        #    【使用普通变量 i】
        #    for i in range(3):    # i 依次是 0, 1, 2
        #        print(f"第 {i} 次")  # 输出: 第 0 次, 第 1 次, 第 2 次
        #
        #    在这里用 `_` 是因为每个 CogVideoXBlock 的初始化参数完全相同，
        #    我们不需要知道当前是第几层。
        #
        # 3. `for _ in range(num_layers)`:
        #    ─────────────────────
        #    组合起来意思是：重复执行 num_layers 次
        #
        #    例如 num_layers=42（5B模型）:
        #    - 创建第 0 个 CogVideoXBlock（_ = 0，但我们不使用这个值）
        #    - 创建第 1 个 CogVideoXBlock（_ = 1，但我们不使用这个值）
        #    - ...
        #    - 创建第 41 个 CogVideoXBlock（_ = 41，但我们不使用这个值）
        #    - 总共创建 42 个完全相同结构的块
        #
        # ──────────────────────────────────────────────────────────────────────
        # 【为什么所有块的参数都一样？】
        # ──────────────────────────────────────────────────────────────────────
        #
        # 虽然参数相同，但每个块有**独立的可学习权重**！
        #
        # 每次调用 CogVideoXBlock(...) 都会：
        # - 创建新的 nn.Linear 层（随机初始化权重）
        # - 创建新的 LayerNorm 层
        # - 创建新的 Attention 层
        #
        # 所以 42 个块虽然结构相同，但权重是独立的，训练后会学到不同的值。
        #
        # 类比：
        # - 结构相同 = 42 个相同户型的公寓
        # - 权重独立 = 每个公寓的装修和家具都不同
        #
        # ──────────────────────────────────────────────────────────────────────
        # 【等价的普通 for 循环写法】
        # ──────────────────────────────────────────────────────────────────────
        #
        # 如果不用列表推导式，可以这样写（功能完全相同）：
        #
        #     transformer_blocks = []
        #     for layer_idx in range(num_layers):
        #         block = CogVideoXBlock(
        #             dim=inner_dim,
        #             num_attention_heads=num_attention_heads,
        #             attention_head_dim=attention_head_dim,
        #             time_embed_dim=time_embed_dim,
        #             dropout=dropout,
        #             activation_fn=activation_fn,
        #             attention_bias=attention_bias,
        #             norm_elementwise_affine=norm_elementwise_affine,
        #             norm_eps=norm_eps,
        #         )
        #         transformer_blocks.append(block)
        #     self.transformer_blocks = nn.ModuleList(transformer_blocks)
        #
        # 列表推导式更简洁，是 Python 的惯用写法。
        #
        self.transformer_blocks = nn.ModuleList(
            [
                CogVideoXBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    time_embed_dim=time_embed_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                )
                for _ in range(num_layers)  # ← 创建 num_layers 个相同结构的块
            ]
        )
        
        # 最终 LayerNorm（不带时间调制）
        self.norm_final = nn.LayerNorm(inner_dim, norm_eps, norm_elementwise_affine)

        # ══════════════════════════════════════════════════════════════════════
        # 4. 输出模块
        # ══════════════════════════════════════════════════════════════════════
        #
        # norm_out: 最后一次时间调制的 LayerNorm
        # ────────────────────────────────────────────────────────────────────
        # 输入:  hidden_states (B, num_patches, inner_dim)
        # 输出:  调制后的特征 (B, num_patches, inner_dim)
        #
        # 与中间层的 LayerNormZero 类似，但只生成 2 组参数 (shift, scale)
        #
        self.norm_out = AdaLayerNorm(
            embedding_dim=time_embed_dim,
            output_dim=2 * inner_dim,  # shift 和 scale 各占一半
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            chunk_dim=1,
        )

        # proj_out: 将 Transformer 输出投影回 patch 维度
        # ────────────────────────────────────────────────────────────────────
        # CogVideoX 1.0 (patch_size_t = None):
        #   output_dim = patch_size² × out_channels = 2 × 2 × 16 = 64
        #
        # CogVideoX 1.5 (patch_size_t != None):
        #   output_dim = patch_size² × patch_size_t × out_channels
        #
        if patch_size_t is None:
            # CogVideoX 1.0: 只有空间 patch
            output_dim = patch_size * patch_size * out_channels
        else:
            # CogVideoX 1.5: 时空联合 patch
            output_dim = patch_size * patch_size * patch_size_t * out_channels

        self.proj_out = nn.Linear(inner_dim, output_dim)

        # 梯度检查点标志（默认关闭）
        # 开启后可节省显存，但会增加计算时间
        self.gradient_checkpointing = False

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections with FusedAttnProcessor2_0->FusedCogVideoXAttnProcessor2_0
    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        > [!WARNING] > This API is 🧪 experimental.
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

        self.set_attn_processor(FusedCogVideoXAttnProcessor2_0())

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        > [!WARNING] > This API is 🧪 experimental.

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        ofs: Optional[Union[int, float, torch.LongTensor]] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[Tuple[torch.Tensor], Transformer2DModelOutput]:
        """
        CogVideoX Transformer 前向传播 - 核心推理/训练函数。
        
        【维度说明】(以 5B 模型为例)
        
        输入:
            hidden_states: (B, T, C, H, W) = (1, 49, 16, 60, 90)
                           视频 latents，PyTorch 格式（通道优先）
                           
                           【与 Flax 版本的区别】
                           PyTorch: (B, T, C, H, W) - 通道在前
                           Flax:    (B, T, H, W, C) - 通道在后
                           
            encoder_hidden_states: (B, 226, 4096)
                                   T5 文本编码器输出
                                   
            timestep: int/float/Tensor
                      当前去噪步骤，如 500
                      
            image_rotary_emb: Optional[(cos, sin)]
                              RoPE 位置编码（如果使用）
        
        输出:
            output: (B, T, C, H, W)
                    去噪后的视频 latents
        
        【完整计算流程】
        
            输入 hidden_states: (B, 49, 16, 60, 90)
                    ↓
            ┌───────────────────────────────────────────────────────────────┐
            │ Step 1: 时间步嵌入                                            │
            │   timestep → time_proj → time_embedding → emb (B, 512)       │
            │   + 可选的 ofs_embedding                                      │
            └───────────────────────────────────────────────────────────────┘
                    ↓
            ┌───────────────────────────────────────────────────────────────┐
            │ Step 2: Patch 嵌入                                            │
            │   视频: (B,49,16,60,90) → (B, 66150, 3072)                   │
            │   文本: (B, 226, 4096) → (B, 226, 3072)                       │
            │   拼接 → 分割                                                 │
            └───────────────────────────────────────────────────────────────┘
                    ↓
            ┌───────────────────────────────────────────────────────────────┐
            │ Step 3: Transformer 块 × 42                                   │
            │   (hidden, encoder, emb) → block → (hidden', encoder')       │
            │   支持 gradient_checkpointing                                 │
            └───────────────────────────────────────────────────────────────┘
                    ↓
            ┌───────────────────────────────────────────────────────────────┐
            │ Step 4: 最终输出                                              │
            │   norm_final → norm_out → proj_out                           │
            │   (B, 66150, 3072) → (B, 66150, 64)                          │
            └───────────────────────────────────────────────────────────────┘
                    ↓
            ┌───────────────────────────────────────────────────────────────┐
            │ Step 5: Unpatchify                                            │
            │   (B, 66150, 64) → (B, 49, 16, 60, 90)                       │
            │   把 tokens 重组回视频格式                                    │
            └───────────────────────────────────────────────────────────────┘
                    ↓
            输出: (B, 49, 16, 60, 90)
        
        Args:
            hidden_states: 视频 latents (B, T, C, H, W)
            encoder_hidden_states: 文本嵌入 (B, text_len, text_dim)
            timestep: 去噪时间步
            timestep_cond: 额外的时间步条件（CogVideoX 未使用）
            ofs: 光流尺度（CogVideoX1.5-5B I2V 使用）
            image_rotary_emb: RoPE 位置编码 (cos, sin)
            attention_kwargs: 注意力额外参数（如 LoRA scale）
            return_dict: 是否返回字典格式
            
        Returns:
            去噪后的视频 latents
        """
        
        # ══════════════════════════════════════════════════════════════════════
        # LoRA 处理（PyTorch 特有，Flax 版本没有）
        # ══════════════════════════════════════════════════════════════════════
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # 设置 LoRA 权重缩放因子
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        # 提取输入维度
        batch_size, num_frames, channels, height, width = hidden_states.shape
        # 例如: batch=1, frames=49, channels=16, height=60, width=90

        # ══════════════════════════════════════════════════════════════════════
        # Step 1: 时间步嵌入
        # ══════════════════════════════════════════════════════════════════════
        #
        # timestep = 500 (标量)
        #     ↓ time_proj (正弦编码)
        # t_emb = (B, inner_dim) = (B, 3072) [5B模型]
        #     ↓ time_embedding (MLP)
        # emb = (B, time_embed_dim) = (B, 512)
        #
        timesteps = timestep
        t_emb = self.time_proj(timesteps)

        # 类型转换：time_proj 输出 float32，需要与输入保持一致
        # 【与 Flax 版本的 dtype 问题对应】
        # 这里的处理与 Flax 版本相同，确保类型一致性
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        # 可选: 添加 OFS 嵌入（图生视频模型使用）
        if self.ofs_embedding is not None:
            ofs_emb = self.ofs_proj(ofs)
            ofs_emb = ofs_emb.to(dtype=hidden_states.dtype)
            ofs_emb = self.ofs_embedding(ofs_emb)
            emb = emb + ofs_emb

        # ══════════════════════════════════════════════════════════════════════
        # Step 2: Patch 嵌入
        # ══════════════════════════════════════════════════════════════════════
        #
        # patch_embed 会:
        # 1. 将视频 patchify 成 token 序列
        # 2. 将文本投影到相同维度
        # 3. 拼接两者
        #
        # 【你之前问的问题】
        # "先拼接再分开是什么意思？"
        #
        # 这里拼接后立即分开，确实是多余的操作。
        # 可能的原因：
        # - patch_embed 内部可能需要对拼接后的序列加位置编码
        # - 或者只是从其他代码复制来的习惯
        #
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)

        text_seq_length = encoder_hidden_states.shape[1]  # 226
        encoder_hidden_states = hidden_states[:, :text_seq_length]  # 文本 tokens
        hidden_states = hidden_states[:, text_seq_length:]           # 视频 tokens

        # ══════════════════════════════════════════════════════════════════════
        # Step 3: Transformer 块
        # ══════════════════════════════════════════════════════════════════════
        #
        # 循环 42 次 (5B模型) 或 30 次 (2B模型)
        # 每个块: LayerNormZero → Attention → LayerNormZero → FFN
        #
        for i, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                # 梯度检查点: 前向时不保存中间激活，反向时重新计算
                # 节省显存，但增加计算时间（约 30%）
                hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    emb,
                    image_rotary_emb,
                    attention_kwargs,
                )
            else:
                # 正常前向传播
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                    attention_kwargs=attention_kwargs,
                )

        # 最终 LayerNorm（不带时间调制）
        hidden_states = self.norm_final(hidden_states)

        # ══════════════════════════════════════════════════════════════════════
        # Step 4: 最终输出处理
        # ══════════════════════════════════════════════════════════════════════
        #
        # norm_out: 最后一次时间调制
        # proj_out: 投影回 patch 维度
        #
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)
        # 现在 hidden_states: (B, 66150, 64) [5B模型]
        # 64 = patch_size² × out_channels = 2 × 2 × 16

        # ══════════════════════════════════════════════════════════════════════
        # Step 5: Unpatchify - 把 tokens 重组回视频格式
        # ══════════════════════════════════════════════════════════════════════
        #
        # 这是 patchify 的逆操作
        #
        # 输入: (B, num_patches, patch_dim)
        # 输出: (B, T, C, H, W)
        #
        p = self.config.patch_size      # 空间 patch 大小，如 2
        p_t = self.config.patch_size_t  # 时间 patch 大小，CogVideoX 1.0 为 None

        if p_t is None:
            # ──────────────────────────────────────────────────────────────────
            # CogVideoX 1.0: 只有空间 patch，每帧独立处理
            # ──────────────────────────────────────────────────────────────────
            #
            # hidden_states: (B, num_patches, patch_dim)
            #              = (B, 66150, 64)
            #              = (B, 49 × 30 × 45, 2 × 2 × 16)
            #
            # 重塑步骤:
            # 1. reshape 分离各个维度:
            #    (B, T, H/p, W/p, C, p, p) = (B, 49, 30, 45, 16, 2, 2)
            #
            # 2. permute 调整顺序:
            #    (B, T, C, H/p, p, W/p, p) = (B, 49, 16, 30, 2, 45, 2)
            #
            # 3. flatten 合并 patch:
            #    (B, T, C, H, W) = (B, 49, 16, 60, 90)
            #
            output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
            output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
        else:
            # ──────────────────────────────────────────────────────────────────
            # CogVideoX 1.5: 时空联合 patch
            # ──────────────────────────────────────────────────────────────────
            #
            # 更复杂的 unpatchify，需要同时处理时间和空间维度
            #
            output = hidden_states.reshape(
                batch_size, (num_frames + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p
            )
            output = output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)

        # ══════════════════════════════════════════════════════════════════════
        # 清理 LoRA（PyTorch 特有）
        # ══════════════════════════════════════════════════════════════════════
        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        # 返回结果
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
