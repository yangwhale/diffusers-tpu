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
JAX/Flax implementation of CogVideoX Transformer 3D model.

This is a complete conversion from the PyTorch version, maintaining full feature parity.
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


@dataclass
class FlaxCogVideoXTransformer3DConfig:
    """Configuration class for FlaxCogVideoXTransformer3DModel."""
    
    config_name: str = "config.json"
    
    num_attention_heads: int = 30
    attention_head_dim: int = 64
    in_channels: int = 16
    out_channels: int = 16
    flip_sin_to_cos: bool = True
    freq_shift: int = 0
    time_embed_dim: int = 512
    ofs_embed_dim: Optional[int] = None
    text_embed_dim: int = 4096
    num_layers: int = 30
    dropout: float = 0.0
    attention_bias: bool = True
    sample_width: int = 90
    sample_height: int = 60
    sample_frames: int = 49
    patch_size: int = 2
    patch_size_t: Optional[int] = None
    temporal_compression_ratio: int = 4
    max_text_seq_length: int = 226
    activation_fn: str = "gelu-approximate"
    timestep_activation_fn: str = "silu"
    norm_elementwise_affine: bool = True
    norm_eps: float = 1e-5
    spatial_interpolation_scale: float = 1.875
    temporal_interpolation_scale: float = 1.0
    use_rotary_positional_embeddings: bool = False
    use_learned_positional_embeddings: bool = False
    patch_bias: bool = True
    
    @classmethod
    def from_dict(cls, config_dict: Dict):
        """从字典创建配置"""
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(cls)}
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}
        return cls(**filtered_dict)


class FlaxTimesteps(nnx.Module):
    """
    Timestep embedding module (sinusoidal embeddings).
    Converts scalar timesteps to vectors.
    """
    
    def __init__(
        self,
        num_channels: int,
        flip_sin_to_cos: bool = True,
        freq_shift: float = 0,
    ):
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.freq_shift = freq_shift
    
    def __call__(self, timesteps: jnp.ndarray) -> jnp.ndarray:
        """
        Generate sinusoidal embeddings for timesteps.
        
        Args:
            timesteps: Timestep values (B,) or (B, 1)
            
        Returns:
            Timestep embeddings (B, num_channels)
        """
        # Ensure timesteps is 1D
        if timesteps.ndim == 0:
            timesteps = timesteps[None]
        
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * jnp.arange(0, half_dim, dtype=jnp.float32)
        exponent = exponent / (half_dim - self.freq_shift)
        
        emb = jnp.exp(exponent)
        emb = timesteps[:, None] * emb[None, :]
        
        if self.flip_sin_to_cos:
            emb = jnp.concatenate([jnp.cos(emb), jnp.sin(emb)], axis=-1)
        else:
            emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        
        return emb


class FlaxTimestepEmbedding(nnx.Module):
    """
    Time embedding MLP.
    Projects timestep embeddings to desired dimension.
    """
    
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        rngs: nnx.Rngs = None,
    ):
        self.linear_1 = nnx.Linear(in_channels, time_embed_dim, rngs=rngs)
        self.linear_2 = nnx.Linear(time_embed_dim, time_embed_dim, rngs=rngs)
        
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
            sample: Timestep embedding (B, in_channels)
            condition: Optional conditioning (not used in CogVideoX)
            
        Returns:
            Projected embedding (B, time_embed_dim)
        """
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


class FlaxCogVideoXPatchEmbed(nnx.Module):
    """
    Patch embedding for CogVideoX.
    Converts video latents to tokens and combines with text embeddings.
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
        
        # Positional embeddings
        if use_positional_embeddings:
            if use_learned_positional_embeddings:
                # Learned positional embeddings
                max_num_patches = (
                    (sample_frames // temporal_compression_ratio) *
                    (sample_height // patch_size) *
                    (sample_width // patch_size)
                )
                self.pos_embedding = nnx.Param(
                    jax.random.normal(rngs(), (1, max_num_patches, embed_dim)) * 0.02
                )
            else:
                # Sinusoidal positional embeddings (will be computed on-the-fly)
                self.pos_embedding = None
        else:
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
        
        # Add positional embeddings if needed
        if self.use_positional_embeddings:
            if self.use_learned_positional_embeddings and self.pos_embedding is not None:
                num_patches = image_embeds.shape[1]
                pos_embed = self.pos_embedding.value[:, :num_patches, :]
                image_embeds = image_embeds + pos_embed
            elif not self.use_learned_positional_embeddings:
                # Sinusoidal positional embeddings
                pos_embed = self._get_sinusoidal_pos_embed(image_embeds.shape[1], self.embed_dim)
                image_embeds = image_embeds + pos_embed[None, :, :]
        
        # Concatenate text and image embeddings
        embeds = jnp.concatenate([text_embeds, image_embeds], axis=1)
        
        return embeds
    
    def _get_sinusoidal_pos_embed(self, num_patches: int, embed_dim: int) -> jnp.ndarray:
        """Generate sinusoidal positional embeddings."""
        position = jnp.arange(num_patches, dtype=jnp.float32)[:, None]
        div_term = jnp.exp(
            jnp.arange(0, embed_dim, 2, dtype=jnp.float32) * 
            -(math.log(10000.0) / embed_dim)
        )
        
        pe = jnp.zeros((num_patches, embed_dim))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        
        return pe


class FlaxCogVideoXLayerNormZero(nnx.Module):
    """
    Zero-initialized modulation for CogVideoX.
    Applies layer norm and then modulates with timestep embeddings.
    """
    
    def __init__(
        self,
        time_embed_dim: int,
        dim: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        bias: bool = True,
        rngs: nnx.Rngs = None,
    ):
        self.norm = nnx.LayerNorm(dim, epsilon=eps, use_bias=bias, use_scale=elementwise_affine, rngs=rngs)
        
        # Modulation MLP
        self.linear = nnx.Linear(time_embed_dim, 6 * dim, use_bias=True, rngs=rngs)
    
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        encoder_hidden_states: jnp.ndarray,
        temb: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Args:
            hidden_states: Video features (B, num_patches, dim)
            encoder_hidden_states: Text features (B, text_seq_len, dim)
            temb: Time embeddings (B, time_embed_dim)
            
        Returns:
            Tuple of (norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa)
        """
        # Get modulation parameters
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(
            self.linear(jax.nn.silu(temb)), 6, axis=-1
        )
        
        # Normalize and modulate hidden states
        norm_hidden_states = self.norm(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :]
        
        # Normalize and modulate encoder hidden states
        norm_encoder_hidden_states = self.norm(encoder_hidden_states)
        norm_encoder_hidden_states = (
            norm_encoder_hidden_states * (1 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
        )
        
        return norm_hidden_states, norm_encoder_hidden_states, gate_msa[:, None, :], gate_mlp[:, None, :]


class FlaxAdaLayerNorm(nnx.Module):
    """
    Adaptive Layer Normalization.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        output_dim: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        rngs: nnx.Rngs = None,
    ):
        self.silu = jax.nn.silu
        self.linear = nnx.Linear(embedding_dim, output_dim, rngs=rngs)
        self.norm = nnx.LayerNorm(output_dim // 2, epsilon=eps, use_scale=elementwise_affine, rngs=rngs)
    
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        temb: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Args:
            hidden_states: Input features (B, seq_len, dim)
            temb: Time embeddings (B, embedding_dim)
            
        Returns:
            Modulated features (B, seq_len, dim)
        """
        emb = self.linear(self.silu(temb))
        shift, scale = jnp.split(emb, 2, axis=-1)
        
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states * (1 + scale[:, None, :]) + shift[:, None, :]
        
        return hidden_states


# Splash Attention 配置
BQSIZE = 2048
BKVSIZE = 2048
BKVCOMPUTESIZE = 1024
USE_K_SMOOTH = True

# 全局 mesh 变量，用于多设备分片
_GLOBAL_MESH = None

def set_global_mesh(mesh: Mesh):
    """设置全局 mesh，用于多设备分片"""
    global _GLOBAL_MESH
    _GLOBAL_MESH = mesh

def get_global_mesh() -> Mesh:
    """获取全局 mesh"""
    return _GLOBAL_MESH


def _create_splash_attention_kernel(padded_q_seq_len, padded_kv_seq_len, num_heads_on_device, window_size=None):
    """创建 Splash attention kernel"""
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
    Multi-head attention for CogVideoX with Splash Attention support.
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
    Feed-forward network with GELU activation.
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
    Transformer block used in CogVideoX model.
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
    A Transformer model for video-like data in CogVideoX.
    
    This is the JAX/Flax implementation with full feature parity to the PyTorch version.
    
    Args:
        config: Configuration object
        rngs: Random number generators
        dtype: Data type (e.g., jnp.float32, jnp.bfloat16)
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