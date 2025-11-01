# Copyright 2025 The HuggingFace Team.
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
Utilities for Flax VAE models including:
- PyTorch compatibility wrapper
- Weight loading from PyTorch checkpoints
- Format conversion helpers
"""

import re
from typing import Any, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx
from flax.traverse_util import unflatten_dict

from .vae import DecoderOutput, DiagonalGaussianDistribution
from ..modeling_outputs import AutoencoderKLOutput


def to_jax_recursive(x):
    """
    Recursively convert PyTorch tensors to JAX arrays.
    Converts 5D video tensors from PyTorch (B, C, T, H, W) to JAX (B, T, H, W, C) format.
    
    Args:
        x: Input (can be Tensor, list, tuple, dict, or other)
        
    Returns:
        Converted JAX array or structure
    """
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
        # Convert 5D video tensors: (B, C, T, H, W) -> (B, T, H, W, C)
        if arr.ndim == 5:
            arr = arr.transpose(0, 2, 3, 4, 1)
        # Special handling for BFloat16
        if x.dtype == torch.bfloat16:
            return jnp.array(arr.astype(np.float32)).astype(jnp.bfloat16)
        else:
            return jnp.array(arr)
    elif isinstance(x, (list, tuple)):
        return type(x)(to_jax_recursive(xx) for xx in x)
    elif isinstance(x, dict):
        return {k: to_jax_recursive(v) for k, v in x.items()}
    else:
        return x


def to_torch_recursive(x):
    """
    Recursively convert JAX arrays to PyTorch tensors.
    Converts 5D video tensors from JAX (B, T, H, W, C) to PyTorch (B, C, T, H, W) format.
    
    Args:
        x: Input (can be JAX array, list, tuple, dict, or other)
        
    Returns:
        Converted PyTorch tensor or structure
    """
    if 'ArrayImpl' in str(type(x)) or isinstance(x, jnp.ndarray):
        # Handle JAX arrays
        np_array = np.array(x)
        # Convert 5D video tensors: (B, T, H, W, C) -> (B, C, T, H, W)
        if np_array.ndim == 5:
            np_array = np_array.transpose(0, 4, 1, 2, 3)
        # Special handling for bfloat16
        if hasattr(x, 'dtype') and x.dtype == jnp.bfloat16:
            return torch.from_numpy(np_array.astype(np.float32)).to(torch.bfloat16)
        else:
            return torch.from_numpy(np_array)
    elif isinstance(x, (list, tuple)):
        return type(x)(to_torch_recursive(xx) for xx in x)
    elif isinstance(x, dict):
        return {k: to_torch_recursive(v) for k, v in x.items()}
    elif hasattr(x, 'sample'):
        # Handle output objects with .sample attribute
        sample = to_torch_recursive(x.sample)
        if hasattr(x, 'replace'):
            return x.replace(sample=sample)
        else:
            return sample
    else:
        return x


class JAXVAEWrapper:
    """
    Wrapper to make JAX VAE compatible with PyTorch pipelines.
    
    This wrapper provides a PyTorch-compatible interface for the Flax VAE,
    handling format conversions (BCTHW ↔ BTHWC) and maintaining caching
    for spatial conditioning.
    
    Args:
        jax_vae: The Flax VAE model
        config: VAE configuration
        mesh: Optional JAX mesh for distributed computation
        dtype: Data type (e.g., torch.bfloat16)
    """
    
    def __init__(self, jax_vae, config, mesh=None, dtype=torch.bfloat16):
        self._vae = jax_vae
        self.config = config
        self._mesh = mesh
        self.dtype = dtype
        self._original_sample = None
        
        # Copy configuration attributes
        self.scaling_factor = getattr(config, 'scaling_factor', 1.15258426)
        self.use_slicing = False
        self.use_tiling = jax_vae.use_tiling
    
    def __getattr__(self, name):
        """Forward attribute access to underlying JAX VAE."""
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            return getattr(self._vae, name)
    
    def clear_cache(self):
        """Clear cached sample to free memory."""
        self._original_sample = None
    
    def enable_tiling(self, *args, **kwargs):
        """Enable tiled encoding/decoding."""
        self._vae.enable_tiling(*args, **kwargs)
        self.use_tiling = True
    
    def disable_tiling(self):
        """Disable tiled encoding/decoding."""
        self._vae.disable_tiling()
        self.use_tiling = False
    
    def enable_slicing(self):
        """Enable batch slicing."""
        self.use_slicing = True
    
    def disable_slicing(self):
        """Disable batch slicing."""
        self.use_slicing = False
    
    def encode(self, sample: torch.Tensor, return_dict: bool = True):
        """
        Encode a batch of videos into latents.
        
        Args:
            sample: Input video tensor (B, C, T, H, W) in PyTorch format
            return_dict: Whether to return AutoencoderKLOutput
            
        Returns:
            AutoencoderKLOutput or tuple with latent distribution
        """
        # Cache original sample for decoder spatial conditioning
        self._original_sample = sample
        
        # Convert to JAX format (already does BCTHW -> BTHWC)
        jax_sample = to_jax_recursive(sample)
        
        # Encode
        mean, logvar = self._vae.encode(jax_sample, deterministic=True)
        
        # Convert to PyTorch (already does BTHWC -> BCTHW)
        mean_torch = to_torch_recursive(mean)
        logvar_torch = to_torch_recursive(logvar)
        
        # Concatenate for DiagonalGaussianDistribution
        h = torch.cat([mean_torch, logvar_torch], dim=1)
        posterior = DiagonalGaussianDistribution(h)
        
        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)
    
    def decode(self, latents: torch.Tensor, return_dict: bool = True):
        """
        Decode latent representations into videos.
        
        Args:
            latents: Latent tensor (B, C, T, H, W) in PyTorch format
            return_dict: Whether to return DecoderOutput
            
        Returns:
            DecoderOutput or tuple with decoded video
        """
        # Apply scaling factor
        latents = latents / self.scaling_factor
        
        # Convert to JAX format (already does BCTHW -> BTHWC)
        jax_latents = to_jax_recursive(latents)
        
        # Prepare spatial conditioning
        if self._original_sample is not None:
            # Use cached original sample
            jax_sample = to_jax_recursive(self._original_sample)
        else:
            # Generation mode: use latents as conditioning
            jax_sample = jax_latents
        
        # Decode
        output = self._vae.decode(jax_latents, zq=jax_sample, deterministic=True)
        
        # Convert to PyTorch (already does BTHWC -> BCTHW)
        torch_output = to_torch_recursive(output)
        
        if not return_dict:
            return (torch_output,)
        return DecoderOutput(sample=torch_output)
    
    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ):
        """
        Forward pass through the VAE.
        
        Args:
            sample: Input video
            sample_posterior: Whether to sample from posterior
            return_dict: Whether to return DecoderOutput
            generator: Random generator for sampling
            
        Returns:
            Reconstructed video
        """
        # Encode
        posterior = self.encode(sample, return_dict=True).latent_dist
        
        # Sample or get mode
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        
        # Decode
        dec = self.decode(z, return_dict=return_dict)
        
        if not return_dict:
            return (dec,)
        return dec


def load_cogvideox_vae_weights(
    pretrained_model_name_or_path: str,
    subfolder: str = "vae",
    filename: str = "diffusion_pytorch_model.safetensors",
    from_hf: bool = True,
) -> Dict[str, Any]:
    """
    Load CogVideoX VAE weights from PyTorch checkpoint and convert to Flax format.
    
    Args:
        pretrained_model_name_or_path: Path or HF model ID
        subfolder: Subfolder containing the weights
        filename: Weight file name
        from_hf: Whether to download from HuggingFace Hub
        
    Returns:
        Dictionary of Flax parameters
    """
    from safetensors import safe_open
    
    if from_hf:
        from huggingface_hub import hf_hub_download
        ckpt_path = hf_hub_download(
            pretrained_model_name_or_path,
            subfolder=subfolder,
            filename=filename
        )
    else:
        import os
        ckpt_path = os.path.join(pretrained_model_name_or_path, subfolder, filename)
    
    print(f"Loading CogVideoX VAE weights from: {ckpt_path}")
    
    # Load tensors
    tensors = {}
    with safe_open(ckpt_path, framework="np") as f:
        for k in f.keys():
            tensors[k] = jnp.array(f.get_tensor(k))
    
    # Convert PyTorch keys to Flax keys
    flax_state_dict = {}
    
    for pt_key, tensor in tensors.items():
        # Remove _orig_mod prefix if present
        if pt_key.startswith("_orig_mod."):
            pt_key = pt_key[len("_orig_mod."):]
        
        flax_key = pt_key
        
        # Map encoder down blocks
        if m := re.match(r'encoder\.down_blocks\.(\d+)\.resnets\.(\d+)\.(.*)', flax_key):
            block_idx, resnet_idx, rest = m.groups()
            flax_key = f'encoder.down_block_{block_idx}.resnet_{resnet_idx}.{rest}'
        
        # Map encoder downsamplers
        elif m := re.match(r'encoder\.down_blocks\.(\d+)\.downsamplers\.0\.(.*)', flax_key):
            block_idx, rest = m.groups()
            flax_key = f'encoder.down_block_{block_idx}.downsampler_0.{rest}'
        
        # Map encoder mid block
        elif m := re.match(r'encoder\.mid_block\.resnets\.(\d+)\.(.*)', flax_key):
            resnet_idx, rest = m.groups()
            flax_key = f'encoder.mid_block.resnet_{resnet_idx}.{rest}'
        
        # Map decoder mid block
        elif m := re.match(r'decoder\.mid_block\.resnets\.(\d+)\.(.*)', flax_key):
            resnet_idx, rest = m.groups()
            flax_key = f'decoder.mid_block.resnet_{resnet_idx}.{rest}'
        
        # Map decoder up blocks
        elif m := re.match(r'decoder\.up_blocks\.(\d+)\.resnets\.(\d+)\.(.*)', flax_key):
            block_idx, resnet_idx, rest = m.groups()
            flax_key = f'decoder.up_block_{block_idx}.resnet_{resnet_idx}.{rest}'
        
        # Map decoder upsamplers
        elif m := re.match(r'decoder\.up_blocks\.(\d+)\.upsamplers\.0\.(.*)', flax_key):
            block_idx, rest = m.groups()
            flax_key = f'decoder.up_block_{block_idx}.upsampler_0.{rest}'
        
        # Add .conv for conv layers (FlaxConv3d wraps conv in .conv attribute)
        needs_conv = False
        if any(pattern in flax_key for pattern in [
            '.conv_in.', '.conv_out.', '.conv1.', '.conv2.',
            '.conv_shortcut.', '.conv_y.', '.conv_b.',
            '.downsampler_', '.upsampler_'
        ]):
            if not (flax_key.endswith('.conv.weight') or flax_key.endswith('.conv.bias')):
                if flax_key.endswith('.weight') or flax_key.endswith('.bias'):
                    needs_conv = True
        
        if needs_conv:
            parts = flax_key.rsplit('.', 1)
            flax_key = f"{parts[0]}.conv.{parts[1]}"
        
        # Handle conv weights: PyTorch (out, in, t, h, w) -> Flax (t, h, w, in, out)
        if "conv" in flax_key and "weight" in flax_key:
            flax_key = flax_key.replace(".weight", ".kernel")
            if len(tensor.shape) == 5:  # 3D conv
                tensor = tensor.transpose(2, 3, 4, 1, 0)
            elif len(tensor.shape) == 4:  # 2D conv (out, in, h, w) -> (h, w, in, out)
                tensor = tensor.transpose(2, 3, 1, 0)
        
        # Handle GroupNorm: weight -> scale
        if ".norm" in flax_key and "encoder" in flax_key:
            if ".weight" in flax_key:
                flax_key = flax_key.replace(".weight", ".scale")
        
        # Handle SpatialNorm in decoder
        if "norm_layer.weight" in flax_key:
            flax_key = flax_key.replace("norm_layer.weight", "norm_layer.scale")
        
        flax_state_dict[flax_key] = tensor
    
    # Convert to nested dict
    flax_state_dict = unflatten_dict(flax_state_dict, sep=".")
    
    print(f"Successfully loaded {len(tensors)} parameters")
    return flax_state_dict


def create_cogvideox_vae_from_pretrained(
    pretrained_model_name_or_path: str,
    config_class,
    model_class,
    rngs: nnx.Rngs,
    dtype: jnp.dtype = jnp.bfloat16,
    mesh=None,
    subfolder: str = "vae",
) -> Tuple[Any, JAXVAEWrapper]:
    """
    Create and load a CogVideoX VAE model from pretrained weights.
    
    Args:
        pretrained_model_name_or_path: HF model ID or local path
        config_class: Configuration class
        model_class: Model class
        rngs: Random number generators
        dtype: Data type for parameters
        mesh: Optional JAX mesh for distributed computation
        subfolder: Subfolder containing VAE weights
        
    Returns:
        Tuple of (flax_model, pytorch_wrapper)
    """
    # Load config
    config = config_class.from_config(
        config_class.load_config(pretrained_model_name_or_path, subfolder=subfolder)
    )
    
    # Create model
    model = model_class(config=config, rngs=rngs, dtype=dtype)
    
    # Load weights
    loaded_weights = load_cogvideox_vae_weights(
        pretrained_model_name_or_path,
        subfolder=subfolder
    )
    
    # Convert to specified dtype
    params = jax.tree_util.tree_map(lambda x: x.astype(dtype), loaded_weights)
    
    # Merge weights into model
    graphdef, _ = nnx.split(model)
    model = nnx.merge(graphdef, params)
    
    # Create PyTorch-compatible wrapper
    wrapper = JAXVAEWrapper(model, config, mesh=mesh, dtype=torch.bfloat16)
    
    print(f"CogVideoX VAE loaded successfully")
    
    return model, wrapper