# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT
# Optimized Flash Attention without padding
# Source: HunyuanVideo-1.5 official implementation

from einops import rearrange


def flash_attn_no_pad(
    qkv, key_padding_mask, causal=False, dropout_p=0.0, softmax_scale=None, deterministic=False
):
    """
    Optimized Flash Attention that handles variable-length sequences without padding.
    
    Args:
        qkv: [B, S, 3, H, D] tensor containing query, key, value
        key_padding_mask: [B, S] boolean mask (True for valid positions)
        causal: whether to use causal masking
        dropout_p: dropout probability
        softmax_scale: scaling factor for softmax
        deterministic: whether to use deterministic mode
    
    Returns:
        output: [B, S, H, D] attention output
    """
    from flash_attn import flash_attn_varlen_qkvpacked_func
    from flash_attn.bert_padding import pad_input, unpad_input
    
    batch_size = qkv.shape[0]
    seqlen = qkv.shape[1]
    nheads = qkv.shape[-2]
    
    # Reshape for unpadding
    x = rearrange(qkv, "b s three h d -> b s (three h d)")
    x_unpad, indices, cu_seqlens, max_s, used_seqlens_in_batch = unpad_input(
        x, key_padding_mask
    )

    # Reshape back to qkv format
    x_unpad = rearrange(x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads)
    
    # Call optimized varlen function
    output_unpad = flash_attn_varlen_qkvpacked_func(
        x_unpad,
        cu_seqlens,
        max_s,
        dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        deterministic=deterministic,
    )
    
    # Pad output back to original shape
    output = rearrange(
        pad_input(
            rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, batch_size, seqlen
        ),
        "b s (h d) -> b s h d",
        h=nheads,
    )
    return output