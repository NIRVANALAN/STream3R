import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention
from typing import Optional, Callable

try:
    import xformers
    import xformers.ops
    from xformers.ops import memory_efficient_attention
    from xformers.ops import MemoryEfficientAttentionFlashAttentionOp, MemoryEfficientAttentionCutlassOp
    # from xformers.ops import RMSNorm

    XFORMERS_AVAILABLE = True
except ImportError:
    # logger.warning("xFormers not available")
    print("xFormers not available")
    XFORMERS_AVAILABLE = False


# Example usage and comparison
def example_usage():
    """Example showing how to use the new CausalMemEffAttention."""
    
    # Model parameters
    batch_size = 2
    seq_len = 1024
    dim = 768
    num_heads = 12
    
    # Create sample input
    x = torch.randn(batch_size, seq_len, dim, device='cuda')
    xpos = torch.arange(seq_len, device='cuda').unsqueeze(0).expand(batch_size, -1)
    
    # Initialize attention module
    attn = CausalFlexAttention(
        dim=dim,
        num_heads=num_heads,
        qk_norm=True,
        attn_drop=0.1,
        proj_drop=0.1
    ).cuda()
    
    # Forward pass
    with torch.cuda.amp.autocast():
        output = attn(x, xpos)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Memory usage: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    
    # Enhanced version with sliding window
    enhanced_attn = EnhancedCausalFlexAttention(
        dim=dim,
        num_heads=num_heads,
        window_size=128,  # Only attend to last 128 tokens
        qk_norm=True
    ).cuda()
    
    output_enhanced = enhanced_attn(x, xpos)
    print(f"Enhanced output shape: {output_enhanced.shape}")


if __name__ == "__main__":
    # Check if FlexAttention is available
    try:
        from torch.nn.attention.flex_attention import flex_attention
        print("FlexAttention is available!")
        example_usage()
    except ImportError:
        print("FlexAttention is not available in this PyTorch version.")
        print("Please update to PyTorch 2.5+ or use the nightly build.")