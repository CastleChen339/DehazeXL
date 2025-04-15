import torch
from torch import nn


class LLMAttention(nn.Module):
    def __init__(
            self,
            dim,

            inner_dim,
            num_heads,
            causal=False,
    ):
        super().__init__()
        self.qkv = nn.Linear(dim, inner_dim * 3, bias=True)
        self.proj = nn.Linear(inner_dim, dim)
        assert inner_dim % num_heads == 0, (inner_dim, num_heads)
        self.num_heads = num_heads

        # Delayed import only if needed (but relative so shouldn't be a problem)
        from .hyper_attn.attention.hyper_attn import HyperAttention

        self.attn = HyperAttention(
            input_dim=inner_dim // num_heads,
            lsh_num_projs=7,
            block_size=256,
            sample_size=256,
            min_seq_len=4096,
        )
        self.causal = causal

    def forward(self, x):
        """
        X: N L H
        """
        B, L, D = x.shape
        q, k, v = (
            self.qkv(x).reshape(B, L, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        )  # B H L D // num_heads
        attn_out = self.attn(q, k, v, causal=self.causal).permute(
            0, 2, 1, 3
        )  # B H L D // num_heads
        attn_out = attn_out.reshape(B, L, -1).contiguous()
        attn_out = self.proj(attn_out)

        return attn_out


class ViTAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            use_mask=False,
            **kwargs,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.use_mask = use_mask
        if use_mask:
            self.att_mask = nn.Parameter(torch.Tensor(self.num_heads, 196, 196))

    def forward(self, x):
        B, N, C = x.shape

        q, k, v = (
            self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.use_mask:
            attn = attn * torch.sigmoid(self.att_mask).expand(B, -1, -1, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
