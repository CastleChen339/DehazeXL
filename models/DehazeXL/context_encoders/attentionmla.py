from torch import nn
import math
import torch


class AngularLSH(torch.nn.Module):
    def __init__(self, num_projs, dim, rng=None):
        super().__init__()
        self.num_projs = num_projs

        if num_projs > 0:
            self.register_buffer(
                "proj_dir",
                torch.randn(dim + (num_projs,), generator=rng),
                persistent=False,
            )
            self.register_buffer(
                "perm",
                self._unit_hamming_distance_array(self.num_projs),
                persistent=False,
            )
            self.register_buffer(
                "enc_vec",
                2 ** torch.arange(self.num_projs).view(1, 1, 1, -1),
                persistent=False,
            )

    def _unit_hamming_distance_array(self, size_n):
        if size_n == 1:
            return torch.tensor([0, 1])
        a = self._unit_hamming_distance_array(size_n - 1)
        return torch.concat([a, torch.flip(a, dims=[0]) + 2 ** (size_n - 1)], 0)

    def hash(self, mat):
        if self.num_projs < 0:
            return torch.zeros(mat.shape[:-1], device=mat.device, dtype=torch.int32)
        mask = torch.einsum("...nd,...dr -> ...nr", mat, self.proj_dir)
        mask = mask > 0
        bin_ids = (mask * self.enc_vec).sum(-1)
        return self.perm[bin_ids]

    def __repr__(self):
        return f"AngularLSH(num_proj={self.num_projs}, proj_dir.shape={self.proj_dir.shape})"


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


def apply_rope_x(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)


class LLM_mlh_Attention(nn.Module):
    def __init__(
            self,
            dim,
            inner_dim,
            num_heads,
            causal=False,
    ):
        super().__init__()

        self.proj = nn.Linear(inner_dim, dim)
        assert inner_dim % num_heads == 0, (inner_dim, num_heads)
        self.num_heads = num_heads

        # Delayed import only if needed (but relative so shouldn't be a problem)

        self.attn = MultiheadHyperLatentAttention(
            input_dim=inner_dim // num_heads,
            lsh_num_projs=7,
            block_size=256,
            sample_size=256,
            min_seq_len=4096,
        )
        self.causal = causal

    def forward(self, x, kv_cache=None):
        """
        X: N L H
        """
        B, L, D = x.shape  # 1 4096 768
        # q, k, v = (
        #    self.qkv(x).reshape(B, L, 3, se lf.num_heads, -1).permute(2, 0, 3, 1, 4)
        # )  # B H L D // num_heads  1 8 4096 96

        attn_out, kv_cache = self.attn(x, kv_cache=kv_cache)

        # .permute(
        #     0, 2, 1, 3
        # )  # B H L D // num_heads  1 4096 8 96
        attn_out = attn_out.reshape(B, L, -1).contiguous()  # 1 4096 768
        attn_out = self.proj(attn_out)  # 1 4096 768

        return attn_out, kv_cache


def indexing(x, indices, chunk_size=-1):
    """
    inputs:
        - x: 4d-tensor with shape [b, h, n, d]
        - indices: 3d-tensor with shape [b, h, s] where each entry should be in [0, n-1]
    output:
        - out: 4d-tensor with shape [b, h, s, d] where out[i,j] = x[i,j][indices[i,j],:]

    A naive implementation:
        out = torch.zeros(b, h, s, d)
        for i in range(b):
            for j in range(h):
                out[i,j] = x[i,j][idx[i,j],:]
        return out
    """
    if chunk_size < 0 or (chunk_size > 0 and x.shape[-2] % chunk_size == 0):
        return x.gather(2, indices.unsqueeze(-1).expand(-1, -1, -1, x.shape[-1]))
    else:
        x = x.gather(2, indices.unsqueeze(-1).expand(-1, -1, -1, x.shape[-1]))
        new_n = math.ceil(x.shape[2] / chunk_size) * chunk_size
        if new_n <= 0 or new_n - x.shape[2] <= 0:
            import pdb

            pdb.set_trace()
        return torch.nn.functional.pad(
            x, (0, 0, 0, new_n - x.shape[2]), mode="constant", value=0.0
        )


def add_self_attentions(attn1, lse1, attn2, lse2):
    """
    inputs:
        - attn1, attn2: 4d-tensors with shape [b, h, n, d]
        - lse1, lse2: 4d-tensors of log-sum-exp with shape [b, h, n, 1]
    output:
        - attn
        = (attn1 * exp(lse1) + attn2 * exp(lse2)) / (exp(lse1) + exp(lse2))
        = (attn1 + attn2 * exp(lse2 - lse1)) / (1 + exp(lse2-lse1))
        = attn1 * c + attn2 * (1-c), where c=1/(1 + exp(lse2-lse1)),
        - lse
        = log(exp(lse1) + exp(lse2))
        = log(exp(lse1) * (1 + exp(lse2 - lse1)))
        = lse1 + log(1 + exp(lse2 - lse1)) = lse1 - log(c)
    """
    c = (1 / (1 + (lse2 - lse1).exp())).to(dtype=attn1.dtype)
    attn = c * attn1 + (1 - c) * attn2
    lse = lse1 - (c + torch.finfo(lse1.dtype).eps).log()
    return attn


class MultiheadHyperLatentAttention(torch.nn.Module):
    def __init__(
            self,
            d_model=768,
            input_dim=96,
            lsh_num_projs=7,
            block_size=256,
            sample_size=256,
            min_seq_len=4096,
            heads=8
            , max_len=40000, rope_theta=10000.0,
            cuda=False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.lsh_num_projs = lsh_num_projs
        self.block_size = block_size
        self.sample_size = sample_size
        self.min_seq_len = min_seq_len
        self.cuda = cuda
        self.lsh = AngularLSH(num_projs=self.lsh_num_projs, dim=(1, 1, input_dim))

        self.qkv = nn.Linear(input_dim * heads, input_dim * heads * 3, bias=True)
        self.heads = heads

        self.d_model = d_model
        self.dh = d_model // heads
        self.q_proj_dim = d_model // 2
        self.kv_proj_dim = (2 * d_model) // 3

        self.qk_nope_dim = self.dh // 2
        self.qk_rope_dim = self.dh // 2

        ## Q projections
        # Lora
        self.W_dq = torch.nn.Parameter(0.01 * torch.randn((d_model, self.q_proj_dim)))
        self.W_uq = torch.nn.Parameter(0.01 * torch.randn((self.q_proj_dim, self.d_model)))
        self.q_layernorm = torch.nn.LayerNorm(self.q_proj_dim)

        ## KV projections
        # Lora
        self.W_dkv = torch.nn.Parameter(0.01 * torch.randn((d_model, self.kv_proj_dim + self.qk_rope_dim)))
        self.W_ukv = torch.nn.Parameter(0.01 * torch.randn((self.kv_proj_dim,
                                                            self.d_model + (self.heads * self.qk_nope_dim))))
        self.kv_layernorm = torch.nn.LayerNorm(self.kv_proj_dim)

        # output projection
        self.W_o = torch.nn.Parameter(0.01 * torch.randn((d_model, d_model)))

        # RoPE
        self.max_seq_len = max_len
        self.rope_theta = rope_theta

        # https://github.com/lucidrains/rotary-embedding-torch/tree/main
        # visualize emb later to make sure it looks ok
        # we do self.dh here instead of self.qk_rope_dim because its better
        freqs = 1.0 / (rope_theta ** (torch.arange(0, self.dh, 2).float() / self.dh))
        emb = torch.outer(torch.arange(self.max_seq_len).float(), freqs)
        cos_cached = emb.cos()[None, None, :, :]
        sin_cached = emb.sin()[None, None, :, :]

        # https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_buffer
        # This is like a parameter but its a constant so we can use register_buffer
        self.register_buffer("cos_cached", cos_cached)
        self.register_buffer("sin_cached", sin_cached)

    def forward(
            self, x: torch.tensor,
            scale=None,
            kv_cache=None
    ):
        # 1 4096 768

        scale = self.d_model ** (-0.5) if scale is None else scale
        attn = self.forward_no_causal_mask(x, scale, kv_cache)
        return attn

    def forward_no_causal_mask(self, x, scale, kv_cache=None, past_length=0):
        B, S, D = x.size()

        # Q Projections
        compressed_q = x @ self.W_dq
        compressed_q = self.q_layernorm(compressed_q)
        Q = compressed_q @ self.W_uq
        Q = Q.view(B, -1, self.heads, self.dh).transpose(1, 2)
        Q, Q_for_rope = torch.split(Q, [self.qk_nope_dim, self.qk_rope_dim], dim=-1)

        # Q Decoupled RoPE
        cos_q = self.cos_cached[:, :, past_length:past_length + S, :self.qk_rope_dim // 2].repeat(1, 1, 1, 2)
        sin_q = self.sin_cached[:, :, past_length:past_length + S, :self.qk_rope_dim // 2].repeat(1, 1, 1, 2)
        Q_for_rope = apply_rope_x(Q_for_rope, cos_q, sin_q)

        # KV Projections
        if kv_cache is None:
            compressed_kv = x @ self.W_dkv
            KV_for_lora, K_for_rope = torch.split(compressed_kv,
                                                  [self.kv_proj_dim, self.qk_rope_dim],
                                                  dim=-1)
            KV_for_lora = self.kv_layernorm(KV_for_lora)
        else:
            new_kv = x @ self.W_dkv
            compressed_kv = torch.cat([kv_cache, new_kv], dim=1)
            new_kv, new_K_for_rope = torch.split(new_kv,
                                                 [self.kv_proj_dim, self.qk_rope_dim],
                                                 dim=-1)
            old_kv, old_K_for_rope = torch.split(kv_cache,
                                                 [self.kv_proj_dim, self.qk_rope_dim],
                                                 dim=-1)
            new_kv = self.kv_layernorm(new_kv)
            old_kv = self.kv_layernorm(old_kv)
            KV_for_lora = torch.cat([old_kv, new_kv], dim=1)
            K_for_rope = torch.cat([old_K_for_rope, new_K_for_rope], dim=1)

        KV = KV_for_lora @ self.W_ukv
        KV = KV.view(B, -1, self.heads, self.dh + self.qk_nope_dim).transpose(1, 2)
        K, V = torch.split(KV, [self.qk_nope_dim, self.dh], dim=-1)
        S_full = K.size(2)

        # K Rope
        K_for_rope = K_for_rope.view(B, -1, 1, self.qk_rope_dim).transpose(1, 2)
        cos_k = self.cos_cached[:, :, :S_full, :self.qk_rope_dim // 2].repeat(1, 1, 1, 2)
        sin_k = self.sin_cached[:, :, :S_full, :self.qk_rope_dim // 2].repeat(1, 1, 1, 2)
        K_for_rope = apply_rope_x(K_for_rope, cos_k, sin_k)

        # apply position encoding to each head
        K_for_rope = K_for_rope.repeat(1, self.heads, 1, 1)

        # split into multiple heads
        q_heads = torch.cat([Q, Q_for_rope], dim=-1)
        k_heads = torch.cat([K, K_for_rope], dim=-1)
        v_heads = V  # already reshaped before the split

        batch_size, head_size, n_query, dim = q_heads.shape
        n_key = k_heads.shape[2]

        if self.min_seq_len > n_query:  # 不执行
            return torch.nn.functional.scaled_dot_product_attention(
                q_heads, k_heads, v_heads,
            ).transpose(1, 2).reshape(B, S, D) @ self.W_o.T, compressed_kv

        # 1. Sorted block-diagonal via sortLSH
        _, query_sort_idx = torch.sort(
            self.lsh.hash(q_heads), dim=2, stable=True
        )  # batch_size x head_size x n
        _, key_sort_idx = torch.sort(self.lsh.hash(k_heads), dim=2, stable=True)
        query_sort_idx_inv = torch.argsort(
            query_sort_idx, dim=2, stable=True
        )  # for recovering the row order

        key_block_size = self.block_size

        query_sorted = indexing(q_heads, query_sort_idx, key_block_size)
        key_sorted = indexing(k_heads, key_sort_idx, key_block_size)
        value_sorted = indexing(v_heads, key_sort_idx, key_block_size)

        num_blocks = key_sorted.shape[2] // key_block_size
        query_block_size = query_sorted.shape[2] // num_blocks

        # Reshape tensors to [batch_size*head_size, 1, block_size, dim] as Flash-attn only allows 4d-tensors
        query_split_per_block = query_sorted.view(-1, 1, query_block_size, dim)
        key_split_per_block = key_sorted.view(-1, 1, key_block_size, dim)
        value_split_per_block = value_sorted.view(-1, 1, key_block_size, dim)

        attn_block = torch.nn.functional.scaled_dot_product_attention(
            query_split_per_block, key_split_per_block, value_split_per_block,
        ).transpose(1, 2).reshape(B, S, D) @ self.W_o.T

        lse_block = torch.logsumexp(query_split_per_block @ key_split_per_block.transpose(-1, -2) * scale, dim=-1,
                                    keepdim=True)

        if attn_block.shape[2] not in attn_block.stride():
            attn_block = attn_block.contiguous()
        attn_block = attn_block.view(
            batch_size, head_size, query_sorted.shape[2], -1
        )

        if lse_block.shape[2] not in lse_block.stride():
            lse_block = lse_block.contiguous()
        lse_block = lse_block.view(batch_size, head_size, query_sorted.shape[2], -1)

        # When inputs are padded, then unpad them 不执行
        if query_sorted.shape[2] != n_query:  # query.shape[2]:
            attn_block, lse_block = (
                attn_block[:, :, :n_query, :],
                lse_block[:, :, :n_query, :],
            )
            query_sorted = query_sorted[:, :, :n_query, :]
            key_sorted = key_sorted[:, :, :n_key, :]
            value_sorted = value_sorted[:, :, :n_key, :]

        # 2. Residual low-rank part via uniform sampling
        # Sample indices uniformly at random
        sample_size = self.sample_size
        if (
                sample_size > 0
                and (n_query > query_block_size)
                and (n_key > key_block_size)
        ):
            sampled_set = torch.randint(
                n_key,
                size=(batch_size, head_size, sample_size),
                device=query_sorted.device,
            )

            # Compute mask for hiding A_ij computed in block-diagonal attention

            weights = n_key / sample_size
            value_subset = indexing(value_sorted, sampled_set)
            key_subset = indexing(key_sorted, sampled_set)

            attn_res = (torch.nn.functional.scaled_dot_product_attention(
                query_sorted, key_subset, value_subset,
            ).transpose(1, 2).reshape(B, S, D) @ self.W_o.T).view(
                batch_size, head_size, query_sorted.shape[2], -1)

            lse_res = torch.logsumexp(query_sorted @ key_subset.transpose(-1, -2) * scale, dim=-1, keepdim=True)

            lse_res = lse_res + math.log(weights)

            # Add two attentions
            if key_block_size > 0:
                attn = add_self_attentions(
                    attn_block, lse_block, attn_res, lse_res
                )
            else:
                attn = attn_res
        else:
            attn = attn_block

        # Re-order rows with the inverse order for query_sorted -> query
        attn = indexing(attn, query_sort_idx_inv)
        return attn, compressed_kv


if __name__ == '__main__':
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 定义模型参数
    dim = 768
    num_patches = 4096
    num_heads = 8
    model = LLM_mlh_Attention(768, 768, 8).to(device).eval()

    print(model)

    # 创建输入数据
    batch_size = 1
    height = 4096
    width = 768
    # inputs = torch.ones([batch_size, num_patches, dim]).to(device)
    X = torch.randn(batch_size, height, width).to(device)
    # 1 4096 768
    # 前向传播
    with torch.no_grad():
        outputs, _ = model(X)

    print(outputs.shape)
    print(outputs)
    # summary(model, (num_heads, num_patches, dim//num_heads))
