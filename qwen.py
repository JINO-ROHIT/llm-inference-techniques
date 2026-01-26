"""a lot of the code is inspired from sebastian rascka here - https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/11_qwen3/standalone-qwen3.ipynb"""


import torch
import torch.nn as nn

torch.manual_seed(42)

QWEN3_CONFIG = {
    "vocab_size": 151_936,    
    "context_length": 40_960,    
    "emb_dim": 1024,             
    "n_heads": 16,                
    "n_layers": 28,             
    "hidden_dim": 3072,   
    "head_dim": 128,               
    "qk_norm": True,  # whether to normalize queries and keys in GQA
    "n_kv_groups": 8,            
    "rope_base": 1_000_000.0,        
    "dtype": torch.bfloat16,     
}


def compute_rope_params(head_dim: int, theta_base: int = 10_000, context_length: int = 4096, dtype: torch.dtype = torch.float32):
    inv_freq = 1.0 / theta_base ** (torch.arange(0, head_dim, 2, dtype = dtype) / head_dim)
    positions = torch.arange(0, context_length, dtype = dtype)

    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0) # [ctx_len, 1] * [1, head_dim / 2]
    angles = torch.cat([angles, angles], dim = 1) # match to head dim

    sin_angle = torch.sin(angles)
    cos_angle = torch.cos(angles)
    return sin_angle, cos_angle


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, offset: int = 0):
    bs, n_head, seq_len, head_dim = x.shape
    x1 = x[..., : head_dim // 2] # ... better than use :, :, :, :head_dim //2
    x2 = x[..., head_dim // 2 :]

    cos = cos[offset: offset + seq_len, :].unsqueeze(0).unsqueeze(0) # (1, 1, seq_len, head_dim)
    sin = sin[offset: offset + seq_len, :].unsqueeze(0).unsqueeze(0)    

    rotated = torch.cat((-x2, x1), dim = -1)
    x_rotated = (x * cos) + (rotated * sin)
    return x_rotated.to(dtype = x.dtype) # okay to use lower precision

class RMSNorm(nn.Module):
    def __init__(self, emb_dim: int, eps: float = 1e-6, bias: bool =False):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x):
        input_dtype = x.dtype

        x = x.to(torch.float32) #upscale

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + self.eps)
        norm_x = norm_x * self.scale

        if self.shift is not None:
            norm_x = norm_x + self.shift

        return norm_x.to(input_dtype)

class FFN(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.gate_proj = nn.Linear(config["emb_dim"], config["hidden_dim"], dtype=config["dtype"], bias=False)
        self.up_proj = nn.Linear(config["emb_dim"], config["hidden_dim"], dtype=config["dtype"], bias=False)
        self.down_proj = nn.Linear(config["hidden_dim"], config["emb_dim"], dtype=config["dtype"], bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        gate = nn.functional.silu(gate)
        up = self.up_proj(x)
        return self.down_proj(gate * up)

class KVCache:
    def __init__(self, n_layers):
        self.cache = [None] * n_layers

    def get(self, layer_idx):
        return self.cache[layer_idx]

    def update(self, layer_idx, value):
        self.cache[layer_idx] = value

    def get_all(self):
        return self.cache

    def reset(self):
        for i in range(len(self.cache)):
            self.cache[i] = None

class GroupedQueryAttention(nn.Module):
    def __init__(
        self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False, dtype=None
    ):
        super().__init__()
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        if head_dim is None:
            assert d_in % num_heads == 0, "`d_in` must be divisible by `num_heads` if `head_dim` is not set"
            head_dim = d_in // num_heads

        self.head_dim = head_dim
        self.d_out = num_heads * head_dim

        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)

        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

    def forward(self, x, mask, cos, sin, start_pos = 0, cache = None):
        b, num_tokens, _ = x.shape

        queries = self.W_query(x)  # (b, num_tokens, num_heads * head_dim)
        keys = self.W_key(x)       # (b, num_tokens, num_kv_groups * head_dim)
        values = self.W_value(x)   # (b, num_tokens, num_kv_groups * head_dim)

        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys = self.k_norm(keys)

        queries = apply_rope(queries, cos, sin, offset = start_pos)
        keys = apply_rope(keys, cos, sin, offset = start_pos)

        #kv cache start
        if cache is not None:
            prev_k, prev_v = cache
            keys = torch.cat([prev_k, keys], dim = 2)
            values = torch.cat([prev_v, values], dim = 2)
        else:
            start_pos = 0
            keys, values = keys, values
        
        next_cache = (keys, values)
        # Expand K and V to match number of heads
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        # Attention
        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)

        context = (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)
        return self.out_proj(context), next_cache


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            head_dim=cfg["head_dim"],
            num_kv_groups=cfg["n_kv_groups"],
            qk_norm=cfg["qk_norm"],
            dtype=cfg["dtype"]
        )
        self.ff = FFN(cfg)
        self.norm1 = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.norm2 = RMSNorm(cfg["emb_dim"], eps=1e-6)

    def forward(self, x, mask, cos, sin, start_pos = 0, cache = None):
        shortcut = x
        x = self.norm1(x)
        #x = self.att(x, mask, cos, sin)  # Shape [batch_size, num_tokens, emb_size]
        x, next_cache = self.att(x, mask, cos, sin, start_pos = start_pos, cache = cache)
        x = x + shortcut 

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut 

        return x, next_cache


class Qwen3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        self.trf_blocks = nn.ModuleList(  
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        if cfg["head_dim"] is None:
            head_dim = cfg["emb_dim"] // cfg["n_heads"]
        else:
            head_dim = cfg["head_dim"]
        sin, cos = compute_rope_params(
            head_dim=head_dim,
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"]
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cfg = cfg
        self.current_pos = 0  # track current position in KV cache


    def forward(self, in_idx, cache = None):
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds

        num_tokens = x.shape[1]

        if cache is not None:
            pos_start = self.current_pos
            pos_end = pos_start + num_tokens
            mask = torch.triu(
                torch.ones(pos_end, pos_end, device = x.device, dtype = torch.bool), diagonal = 1)[pos_start:pos_end, :pos_end]
            self.current_pos = pos_end
        else:
            pos_start = 0 
            mask = torch.triu(torch.ones(num_tokens, num_tokens, device = x.device, dtype = torch.bool), diagonal = 1)
        
        # (1, 1, num_tokens, num_tokens) to broadcast across batch and heads
        mask = mask[None, None, :, :]

        for i, block in enumerate(self.trf_blocks):
            blk_cache = cache.get(i) if cache else None
            x, new_blk_cache = block(x, mask, self.cos, self.sin,
                                     start_pos = pos_start,
                                     cache = blk_cache)
            if cache is not None:
                cache.update(i, new_blk_cache)

        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg["dtype"]))
        return logits

    def reset_kv_cache(self):
        self.current_pos = 0