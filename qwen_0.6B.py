"""a lot of the code is inspired from sebastian rascka here - https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/11_qwen3/standalone-qwen3.ipynb"""

import torch
import torch.nn as nn


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


class FNN:
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config["emb_dim"], config["hidden_dim"], dtype=config["dtype"], bias=False)
        self.up_proj = nn.Linear(config["emb_dim"], config["hidden_dim"], dtype=config["dtype"], bias=False)
        self.down_proj = nn.Linear(config["hidden_dim"], config["emb_dim"], dtype=config["dtype"], bias=False)
    
    def forward(self, x: torch.Tensor):
        gate = self.gate_proj(x)
        gate = nn.functional.silu(gate)
        up = self.up_proj(x)
        return self.down_proj(gate * up)
