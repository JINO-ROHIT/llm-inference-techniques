from pathlib import Path
import time

from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

from qwen import Qwen3Model, QWEN3_CONFIG, KVCache
from utils import load_weights_into_qwen, Qwen3Tokenizer

import torch

model = Qwen3Model(QWEN3_CONFIG)
model.eval()

device = torch.device("mps") # i have a mac

repo_id = f"Qwen/Qwen3-0.6B-Base"
local_dir = Path(repo_id).parts[-1]
weights_file = hf_hub_download(
        repo_id = repo_id,
        filename = "model.safetensors",
        local_dir = local_dir,
    )
weights_dict = load_file(weights_file)

load_weights_into_qwen(model, QWEN3_CONFIG, weights_dict)
model.to(device)
del weights_dict

tokenizer = Qwen3Tokenizer(
        tokenizer_file_path = "Qwen3-0.6B-Base/tokenizer.json",
        repo_id = repo_id,
        apply_chat_template = False,
        add_generation_prompt = False,
        add_thinking = False
)


## basic infer loop

def greedy_infer(input_token_ids: list, max_tokens: int = 500, eos_token_id: tuple = (151645, 151643), stream = False, kv_cache = None):

    token_ids = torch.tensor(input_token_ids, device = device).unsqueeze(0)

    start_time = time.perf_counter()
    generated_tokens = 0

    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(token_ids)[:, -1]      # (1, vocab_size)
            next_token = torch.argmax(logits, dim=-1)  # (1,)

            token = next_token.item()
            if token in eos_token_id:
                break

            token_ids = torch.cat([token_ids, next_token.unsqueeze(0)], dim = -1)
            generated_tokens += 1

            if stream:
                print(tokenizer.decode([token]), end = "", flush = True)

    end_time = time.perf_counter()
    elapsed = end_time - start_time
    tokens_per_sec = generated_tokens / elapsed

    stats = {
        "generated_tokens": generated_tokens,
        "total_time": elapsed,
        "toks/s": tokens_per_sec,
    }

    return token_ids, stats

def greedy_infer_with_cache(input_token_ids: list, max_tokens: int = 500, eos_token_id: tuple = (151645, 151643), stream = False, kv_cache = None):

    token_ids = torch.tensor(input_token_ids, device = device).unsqueeze(0)

    start_time = time.perf_counter()
    generated_tokens = 0

    if kv_cache is not None:
        logits = model(token_ids, cache = kv_cache)
    
    with torch.no_grad():
        for _ in range(max_tokens):
            next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True)

            token = next_token.item()
            if token in eos_token_id:
                break

            token_ids = torch.cat([token_ids, next_token], dim=1)
            generated_tokens += 1

            if stream:
                print(tokenizer.decode([token]), end = "", flush = True)

            # feed only the new token to the model; cache handles history
            logits = model(next_token, cache = cache)

    end_time = time.perf_counter()
    elapsed = end_time - start_time
    tokens_per_sec = generated_tokens / elapsed

    stats = {
        "generated_tokens": generated_tokens,
        "total_time": elapsed,
        "toks/s": tokens_per_sec,
    }

    return token_ids, stats

max_tokens = 500
eos_token_id = [151645, 151643]

prompt = "Give me a short introduction to large language models."

input_token_ids = tokenizer.encode(prompt)
text = tokenizer.decode(input_token_ids)
print(f"Query: {text}")

cache = KVCache(n_layers=model.cfg["n_layers"])
_, stats = greedy_infer(input_token_ids = input_token_ids, stream = True)
print("\n")
print(f"stats without kv cache")
print(stats)


_, stats = greedy_infer_with_cache(input_token_ids = input_token_ids, stream = True, kv_cache = cache)
print("\n")
print(f"stats with kv cache")
print(stats)
