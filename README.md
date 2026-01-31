# llm-inference-techniques

## A guide to learn and implement inference techniques from scratch

### Benchmarks

| technique | MPS (toks/s) | CUDA (toks/s) | Prefill (s) | Transfer (s) | Decode (s) | 
|-----------|--------------|---------------|-------------|--------------|------------|
| Greedy sampling | 12.68 | 39.11 | - | - | - | 
| Greedy sampling + KV cache | 33.73 | 42.68 | - | - | - |
| Prefill / Decode Disaggregation | - | 43.99 | 0.0284 | 2.0000 | 1.9323 |

Caveat: PD disaggreation should have way more gains when used the correct hardware, my implementation is a simulation

#### TO-DO

1. switch to a larger model to try speculative decoding
2. decouple kv outside the model