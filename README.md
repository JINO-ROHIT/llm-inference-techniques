# llm-inference-techniques

## A guide to learn and implement inference techniques from scratch

### Benchmarks

| technique | MPS (toks/s) | CUDA (toks/s) | Prefill (s) | Transfer (s) | Decode (s) | 
|-----------|--------------|---------------|-------------|--------------|------------|
| Greedy sampling | 12.68 | 39.11 | - | - | - | 
| Greedy sampling + KV cache | 33.73 | 42.68 | - | - | - |
| Prefill / Decode Disaggregation | - | 43.99 | 0.0284 | 2.0000 | 1.9323 |

Caveat: PD disaggreation should have way more gains when used the correct hardware, my implementation is a simulation


1. [paged attention](./nano-paged-attention/) - my implementation for a minimal paged attention.
2. [ORCA](./ORCA/) - my implementation for ORCA serving engine.
3. [zmq](./zmq/) - vllm uses zmq to communicate and stream requests.