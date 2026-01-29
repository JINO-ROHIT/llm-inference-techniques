### supplementary/reference resources

here you will find resources i used to learn/experiment different things.

1. nano vllm - https://github.com/GeeeekExplorer/nano-vllm/

2. cuda graphs - https://docs.pytorch.org/docs/2.9/notes/cuda.html
    #### to-do
        - look at partial graph capture 









### concepts to know

## 1. Inference Performance Metrics

### latency
latency measures the time elapsed from when a request is submitted until the response arrives. It can be broken down into several key components:

- **time to first token (TTFT)**  
  time from query submission to receiving the *first output token*.   
  this includes request queuing + prefill phase + network latency.  


- **time per output token (TPOT)**  
  average time between consecutive tokens during the decode (generation) phase.  
  formula: (exclude the first token)
  $$ TPOT = \frac{\text{Total Latency} - TTFT}{\text{Number of Output Tokens}} $$

- **end to end request latency**  
  total time from query submission to receiving the complete response.  
  formula:  
  $$ total = TTFT + (TPOT x total tokens generated)  $$

### throughput
throughput measures how many requests or tokens the system can process per unit of time.

- **tokens per second (TPS)**  
  - **Prefill TPS** ≈ $\frac{\text{Prompt tokens}}{TTFT}$  
  - **Decode TPS**  = $\frac{1}{TPOT}$

- **requests per second (RPS)**  
  number of complete requests or queries the system can handle per second.  
  each request counts as 1 regardless of length or complexity.  

### resource utilization

- **compute utilization**  
  percentage of time GPU / CPU is actively performing LLM related computation.

- **memory utilization**  
  VRAM (GPU) or RAM (system) usage, including:  
  - static model weights  
  - dynamic activations 
  - optimizer 
  - KV cache (very significant in long context scenarios)  
