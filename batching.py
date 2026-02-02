from collections import deque
import time
from abc import ABC, abstractmethod

MAX_BATCH_SIZE = 64
MAX_TOKENS_PER_SEQ = 512 
TOKENS_PER_STEP = 1         

class Request:
    def __init__(self, seq_id: int, input_len: int = 128):
        self.seq_id = seq_id
        self.input_len = input_len         
        self.generated_tokens = 0   
        self.completed = False
    
    @property
    def is_completed(self):
        return self.completed or self.generated_tokens >= MAX_TOKENS_PER_SEQ
    
    def generate_one_token(self):
        # this should actually be the decode step in the llm
        if self.is_completed:
            return False
        self.generated_tokens += TOKENS_PER_STEP
        if self.generated_tokens >= MAX_TOKENS_PER_SEQ:
            self.completed = True
        return True


class Batching(ABC):
    def __init__(self):
        self.queue = deque()
        self.active_batch = []
    
    def add_request(self, request):
        self.queue.appendleft(request)   # FIFO
    
    @abstractmethod
    def _get_next_batch(self):
        pass

    def step(self):
        batch = self._get_next_batch()
        if not batch:
            return 0
        
        tokens_generated = 0
        still_active = []
        
        for req in batch:
            if req.generate_one_token():
                tokens_generated += 1
                still_active.append(req)
        
        self.active_batch = still_active
        return tokens_generated


class StaticBatching(Batching):
    def _get_next_batch(self):
        # only refill when current batch is completely finished
        self.active_batch = [r for r in self.active_batch if not r.is_completed]
        
        if not self.active_batch and self.queue:
            while self.queue and len(self.active_batch) < MAX_BATCH_SIZE:
                self.active_batch.append(self.queue.popleft())
        
        return self.active_batch


class ContinuousBatching(Batching):
    def _get_next_batch(self):
        # keep filling the batch every step
        self.active_batch = [r for r in self.active_batch if not r.is_completed]
        
        while self.queue and len(self.active_batch) < MAX_BATCH_SIZE:
            self.active_batch.append(self.queue.popleft())
        
        return self.active_batch


def run_benchmark(batcher_class, num_requests: int, verbose: bool = False):
    batcher = batcher_class()
    
    for i in range(num_requests):
        input_len = 64 + (i % 200)   # 64–263 tokens prompt
        req = Request(seq_id = i, input_len = input_len)
        batcher.add_request(req)
    
    start_time = time.time()
    total_tokens_generated = 0
    steps = 0
    
    while True:
        tokens_this_step = batcher.step()
        if tokens_this_step == 0:
            break
        total_tokens_generated += tokens_this_step
        steps += 1
        
        if verbose and steps % 50 == 0:
            print(f"Step {steps:4d} | Active: {len(batcher.active_batch):3d} | "
                  f"Tokens/s: {total_tokens_generated/(time.time()-start_time):.1f}")
    
    duration = time.time() - start_time
    throughput = total_tokens_generated / duration if duration > 0 else 0
    
    return {
        "duration_s": duration,
        "total_tokens": total_tokens_generated,
        "throughput_tokens_s": throughput,
        "steps": steps,
        "avg_batch_size": total_tokens_generated / steps if steps > 0 else 0
    }


if __name__ == "__main__":
    N_REQUESTS = 200
    
    print(f"\nbenchmarking with {N_REQUESTS} requests "
          f"(max batch = {MAX_BATCH_SIZE}, max tokens/seq = {MAX_TOKENS_PER_SEQ})\n")
    
    static = run_benchmark(StaticBatching, N_REQUESTS)
    cont   = run_benchmark(ContinuousBatching, N_REQUESTS)

    def _print_results(name, res):
        print(f"{name}:")
        print(f"  duration:            {res['duration_s']:.4f} s")
        print(f"  total output tokens: {res['total_tokens']:,}")
        print(f"  throughput:          {res['throughput_tokens_s']:.4f} tok/s")
        print(f"  average batch size:  {res['avg_batch_size']:.1f}")
        print()

    _print_results("static batching", static)
    _print_results("continuous batching", cont)

    speedup = (cont['throughput_tokens_s'] / static['throughput_tokens_s'] - 1) if static['throughput_tokens_s'] > 0 else None
    print(f"continuous batching is {speedup*100:.0f}% faster")