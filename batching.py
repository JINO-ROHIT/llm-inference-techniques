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
