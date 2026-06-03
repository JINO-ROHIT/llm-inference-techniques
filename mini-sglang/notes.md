my personal notes on the mini-sglang inference server

1. __main__.py - the module executed when you run python -m <package>. it's a special filename Python recognizes as the entry point for a package run with -m. this enables you to run "python -m minisgl".

2. `launch.py` - this starts to spin 1 process for scheduler, N tokenizers and 1 detokenzier.

3. 

```
for _ in range(num_tokenizers + 2):
            logger.info(ack_queue.get())
```

this is a blocking mp.Queue.get(). the main process waits here until each worker puts a string into the queue (e.g. "Scheduler is ready" from the primary scheduler, plus readiness messages from tokenizer/detokenizer workers). you can check inside the function where it says ack_queue.put("message")

4. `tokenizer/` class is straightf forward for both tokenize and detokenize.

5. `layers/` linear and embedding layer have tp 

6. `distributed/` implementations for all reduce and all gather

7. 