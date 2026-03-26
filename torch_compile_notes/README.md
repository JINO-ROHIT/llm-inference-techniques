### how to use torch compile

ref - https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html

torch compile is a method to speed up pytorch code after 2.0 using JIT compilation and requires almost litle to no change.
any python fn or pytorch module can be passed and will be replaced by the optimized one.

torch.compile takes extra time to compile the model on the first few executions. torch.compile re-uses compiled code whever possible, so if we run our optimized model several more times, we should see a significant improvement compared to eager. check ex 03_speedup.py


#### graph breaks

The graph break is one of the most fundamental concepts within torch.compile. It allows torch.compile to handle arbitrary Python code by interrupting compilation, running the unsupported code, then resuming compilation. The term “graph break” comes from the fact that torch.compile attempts to capture and optimize the PyTorch operation graph. When unsupported Python code is encountered, then this graph must be “broken”. Graph breaks result in lost optimization opportunities, which may still be undesirable, but this is better than silent incorrectness or a hard crash.