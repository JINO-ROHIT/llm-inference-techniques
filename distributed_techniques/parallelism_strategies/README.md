### parallelism techniques for llms

there are times where the llm doesnt fit in the gpu, or maybe you have multiple gpus that you want to make use of for fatser training etc. to scale the llm across multiple gpus, we use parallelism techniques.

1. data parallelism

the idea is if you have 5 gpus -
1. you make copies of the llm in all 5 gpus.
2. you divide the data into micro batches and run them on the 5 gpus in parallel.
3. each gpu computes its own forward and backward pass independently.
4. now to get an updated final state of the llm, you do a nccl all-reduce operation on the gradients so now every gpu has the same averaged gradient, and each gpu updates its own weights identically.

a naive way is to wait for all the backward pass to finish and then reduce, but this makes the gpu idle and we dont want it.

we need to overlap compute and communication. there are 2 main techniques -


1. overlap the communication with backward pass - so the idea is as soon as the final layer backward pass is done, you can apply reduce on the gradient, while the earlier layers continue their backward pass. since backprop goes from last layer to first, you can pipeline this nicely
2. bucketing gradient - instead of running a communication for each step of the gradient, we can group them and then apply. this saves on communication cost.


data parallelism has its limits though. beyond a certain dp rank, throughput drops due to communication overhead scaling with the number of gpus. also, this whole approach assumes the model fits on a single gpu. if it doesn't, dp alone can't help you.

dp simply improves throughput by parallelizing many batches of data across gpus, the model itself isn't split at all.