# parallelism techniques for llms

there are times where the llm doesnt fit in the gpu, or maybe you have multiple gpus that you want to make use of for fatser training etc. to scale the llm across multiple gpus, we use parallelism techniques.

## data parallelism

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

## Zero Redundancy Optimizer (ZeRO)

the idea of zero is to shard the model parameters and gradients across the dp ranks, with each node only storing a slice of the items. These slices are then reconstructed when and if needed, thereby dividing memory usage by the data parallel degree Nd.

1. zero 1 - partition optimizer state

in the traditional dp, all the ranks gather the same gradients and perform identical optimizer step. this is a lot of duplicated work.

the core idea is instead of every gpu holding the full optimizer states, you partition them across the N gpus. so each gpu only holds 1/N of the optimizer states and only updates 1/N of the weights during the optimizer step.
but during the forward pass, every replica still needs the full set of parameters. so after the optimizer step, you need an all-gather to redistribute the updated weights back to all gpus.

sequence of operations for a single training step:

1. during the forward pass, each replica uses the full set of params, but runs on different micro-batches.
2. during backward pass, we use the full gradients computed locally per replica.
3. all reduce on the gradients, here each gpu ends up with only 1/N of the gradients. this replaces the all-reduce from vanilla dp.
4. each gpu runs the optimizer step on its local slice of optimizer states where it produces 1/N updated params.
5. then you do an all-gather on the params where every gpu gets the full set of updated weights back.

this way, you save tons of memory instead of holding optimizer full state , we hold only a portion of it.


2. zero 2 - partition the gradients now

in zero 1, since you only use a portion of the optimizer state, you only need a portion of the gradients, not the whole thing. 
so we dont do a all reduce, instead we do a scatter reduce on the gradients to get a portion of it. this is again extra memory savings since now we 
store only a portion of the gradients along with the optimizers.

3. zero 3 - parition the model layers as well

in zero 3, say you have 10 layers, for each forward and backward pass, you keep 1/N dp rank portion of the layer and at each step, you do an all gather
to get the entire layer, compute and then discard.

this sounds like a lot of communication overhead, but zero 3 uses overlapping where when layer N is being calculated, layer N + 1 is being prefetched, so the gpu is never idle.


despite having major savings by sharding optimizer, gradients and model parameters, you still cannot shard the activations. 

## memory usage in transformers

lets go back and see how activations impact the memory usage.

when you train an llm, you store several things in memory - 
- model weights
- model gradients
- optimizer states
- activations needed to compute the gradients

if N is the number of parameters and in fp32( 4 bytes) -
- memory for params = 4N
- memory for grad - 4N
- memory for optimizer state - (4 + 4)N

for the activation memory read here - https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=memory_for_activations
but the general idea is that it grows out of control as you increase batch size and the sequence length.

and since the activations depend on the input, its hard to shard them using the data parallelism techniques we saw above.


## tensor parallelism

ref: https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=tensor_parallelism_in_a_transformer_block

this technique shards weights, gradients, and optimizer states as well as activations.
the reason why this works is because of the math property of how you can use both row wise and column wise in matrix mul.

this way you can shard both the linear and attention head either column or row wise.

the problem is you still need the entire activations for dropout, layernorm etc because you need the whole hidden dimension.

the standard way to do this is megatron-lm style where you alternate column and row parallel linear layers back to back so the all-gather from the column parallel output feeds directly into the row parallel input. this way you only need 2 communication ops per transformer block - one all-reduce after the row parallel linear and one after the mlp.

for attention specifically, each gpu gets a subset of heads. so if you have 32 heads across 4 gpus, each gpu handles 8 heads. this works cleanly because attention heads are independent of each other. for mlp blocks, column parallel splits the weight matrix vertically so each gpu computes a slice of the intermediate hidden dim. then row parallel splits horizontally and each gpu holds a slice of the input and does a partial matmul, then you all-reduce to sum the partial results.

the communication cost is the main tradeoff. every forward and backward pass requires all-reduce calls within the tensor parallel group. this means tensor parallelism is very sensitive to interconnect bandwidth. it basically only makes sense within a single node where you have nvlink, not across nodes over infiniband because the latency is too much. the other tradeoff is that increasing tensor parallel degree shrinks the per gpu compute chunk, so at some point the matmuls become too small to efficiently utilize the gpu and you lose more from underutilization than you gain from the memory savings.

## sequence parallelism ( too complicated, come back to this later)
