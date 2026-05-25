## Notes on quantization

read the primer on my blogs/ section - https://jino-rohit.github.io/blogs/blog.html


a nice mental model for quantization

quantization solves essentially a data movement problem. the problem is doing arithmetic is insanely quicker compared to moving data from one place to another.

why do we have to move things then?

its because the SRAM/registers are quite small compared to the HBM, where we hold all the weights of the llm. during each token generation, we carry a pice of data from HBM to SRAM, compute and then move back again. this movement is actually what makes the overall system slower not the computation itself.


when you quantize from a larger fp16 to say fp8, this doesnt mean computation is faster, it means we have lesser data to transfer and hence the overall system is way faster now.


let me illustrate with an example. 

say you have a A100 with -
- memory bandwith of 2000 GB/s
- computing power of 312 TFLOPS

you want to generate from a model with 100 B parameters.

data transfer

for fp16, we need 16 bits or 2 bytes for each param. so total data volume = 100 x 10^9 x 2 = 200 GB
for fp8, we need 1 byte, so total data volume = 100 GB


actual computation( im going to simplify this to just be matrix multiplication that does multiplications and additions)

total computation load = 100 x 10^9 x 2 = 200 GFLOPS


now, how long does it take to generate one token in an A100?

the fp16 case

load data = 200 GB / 2000 GB/s = 0.1 s
to compute = 200 GLOPS / 312 TFLOPS = 0.0006 s

look at that, data transfer is over 100x slower than the actual computation itself.



now the fp8 case

load data = 100 GB / 2000 GB/s = 0.05 s
computations ~= 0.0006 s ( maybe a bit faster than fp16)

but look at this now, yes computation might be a bit faster but indeed loading data has now become signicantly faster!

this is in principle what all quantization algorithms try to do.



1. symmetric quantization (ref: https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization)

in symmetric quantization, the range of the original floating-point values is mapped to a symmetric range around zero in the quantized space. this means zero will
always be zero. its also called absmax quantization.


lets say we want to convert fp16 to int8 for the numbers

```
5.47 8.21 3.145 10.94
```

the max value in this range is 10.94
the max value of the int8 range is 127

scale factor = 10.94/127 = 0.086

now we quant the first value 5.47 to become = round(value / scale)
                                            = round(5.47 / 0.086)
                                            = round(63.60)
                                            = 64

to perform dequant, it becomes = scale * value
                               = 0.086 * 64
                               = 5.50

the error factor here is (5.50 - 5.47) = 0.03


2. Asymmetric quantization

In asymmetric quantization, the minimum and maximum values from the floating-point range map directly to the minimum and maximum values of the quantized range (not centered at zero).

floating-point range = [r_min, r_max]  
integer range = [q_min, q_max]

for INT8: [−128, 127]

scale factor

$$S = \frac{q_{max} - q_{min}}{r_{max} - r_{min}}$$

zero point

$$Z = q_{min} - r_{min} \times S$$

quantization formula = q = round(r/S)

dequantization formula = r = S * (q)
	​



1. `bitsandbytes` or the `int8` paper 

- https://arxiv.org/pdf/2208.07339
- https://huggingface.co/blog/hf-bitsandbytes-integration


in this technique, we do matrix multiplication in int8 wotj almost no performance degradation.

the main algorithm -
1. from the hidden states(inputs), get the outliers(meaning something above a threshold) by column.
2. matmul the outliers in fp16 and the others in int8. we can afford to do this since the outliers are usually ~1%.
3. dequantize the non-outlier results and add them with the fp16 outlier results to produce the final fp16 output.

instead of using a single scale for the entire tensor (which performs badly because one large value can distort the whole representation), bnb uses vector-wise quantization. that means each row / column gets its own scale, allowing much better preservation of information while still getting the efficiency benefits of int8.


2. `awq` quantization

paper - https://arxiv.org/pdf/2306.00978

AWQ is a method for low-bit quantization of model weights. using this method, model weights can be quantized to 4 bits and then dequantized to FP16 (W4A16) during activation calculation. also, weights can be quantized to 3 bits or 8 bits based on AWQ, and then calculated using 4 bits, 8 bits, or 16 bits, leading to methods such as W4A4 and W4A8.

but the authors say W4A16 gives minimal loss of accuracy, so its most preferred.


there are three major points to understanding awq quantization - 

1. not all weights are equally important, only a small portion matter a lot

the authors show that only a very small subset of weights (around 0.1%–1%) have a huge impact on model accuracy. most weights can actually tolerate aggressive low-bit quantization without hurting the model too much.

this leads to an important idea -

if we can somehow protect only the important weights while quantizing the rest to int4/int3, we can drastically reduce memory usage and improve inference speed while keeping accuracy almost unchanged.


how do we identify these important (salient) weights?


1. random selection

randomly choose 0.1%–1% of weights and keep them in fp16.
this performs badly and is almost identical to naive round-to-nearest (rtn) quantization.


2. selection based on weight magnitude

here, weights are first sorted and the ones with the largest absolute values are treated as important.
surprisingly, the paper finds this performs almost as badly as random selection.
this is a pretty important observation because many older quantization methods assume:
larger weights = more important weights
but awq shows this is often not true for llms.

3. selection based on activation distribution


instead of looking at weight values themselves, the paper looks at activation magnitudes.

here, "activation" means the input tensor multiplied with the weight matrix during matmul.

for a linear layer - y = Wx

the activation is x and not the output.


the paper finds that selecting important weights based on activation statistics works dramatically better than selecting based on weight magnitude.
in fact, preserving only 0.1%–1% of activation-important weights in fp16 gives accuracy very close to full fp16 inference.


they also use channel-level instead of element-level where -

each input channel (column of the weight matrix) is treated as a unit
activation magnitudes are averaged per channel
channels with the largest activation magnitudes are considered salient

the problem with this approach
if some elements in the weight matrix are stored in FP16 format and others in INT4 format, both storage and retrieval becomes complicated and slow.
therefore, they propose scaling.

2. amplifying significant weights during quantization can reduce quantization error.

check the paper bit for this mathy equations

3. an algorithm to calculate the scaling factors


the goal is to minimize the difference between:
the original fp16 layer output
the quantized layer output after scaling

![alt text](image.png)

directly optimizing this equation is hard for two reasons.

1. quantization is non-differentiable

quantization contains a rounding operation round(soemthing) which is non-differentiable.
small changes in weights can suddenly jump to different integer buckets, making gradient-based optimization unstable.

2. the optimization problem is non-convex


awq simplifies this by observing channels with larger activations are usually more important.

so important channels should receive larger scaling factors and unimportant channels should receive smaller scaling factors
so instead of optimizing every scaling factor independently, awq derives scaling directly from activation statistics.

activation-based scaling

for each input channel:

- compute the average activation magnitude
- use this as the base importance score

the scaling factor becomes:

s = sX ^ alpha
sX is the average activation magnitude per channel
alpha ∈ [0,1] controls scaling strength

now we only need to search for alpha. this is done by taking 20 numbers on average in the interval [0,1], such as 0, 0.05, 0.10, 0.15, etc., and then calculates the different values ​​for each number. the optimal MSE loss is the one that minimizes the loss under alpha.


`group_size` is the only hyper param.

- a larger group size results in more weights per group and fewer quantization parameters, which may reduce the accuracy of the quantized model, but also lowers computational and storage costs.
- a small group size has fewer weights per group and more quantization parameters, which may result in higher model accuracy, but also increases computational and storage costs.
- the standard default value is 128.


quoting from this - https://github.com/vllm-project/llm-compressor/issues/1522

we found that using randomly generated tokens is good enough for smaller models (e.g. 8B scale), whereas for larger ones (e.g. 70B and 405B), we need to use a proper dataset to get an accurate quantized model. This aligns well with the intuition from above: during quantization, we want to trigger outliers/activations to properly capture their behavior.