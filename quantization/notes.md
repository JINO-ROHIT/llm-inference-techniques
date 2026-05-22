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

**Floating-point range:** [r_min, r_max]  
**Integer range:** [q_min, q_max]

For INT8: [−128, 127]

**Scale factor:**

$$S = \frac{q_{max} - q_{min}}{r_{max} - r_{min}}$$

**Zero point:**

$$Z = q_{min} - r_{min} \times S$$

**Quantization formula:**

q = round(r/S)

**Dequantization formula:**

r = S * (q)
	​

	​


1. `bitsandbytes` or the `int8` paper 

- https://arxiv.org/pdf/2208.07339
- https://huggingface.co/blog/hf-bitsandbytes-integration

a technique in which we do matrix mul in int8 and get almost no performance degradation.

1. from the hidden states(inputs), get the outliers(meaning something above a threshold) by column
2. these outliers, matmul them in fp16 and the others in int8. we can afford to do this since the outliers are usually ~1%.
3. dequant the non outliers(others) and add them to the outliers to get the result in fp16.


how is this technique applied? not across the whole tensor. we do something called a vector wise quantization.

if we used a single scale for the whole tensor, the performance would be quite poor since one large value is enough to distort the whole representation.

so instead we have a scale per row/per column.

2. `fp8` quantization

1. we use a RTN or round to nearest quantization scheme
2. targetting all the linear layers
3. for the weights, instead of a single scaling factor, we use a scale per channel
4. now for the activations, we scale independently for each tokens, ie dynamic quantization.