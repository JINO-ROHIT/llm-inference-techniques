## Notes on quantization

read the primer on my blogs/ section - https://jino-rohit.github.io/blogs/blog.html

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