#### quantization for llms

model quantization is a compression technique to represent high precision values in lower precision to reduce memory consumption and improve speed.

example: convert 32 bit model weights into 4 bits for inference

we can do a small calculation to check memory improvements

remember:
- 1 byte = 8 bits  
- 1 KiB = 1024 bytes  
- 1 MiB = 1024 KiB  
- 1 GiB = 1024 MiB  

```
1 GB = (1024)^3 B
```

1. take llama 70b in 32 bits precision

number of params = 70 billion
precision = 32 bits = 4 bytes

fp32 model size = 280 * 10^9 / 1024^3
     ~= 260.7 GB


2. now to 4 bits preciion

number of params = 70 billion
precision = 4 bits = 0.5 byte

int4 = 70 * 0.5 * 10^9 / 1024^3
     ~= 32.59 GB


```
memory improvements = 260.7 / 32.59 ~= 8x reduction
```