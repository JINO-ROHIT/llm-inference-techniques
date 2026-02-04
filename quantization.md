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


#### comparison of major numerical formats

| Format | Total Bits | Sign Bit (S) | Exponent (E) | Mantissa / Fraction (M) | Numerical Range (Approx.) | Memory Usage (Relative to FP32) | Primary Uses and Characteristics |
|-------|------------|--------------|--------------|--------------------------|----------------------------|----------------------------------|---------------------------------|
| FP32  | 32 bits    | 1 bit        | 8 bits       | 23 bits                  | ±3.4 × 10³⁸               | 1x (baseline)                   | high precision, baseline for training and inference |
| FP16  | 16 bits    | 1 bit        | 5 bits       | 10 bits                  | ±6.5 × 10⁴                | 50%                             | faster training and inference, limited dynamic range, usually needs loss scaling |
| BF16  | 16 bits    | 1 bit        | 8 bits       | 7 bits                   | ±3.4 × 10³⁸               | 50%                             | used in bf16/fp32 mixed training, wide dynamic range |
| FP8   | 8 bits     | 1 bit        | 4–5 bits*   | 2–3 bits*                | ~10²–10³*                 | 25%                             | aggressive low-precision training and inference, hardware dependent, needs careful scaling |
| INT8  | 8 bits     | 1 bit        | none         | 7 bits       | -128 to 127               | 25%                             | inference acceleration with a good balance of speed and precision |
| INT4  | 4 bits     | 1 bit        | none         | 3 bits      | -8 to 7                   | 12.5%                           | extreme compression, requires complex algorithms to work well |

