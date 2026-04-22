### collective operations in nccl

this is a collection of the main collective operations in nccl that i used for practice.

1. `gather.py` - gather says if you have N gpus each having different data, all the data from each gpu gets sent to one single master gpu.
2. `all_gather.py` - here, instead of a single gpu having all the data copy, all the gpus have all the copies of each other.