## torch distributed

`--nnodes` - this is the number of machines/servers you have
`--nproc_per_node` - this is the number of gpus you have within the machine.

### meta device

this is an abstract device that records metadata but no data. this means you dont need to load tensors on cpu/gpu but check transofrmations, analysis on the
tensors etc without actually spending time on loading stuff, no OOMs etc.

### process group

the main crux of doing distributed training is a way for processes to find and talk to each other. you do this using process group.

also let say we have 4 gpus, we need gpu 1 and 3 to talk, and gpu 2 and 4 to talk to each other, and not with other. process groups help you do this.

### device mesh

a deviceMesh is essentially a structured way to create and manage many process groups. as you scale more and more gpus, using process groups alone gets quite complicated.

### dtensor

the native tensor type used for distributed training.

you can shard, replicate and partial ops.


add examples for all.
