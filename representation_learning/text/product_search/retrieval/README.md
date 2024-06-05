# Train

```bash
python3.10 -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --master_port=12345 train.py
python3.10 -m torch.distributed.run --nproc_per_node=1 --nnodes=1 --master_port=12345 train.py
 ```

## Plots

![image](../assets/accuracy_curves.png)