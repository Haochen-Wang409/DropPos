## Fine-tuning DropPos

A typical command to fine-tune ViT-B/16 with **multi-node distributed training**. 
For instance, run the following command to fine-tune DropPos with ViT-Base with 32 GPUs (4 nodes x 8 GPUs):
```bash
python -m torch.distributed.launch --nproc_per_node=8 \
    --nnodes 4 --node_rank 0 --master_port 12320 --master_addr=$ip_node_0 \
    main_finetune.py \
    --batch_size 32 \
    --accum_iter 1 \
    --model vit_base_patch16 \
    --finetune /path/to/checkpoint \
    \
    --epochs 100 \
    --warmup_epochs 5 \
    --blr 1e-3 --layer_decay 0.75 --weight_decay 0.05 \
    --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval \
    --data_path /path/to/imagenet \
    --nb_classes 1000 \
    --output_dir  ./output_dir \
    --log_dir   ./log_dir \
    --experiment droppos_vit_base_patch16_in1k_ep800
```
on the first node. 
On other nodes, run the same command with ```--node_rank 1```, ```--node_rank 2```, and ```--node_rank 3``` respectively. 
```--master_addr``` is set as the ip of the node 0.

Please modify ```/path/to/imagenet``` to your ```<data_path>````.
You can also move the txt files **IN1K/train.txt** and **IN1K/val.txt** to your imagenet root path.
Please find these files [here](https://github.com/implus/UM-MAE/tree/main/IN1K).

More scripts can be found in [scripts](https://github.com/Haochen-Wang409/DropPos/tree/main/scripts).
