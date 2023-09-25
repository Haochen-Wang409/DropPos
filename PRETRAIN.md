## Pre-training DropPos

A typical command to pre-train ViT-B/16 with **multi-node distributed training**. 
For instance, run the following command to pre-train DropPos with ViT-Base with 32 GPUs (4 nodes x 8 GPUs):
```bash
python -m torch.distributed.launch --nproc_per_node=8 \
    --nnodes 4 --node_rank 0 --master_port 12320 --master_addr=$ip_node_0 \
    main_pretrain.py \
    --batch_size 64 \
    --accum_iter 1 \
    --model DropPos_mae_vit_base_patch16_dec512d2b \
    \
    --drop_pos_type mae_pos_target \
    --mask_token_type param \
    --pos_mask_ratio 0.75 \
    --pos_weight 0.05 \
    --label_smoothing_sigma 1 \
    --sigma_decay \
    --attn_guide \
    \
    --input_size 224 \
    --token_size 14 \
    --mask_ratio 0.75 \
    --epochs 800 \
    --warmup_epochs 10 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path /path/to/imagenet \
    --output_dir  ./output_dir \
    --log_dir   ./log_dir \
    --experiment droppos_pos_mask0.75_posmask0.75_smooth1to0_sim_in1k_ep800
```
on the first node. 
On other nodes, run the same command with ```--node_rank 1```, ```--node_rank 2```, and ```--node_rank 3``` respectively. 
```--master_addr``` is set as the ip of the node 0.

Please modify the ```/path/to/imagenet/``` to your ```<data_path>```.
You can also move the txt files **IN1K/train.txt** and **IN1K/val.txt** to your imagenet root path.
Please find these files [here](https://github.com/implus/UM-MAE/tree/main/IN1K).

More scripts can be found in [scripts](https://github.com/Haochen-Wang409/DropPos/tree/main/scripts).
