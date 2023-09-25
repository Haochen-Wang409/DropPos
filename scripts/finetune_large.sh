python -m torch.distributed.launch --nproc_per_node=8 --nnodes 1 --node_rank 0 \
    main_finetune.py \
    --batch_size 64 \
    --accum_iter 1 \
    --model vit_large_patch16 \
    --finetune /path/to/checkpoint \
    \
    --epochs 50 \
    --warmup_epochs 5 \
    --blr 1e-3 --layer_decay 0.75 --weight_decay 0.05 \
    --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval \
    --data_path /path/to/imagenet \
    --nb_classes 1000 \
    --output_dir  ./output_dir \
    --log_dir   ./log_dir \
    --experiment droppos_vit_large_patch16_in1k_ep800
