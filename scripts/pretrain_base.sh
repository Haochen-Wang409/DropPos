python -m torch.distributed.launch --nproc_per_node=8 --nnodes 1 --node_rank 0 \
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
