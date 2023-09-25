# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# MAE: https://github.com/facebookresearch/mae
# SimMIM: https://github.com/microsoft/SimMIM
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import builtins

import PIL
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets.folder import default_loader

import timm

assert timm.__version__ == "0.3.2"  
import timm.optim.optim_factory as optim_factory
from timm.utils import ModelEma

from util import utils
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.datasets import ImageListFolder
from util.pos_embed import interpolate_pos_embed

from engine_pretrain import train_one_epoch
from mask_transform import MaskTransform

import models_mae
import models_DropPos_mae


def get_args_parser():
    parser = argparse.ArgumentParser('DropPos pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--token_size', default=int(224 / 16), type=int,
                        help='number of patch (in one dimension), usually input_size//16')
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--norm_pix_loss', type=utils.bool_flag, default=True,
                        help='Use (per-patch) normalized pixels as targets for computing loss')

    # Mask generator (UM-MAE)
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--mask_block', action='store_true',
                        help='Block sampling for supporting pyramid-based vits')
    parser.set_defaults(mask_block=False)

    # DropPos parameters
    parser.add_argument('--drop_pos_type', type=str,
                        choices=['vanilla_mae',         # original MAE
                                 'mae_pos_target',      # DropPos with patch masking
                                 'multi_task'],         # DropPos with an auxiliary MAE loss
                        default='mae_pos_target')
    parser.add_argument('--mask_token_type', type=str,
                        choices=['param',       # learnable parameters
                                 'zeros',       # zeros
                                 'wrong_pos'],  # random wrong positions
                        default='param')
    parser.add_argument('--pos_mask_ratio', default=0.75, type=float,
                        help='Masking ratio of position embeddings.')
    parser.add_argument('--shuffle', action='store_true',
                        help='Shuffle before forward to encoder.')
    parser.set_defaults(shuffle=False)
    parser.add_argument('--pos_weight', type=float, default=1.,
                        help='Loss weight for position prediction when multi-task.')
    parser.add_argument('--label_smoothing_sigma', type=float, default=0.,
                        help='Label smoothing parameter for position prediction, 0 means no smoothing.')
    parser.add_argument('--sigma_decay', action='store_true',
                        help='Decay label smoothing sigma during training (linearly to 0).')
    parser.set_defaults(sigma_decay=False)
    parser.add_argument('--conf_ignore', action='store_true',
                        help='Ignore confident patches when computing objective.')
    parser.set_defaults(conf_ignore=False)
    parser.add_argument('--attn_guide', action='store_true',
                        help='Attention-guided loss weight.')
    parser.set_defaults(attn_guide=False)

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=False, help="""Whether or not
            to use half precision for training. Improves training time and memory requirements,
            but can provoke instability and slight decay of performance. We recommend disabling
            mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=5., help="""Maximal parameter
            gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
            help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR, 40 for MAE and 10 for SimMIM')

    # Dataset parameters
    parser.add_argument('--data_path', default='/path/to/imagenet/', type=str,
                        help='dataset path')

    # Misc parameters
    parser.add_argument('--output_dir', default='./output_dir', help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir', help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--load_from', default='', help='load pretrained checkpoint model')
    parser.add_argument('--experiment', default='exp', type=str, help='experiment name (for log)')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # simple augmentation
    transform_train = MaskTransform(args)

    # build dataset
    dataset_train = ImageListFolder(os.path.join(args.data_path, 'train'), transform=transform_train,
                                    ann_file=os.path.join(args.data_path, 'train.txt'))
    print(dataset_train)
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    if global_rank == 0 and args.log_dir is not None:
        log_dir = os.path.join(args.log_dir, f"{args.model}_{args.experiment}")
        os.makedirs(log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=log_dir)
    else:
        log_writer = None

    # define the model
    if args.drop_pos_type == 'vanilla_mae':
        model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
    elif args.drop_pos_type == 'mae_pos_target':
        model = models_DropPos_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss,
                                                        mask_token_type=args.mask_token_type,
                                                        shuffle=args.shuffle,
                                                        multi_task=False,
                                                        conf_ignore=args.conf_ignore,
                                                        attn_guide=args.attn_guide)
    elif args.drop_pos_type == 'multi_task':
        model = models_DropPos_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss,
                                                        mask_token_type=args.mask_token_type,
                                                        shuffle=args.shuffle,
                                                        multi_task=True,
                                                        conf_ignore=args.conf_ignore,
                                                        attn_guide=args.attn_guide)
    else:
        raise Exception('unknown model type: {}'.format(args.drop_pos_type))

    model.cuda()
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    if args.finetune:
        # load pretrained model
        checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        if 'state_dict' in checkpoint:
            checkpoint_model = checkpoint['state_dict']
        else:
            checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        checkpoint_model = {k.replace("module.", ""): v for k, v in checkpoint_model.items()}

        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print('missing keys:', msg.missing_keys)
        print('unexpected keys:', msg.unexpected_keys)

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))

    print(optimizer)
    # loss_scaler = NativeScaler()
    loss_scaler = torch.cuda.amp.GradScaler() if args.use_fp16 else None

    ckpt_path = os.path.join(args.output_dir, f"{args.model}_{args.experiment}_temp.pth")
    if not os.path.isfile(ckpt_path):
        print("Checkpoint not founded in {}, train from random initialization".format(ckpt_path))
    else:
        print("Found checkpoint at {}".format(ckpt_path))
        misc.load_model(args=args, ckpt_path=ckpt_path, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model=model,
            data_loader=data_loader_train,
            optimizer=optimizer,
            epoch=epoch,
            loss_scaler=loss_scaler,
            log_writer=log_writer,
            args=args,
        )

        save_dict = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model": args.model,
        }
        if loss_scaler is not None:
            save_dict['loss_scaler'] = loss_scaler.state_dict()

        ckpt_path = os.path.join(args.output_dir, f"{args.model}_{args.experiment}_temp.pth")
        utils.save_on_master(save_dict, ckpt_path)
        print(f"model_path: {ckpt_path}")

        if args.output_dir and ((epoch + 1) % 100 == 0 or epoch + 1 == args.epochs):
            ckpt_path = os.path.join(args.output_dir,
                                     "{}_{}_{:04d}.pth".format(args.model, args.experiment,
                                                                     epoch))
            utils.save_on_master(save_dict, ckpt_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(
                    args.output_dir,
                    "{}_{}_log.txt".format(
                        args.model,
                        args.experiment
                    )
            ), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    if not misc.is_main_process():
        def print_pass(*args):
            pass
        builtins.print = print_pass

    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
