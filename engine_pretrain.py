# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------
import math
import sys
from typing import Iterable
import builtins

import torch
import torch.nn.functional as F

import util.misc as misc
import util.lr_sched as lr_sched
import util.utils as utils


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    epoch: int,
                    loss_scaler,
                    log_writer=None,
                    args=None):
    if not misc.is_main_process():
        def print_pass(*args):
            pass
        builtins.print = print_pass

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    print_freq = 20

    accum_iter = args.accum_iter
    optimizer.zero_grad()

    if args.drop_pos_type in ['mae_pos_target', 'multi_task']:
        sigma = (1 - epoch / float(args.epochs)) * args.label_smoothing_sigma if args.sigma_decay else args.label_smoothing_sigma
        num_patches = (args.input_size // args.token_size) ** 2
        smooth = _get_label_smoothing_map(int(num_patches), sigma)

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (batch, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        it = len(data_loader) * epoch + data_iter_step
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        images, bool_masked_pos = batch

        samples = images.cuda(non_blocking=True)
        bool_masked_pos = bool_masked_pos.cuda(non_blocking=True).flatten(1).to(torch.bool)   # (N, L)

        with torch.cuda.amp.autocast(loss_scaler is not None):
            if args.drop_pos_type == 'vanilla_drop_pos':
                acc1, loss = model(samples, mask_ratio=args.mask_ratio)
            elif args.drop_pos_type == 'mae_pos_target':
                acc1, loss = model(samples, mask_ratio=args.mask_ratio,
                                   pos_mask_ratio=args.pos_mask_ratio, smooth=smooth)
            elif args.drop_pos_type == 'vanilla_mae':
                loss = model(samples, mask_ratio=args.mask_ratio)
            elif args.drop_pos_type == 'multi_task':
                acc1, loss_drop_pos, loss_mae = model(samples, mask_ratio=args.mask_ratio,
                                                      pos_mask_ratio=args.pos_mask_ratio,
                                                      smooth=smooth)
                loss = args.pos_weight * loss_drop_pos + loss_mae

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, skip".format(loss_value))
            sys.exit(1)

        loss /= accum_iter

        if loss_scaler is None:
            loss.backward()
            if args.clip_grad:
                grad_norm = utils.clip_gradients(model, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, model, freeze_last_layer=0)
            optimizer.step()
        else:
            loss_scaler.scale(loss).backward()
            if args.clip_grad:
                loss_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                grad_norm = utils.clip_gradients(model, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, model, freeze_last_layer=0)
            loss_scaler.step(optimizer)
            loss_scaler.update()

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss_value)
        if args.drop_pos_type == 'multi_task':
            metric_logger.update(mae=loss_mae.item())
            metric_logger.update(pos=loss_drop_pos.item())

        if args.drop_pos_type != 'vanilla_mae':
            metric_logger.update(acc1=acc1)
        metric_logger.update(lr=lr)

        # loss_value_reduce = misc.all_reduce_mean(loss_value)
        # if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
        #     log_writer.add_scalar('train_loss', loss_value_reduce, it)
        #     log_writer.add_scalar('lr', lr, it)
        #     if args.drop_pos_type == 'multi_task':
        #         log_writer.add_scalar('mae_loss', misc.all_reduce_mean(loss_mae), it)
        #         log_writer.add_scalar('pos_loss', misc.all_reduce_mean(loss_drop_pos), it)
        #     if args.drop_pos_type != 'vanilla_mae':
        #         log_writer.add_scalar('acc1', misc.all_reduce_mean(acc1), it)

        if (data_iter_step + 1) >= len(data_loader):
            break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def _get_label_smoothing_map(num_patches, sigma):
    if sigma == 0.:
        # without label smoothing
        return torch.eye(num_patches)

    weight = torch.zeros([num_patches, num_patches])
    w = int(math.sqrt(num_patches))

    # for each patch i (0 to num_patches-1), its coordinate is (i // w, i % w)
    for i in range(num_patches):
        x_i, y_i = i // w, i % w
        for j in range(num_patches):
            x_j, y_j = j // w, j % w
            dist = (x_i - x_j) ** 2 + (y_i - y_j) ** 2
            weight[i, j] = math.exp(-dist / sigma ** 2)

    # normalize
    return weight / weight.sum(-1)