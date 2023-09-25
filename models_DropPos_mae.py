# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import PatchEmbed, Block, DropPath, Mlp

from util.pos_embed import get_2d_sincos_pos_embed
from einops import rearrange


class DropPositionMaskedAutoEncoderViT(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=True,
                 mask_token_type='param', shuffle=False, multi_task=False, conf_ignore=False, attn_guide=False):
        super().__init__()

        self.norm_pix_loss = norm_pix_loss
        self.mask_token_type = mask_token_type
        self.shuffle = shuffle
        self.multi_task = multi_task
        self.conf_ignore = conf_ignore
        self.attn_guide = attn_guide

        # --------------------------------------------------------------------------
        # DropPos encoder specifics
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding
        # mask token for position
        self.mask_pos_token = nn.Parameter(torch.zeros(1, 1, embed_dim),
                                           requires_grad=True if mask_token_type == 'param' else False)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # DropPos decoder specifics (w/o position embedding)
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        # self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
        #                                       requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, num_patches, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        if multi_task:
            # --------------------------------------------------------------------------
            # MAE decoder specifics (w/ position embedding)
            self.aux_decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
            self.aux_decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                                      requires_grad=False)  # fixed sin-cos embedding
            # mask token for patches
            self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim),
                                           requires_grad=True if mask_token_type == 'param' else False)

            self.aux_decoder_blocks = nn.ModuleList([
                Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None,
                      norm_layer=norm_layer)
                for i in range(decoder_depth)])

            self.aux_decoder_norm = norm_layer(decoder_embed_dim)
            self.aux_decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch
            # --------------------------------------------------------------------------

        # label smoothing for positions
        # self._get_label_smoothing_map(num_patches, sigma)
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        if self.multi_task:
            decoder_pos_embed = get_2d_sincos_pos_embed(self.aux_decoder_pos_embed.shape[-1],
                                                        int(self.patch_embed.num_patches ** .5), cls_token=True)
            self.aux_decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
            torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_pos_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        # x = rearrange(imgs, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        # remove the second subset
        ids_remove = ids_shuffle[:, len_keep:]

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore).bool()

        return ids_keep, mask, ids_restore, ids_remove

    @torch.no_grad()
    def get_last_attention(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block, following timm
                # y, attn = blk.attn(blk.norm1(x))
                x = blk.norm1(x)
                B, N, C = x.shape
                qkv = blk.attn.qkv(x).reshape(B, N, 3, blk.attn.num_heads, C // blk.attn.num_heads).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

                attn = (q @ k.transpose(-2, -1)) * blk.attn.scale
                attn = attn.softmax(dim=-1)
                attn = blk.attn.attn_drop(attn)
                return attn

    @torch.no_grad()
    def get_feature_similarity(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        cls_tokens = x[:, :1, :]
        patch_tokens = x[:, 1:, :]
        sim = F.cosine_similarity(cls_tokens, patch_tokens, dim=-1)

        return F.softmax(sim / 0.1, dim=-1)

    def forward_encoder(self, x, mask_ratio, pos_mask_ratio):
        outs = {}
        inputs = x.detach().clone()

        # embed patches w/o [cls] token
        x = self.patch_embed(x)
        N, L, D = x.shape

        # generate mask
        ids_keep, mask, ids_restore, ids_remove = self.random_masking(x, mask_ratio)
        outs['mask'], outs['ids_keep'], outs['ids_restore'] = mask, ids_keep, ids_restore
        # gather patch embeddings and position embeddings
        x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        pos_embed_all = self.pos_embed[:, 1:, :].data.repeat(N, 1, 1)  # w/o [cls] token
        pos_embed_vis = torch.gather(pos_embed_all, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)).detach()

        # random masking for position embedding
        ids_keep_pos, mask_pos, ids_restore_pos, ids_remove_pos = self.random_masking(x, pos_mask_ratio)
        outs['mask_pos'], outs['ids_keep_pos'], outs['ids_restore_pos'] = mask_pos, ids_keep_pos, ids_restore_pos

        # gather position embeddings
        pos_embed = torch.gather(pos_embed_vis, dim=1, index=ids_keep_pos.unsqueeze(-1).repeat(1, 1, D))

        # append mask tokens to position embeddings
        mask_pos_length = mask_pos.sum().item()
        if self.mask_token_type == 'param':
            mask_pos_tokens = self.mask_pos_token.repeat(N, mask_pos_length, 1)
        elif self.mask_token_type == 'zeros':
            mask_pos_tokens = torch.zeros((N, mask_pos_length, self.embed_dim)).to(x.device)
        elif self.mask_token_type == 'wrong_pos':
            removed_pos_embed = torch.gather(pos_embed_vis, dim=1, index=ids_remove_pos.unsqueeze(-1).repeat(1, 1, D))
            # convert to numpy, since numpy shuffles the first dimension, we have to transpose first
            removed_pos_embed = removed_pos_embed.detach().cpu().permute(1, 0, 2).numpy()        # [N, L, D] -> [L, N, D]
            np.random.shuffle(removed_pos_embed)
            # restore to torch
            removed_pos_embed = torch.from_numpy(removed_pos_embed).permute(1, 0, 2)    # [L, N, D] -> [N, L, D]
            mask_pos_tokens = removed_pos_embed.to(x.device)
        else:
            raise Exception('unknown mask_token_type: {}'.format(self.mask_token_type))

        pos_embed = torch.cat([pos_embed, mask_pos_tokens], dim=1)

        # restore position embeddings before adding
        pos_embed = torch.gather(pos_embed, dim=1, index=ids_restore_pos.unsqueeze(-1).repeat(1, 1, D))

        # add position embedding w/o [cls] token
        x = x + pos_embed

        if self.shuffle:
            # generate shuffle indexes first
            ids_keep_shuffle, _, ids_restore_shuffle, _ = self.random_masking(x, 0.)

            # gather
            x = torch.gather(x, dim=1, index=ids_keep_shuffle.unsqueeze(-1).repeat(1, 1, D))
            outs['ids_restore_shuffle'] = ids_restore_shuffle

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # get last self-attention
        if self.attn_guide:
            # get attentions
            # attn = self.get_last_attention(inputs)
            # attn = attn[:, :, 0, 1:].mean(1)    # [N, num_patches]
            # outs['attn_full'] = attn

            # get similarities
            attn = self.get_feature_similarity(inputs)
            outs['attn_full'] = attn

            # gather visible patches
            attn = torch.gather(attn, dim=1, index=ids_keep)
            outs['attn'] = attn / attn.sum(-1, keepdims=True)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        outs['x'] = self.norm(x)

        return outs

    def forward_decoder(self, outs):
        x = outs['x']

        # --------------------------------------------------------------------------
        # DropPos decoder forward
        x = self.decoder_embed(x)
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        x = x[:, 1:, :]

        if self.shuffle:
            ids_restore_shuffle = outs['ids_restore_shuffle']
            x = torch.gather(x, dim=1, index=ids_restore_shuffle.unsqueeze(-1).repeat(1, 1, x.shape[-1]))
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder forward
        if self.multi_task:
            x_mae = outs['x'].clone()
            x_mae = self.aux_decoder_embed(x_mae)

            # append mask tokens to sequence
            mask_tokens = self.mask_token.repeat(x_mae.shape[0], outs['ids_restore'].shape[1] + 1 - x_mae.shape[1], 1)
            x_ = torch.cat([x_mae[:, 1:, :], mask_tokens], dim=1)  # no cls token
            x_ = torch.gather(x_, dim=1, index=outs['ids_restore'].unsqueeze(-1).repeat(1, 1, x_mae.shape[2]))  # unshuffle
            x_mae = torch.cat([x_mae[:, :1, :], x_], dim=1)  # append cls token

            # add pos embed
            x_mae = x_mae + self.aux_decoder_pos_embed

            # apply Transformer blocks
            for blk in self.aux_decoder_blocks:
                x_mae = blk(x_mae)
            x_mae = self.aux_decoder_norm(x_mae)

            # predictor projection
            x_mae = self.aux_decoder_pred(x_mae)

            # remove cls token
            x_mae = x_mae[:, 1:, :]

            return x, x_mae
        # --------------------------------------------------------------------------

        return x

    def forward_mae_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward_drop_pos_loss(self, pred, mask, ids_keep, mask_pos, smooth, attn):
        smooth = smooth.to(pred.device).detach()

        N, L = mask.shape
        num_vis = pred.shape[1]
        labels = torch.arange(L).repeat(N, 1).to(pred.device).detach()
        labels = torch.gather(labels, dim=1, index=ids_keep)

        labels_smooth = torch.gather(smooth.repeat(N, 1, 1), dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, L))
        log_prob = F.log_softmax(pred, dim=-1)      # [N, x, L]
        if self.conf_ignore:
            # ignore if confidence > smooth
            conf_thresh = torch.diag(smooth).cuda().detach().unsqueeze(0).repeat(N, 1)     # [N, L]
            conf_thresh = torch.gather(conf_thresh, dim=1, index=ids_keep)  # [N, x]
            with torch.no_grad():
                prob = F.softmax(pred, dim=-1)  # [N, x, L]
                conf = torch.gather(prob, dim=2, index=ids_keep.unsqueeze(1).repeat(1, num_vis, 1)) # [N, x, x]
                conf = torch.diagonal(conf, dim1=1, dim2=2)
                conf_mask = (conf < conf_thresh).bool()
            mask_pos = mask_pos * conf_mask

        if attn is not None:
            mask_pos = mask_pos * attn

        # loss = criterion(pred.permute(0, 2, 1), labels) * mask_pos
        loss = (-labels_smooth * log_prob).sum(-1) * mask_pos
        loss = loss.sum() / mask_pos.sum()

        # evaluate position acc
        with torch.no_grad():
            pred_position = torch.argmax(pred.detach(), dim=-1)
            acc1 = (pred_position == labels) * mask_pos
            acc1 = acc1.sum() / mask_pos.sum()
            acc1 = acc1.item()

        return acc1, loss

    def forward(self, imgs, mask_ratio, pos_mask_ratio, smooth):
        outs = self.forward_encoder(imgs, mask_ratio, pos_mask_ratio)
        pred = self.forward_decoder(outs)
        if self.multi_task:
            acc1, loss_drop_pos = self.forward_drop_pos_loss(pred[0], outs['mask'], outs['ids_keep'], outs['mask_pos'], smooth,
                                                             attn=outs['attn'] if self.attn_guide else None)
            loss_mae = self.forward_mae_loss(imgs, pred[1], outs['mask'])
            return acc1, loss_drop_pos, loss_mae

        return self.forward_drop_pos_loss(pred, outs['mask'], outs['ids_keep'], outs['mask_pos'], smooth,
                                          attn=outs['attn'] if self.attn_guide else None)


def DropPos_mae_vit_small_patch16_dec512d2b(**kwargs):
    model = DropPositionMaskedAutoEncoderViT(
        patch_size=16, embed_dim=384, depth=12, num_heads=6,
        decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def DropPos_mae_vit_small_patch16_dec512d8b(**kwargs):
    model = DropPositionMaskedAutoEncoderViT(
        patch_size=16, embed_dim=384, depth=12, num_heads=6,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def DropPos_mae_vit_base_patch16_dec512d2b(**kwargs):
    model = DropPositionMaskedAutoEncoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def DropPos_mae_vit_base_patch32_dec512d2b(**kwargs):
    model = DropPositionMaskedAutoEncoderViT(
        patch_size=32, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def DropPos_mae_vit_base_patch16_dec512d8b(**kwargs):
    model = DropPositionMaskedAutoEncoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def DropPos_mae_vit_large_patch16_dec512d2b(**kwargs):
    model = DropPositionMaskedAutoEncoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def DropPos_mae_vit_large_patch16_dec512d8b(**kwargs):
    model = DropPositionMaskedAutoEncoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def DropPos_mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = DropPositionMaskedAutoEncoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
DropPos_mae_vit_base_patch16 = DropPos_mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
DropPos_mae_vit_large_patch16 = DropPos_mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
DropPos_mae_vit_huge_patch14 = DropPos_mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
