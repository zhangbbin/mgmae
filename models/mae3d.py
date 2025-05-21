import torch
import torch.nn as nn

import numpy as np
import random
from networks import patch_embed_layers

from timm.models.layers.helpers import to_3tuple

__all__ = ["MAE3D"]

def build_3d_sincos_position_embedding(grid_size, embed_dim, num_tokens=1, temperature=10000.):
    grid_size = to_3tuple(grid_size)
    h, w, d = grid_size
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_d = torch.arange(d, dtype=torch.float32)

    grid_h, grid_w, grid_d = torch.meshgrid(grid_h, grid_w, grid_d)
    assert embed_dim % 6 == 0, 'Embed dimension must be divisible by 6 for 3D sin-cos position embedding'
    pos_dim = embed_dim // 6
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1. / (temperature**omega)
    out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
    out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
    out_d = torch.einsum('m,d->md', [grid_d.flatten(), omega])
    pos_emb = torch.cat([torch.sin(out_h), torch.cos(out_h), torch.sin(out_w), torch.cos(out_w), torch.sin(out_d), torch.cos(out_d)], dim=1)[None, :, :]

    assert num_tokens == 1 or num_tokens == 0, "Number of tokens must be of 0 or 1"
    if num_tokens == 1:
        pe_token = torch.zeros([1, 1, embed_dim], dtype=torch.float32)
        pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
    else:
        pos_embed = nn.Parameter(pos_emb)
    pos_embed.requires_grad = False
    return pos_embed

def build_perceptron_position_embedding(grid_size, embed_dim, num_tokens=1):
    pos_emb = torch.rand([1, np.prod(grid_size), embed_dim])
    nn.init.normal_(pos_emb, std=.02)

    assert num_tokens == 1 or num_tokens == 0, "Number of tokens must be of 0 or 1"
    if num_tokens == 1:
        pe_token = torch.zeros([1, 1, embed_dim], dtype=torch.float32)
        pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
    else:
        pos_embed = nn.Parameter(pos_emb)
    return pos_embed

def patchify_image(x, patch_size):
    """
    ATTENTION!!!!!!!
    Different from 2D version patchification: The final axis follows the order of [ph, pw, pd, c] instead of [c, ph, pw, pd]
    """
    # patchify input, [B,C,H,W,D] --> [B,C,gh,ph,gw,pw,gd,pd] --> [B,gh*gw*gd,ph*pw*pd*C]
    B, C, H, W, D = x.shape
    patch_size = to_3tuple(patch_size)
    grid_size = (H // patch_size[0], W // patch_size[1], D // patch_size[2])

    x = x.reshape(B, C, grid_size[0], patch_size[0], grid_size[1], patch_size[1], grid_size[2], patch_size[2]) # [B,C,gh,ph,gw,pw,gd,pd]
    x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(B, np.prod(grid_size), np.prod(patch_size) * C) # [B,gh*gw*gd,ph*pw*pd*C]

    return x

def unpatchify_image(x):
    """
    ATTENTION!!!!!!
    x: (N, L, patch_size**3)
    imgs: (B, 1, H, W, D)
    """
    p = 16
    h = w = d = 6
    assert h * w * d == x.shape[1]

    x = x.reshape(x.shape[0], h, w, d, p, p, p, 1)
    x = torch.einsum('nhjkpqrc->nchpjqkr', x)
    imgs = x.reshape(x.shape[0], 1, h * p, w * p, d * p)

    return imgs


def batched_shuffle_indices(batch_size, length, device):
    """
    Generate random permutations of specified length for batch_size times
    Motivated by https://discuss.pytorch.org/t/batched-shuffling-of-feature-vectors/30188/4
    """
    rand = torch.rand(batch_size, length).to(device)
    batch_perm = rand.argsort(dim=1)
    return batch_perm

class MAE3D(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, 
                 encoder, 
                 decoder, 
                 args):
        super().__init__()
        self.args = args
        input_size = to_3tuple(args.input_size)
        patch_size = to_3tuple(args.patch_size)
        self.input_size = input_size
        self.patch_size = patch_size

        out_chans = args.in_chans * np.prod(self.patch_size)
        self.out_chans = out_chans

        grid_size = []
        for in_size, pa_size in zip(input_size, patch_size):
            assert in_size % pa_size == 0, "input size and patch size are not proper"
            grid_size.append(in_size // pa_size)
        self.grid_size = grid_size

        # build positional encoding for encoder and decoder
        if args.pos_embed_type == 'sincos':
            with torch.no_grad():
                self.encoder_pos_embed = build_3d_sincos_position_embedding(grid_size, 
                                                                            args.encoder_embed_dim, 
                                                                            num_tokens=0)
                self.decoder_pos_embed = build_3d_sincos_position_embedding(grid_size, 
                                                                            args.decoder_embed_dim, 
                                                                            num_tokens=0)
        elif args.pos_embed_type == 'perceptron':
            self.encoder_pos_embed = build_perceptron_position_embedding(grid_size,
                                                                        args.encoder_embed_dim,
                                                                        num_tokens=0)
            with torch.no_grad():
                self.decoder_pos_embed = build_3d_sincos_position_embedding(grid_size, 
                                                                            args.decoder_embed_dim, 
                                                                            num_tokens=0)

        self.encoder_pos_embed_local = build_3d_sincos_position_embedding((2, 2, 2),
                                                                            args.encoder_embed_dim,
                                                                            num_tokens=0)

        self.decoder_pos_embed_local = build_3d_sincos_position_embedding((2, 2, 2),
                                                                    args.decoder_embed_dim,
                                                                    num_tokens=0)

        # build encoder and decoder
        embed_layer = getattr(patch_embed_layers, args.patchembed)
        self.encoder = encoder(patch_size=patch_size,
                               in_chans=args.in_chans,
                               embed_dim=args.encoder_embed_dim,
                               depth=args.encoder_depth,
                               num_heads=args.encoder_num_heads,
                               embed_layer=embed_layer)
        self.decoder = decoder(patch_size=patch_size,
                               num_classes=out_chans,
                               embed_dim=args.decoder_embed_dim,
                               depth=args.decoder_depth,
                               num_heads=args.decoder_num_heads)

        self.encoder_to_decoder = nn.Linear(args.encoder_embed_dim, args.decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, args.decoder_embed_dim))
        
        self.patch_norm = nn.LayerNorm(normalized_shape=(out_chans,), eps=1e-6, elementwise_affine=False)

        self.criterion = nn.MSELoss()

        # initialize encoder_to_decoder and mask token
        nn.init.xavier_uniform_(self.encoder_to_decoder.weight)
        nn.init.normal_(self.mask_token, std=.02)

    def local_sampling(self, x):
        # 定义子体积的大小
        sub_volume_shape = (32, 32, 32)

        # 获取每个维度的最大起始索引
        max_start_idx = (x.shape[2] - sub_volume_shape[0],
                         x.shape[3] - sub_volume_shape[1],
                         x.shape[4] - sub_volume_shape[2])

        # 存储子体积的张量列表
        sub_volumes = []

        for i in range(x.shape[0]):
            # 随机选择每个维度的起始索引
            start_x = random.randint(0, max_start_idx[0])
            start_y = random.randint(0, max_start_idx[1])
            start_z = random.randint(0, max_start_idx[2])

            # 截取子体积
            sub_volume = x[i:i + 1, :,
                         start_x:start_x + sub_volume_shape[0],
                         start_y:start_y + sub_volume_shape[1],
                         start_z:start_z + sub_volume_shape[2]]
            sub_volumes.append(sub_volume)

        # 将子体积列表拼接成形状为 (8, 1, 32, 32, 32) 的张量
        result = torch.cat(sub_volumes, dim=0)

        return result

    def nuclear_norm(self, x):
        # x: (batch_size, sequence_length, features)
        x = x[:, 1:, :]
        # 归一化
        x = torch.softmax(x, dim=-1)
        # 计算核范数
        norms = []
        for i in range(x.shape[0]):  # 对每个样本计算
            # 对最后两个维度进行SVD
            _, s, _ = torch.svd(x[i].view(-1, 768))
            norm = s.sum()  # 核范数是奇异值的和
            norms.append(norm)
        return -torch.tensor(norms).mean()  # 返回均值核范数

    def local_forward(self, x):
        args = self.args
        batch_size = x.size(0)
        in_chans = x.size(1)
        assert in_chans == args.in_chans
        out_chans = self.out_chans

        x = patchify_image(x, self.patch_size)  # (8, 8, 4096)

        # compute length for selected and masked
        length = np.prod((2, 2, 2))
        sel_length = int(length * (1 - 0.5))
        msk_length = length - sel_length

        # generate batched shuffle indices
        shuffle_indices = batched_shuffle_indices(batch_size, length, device=x.device)

        # select and mask the input patches
        shuffled_x = x.gather(dim=1, index=shuffle_indices[:, :, None].expand(-1, -1, out_chans))
        sel_x = shuffled_x[:, :sel_length, :]
        msk_x = shuffled_x[:, -msk_length:, :]
        # select and mask the indices
        sel_indices = shuffle_indices[:, :sel_length]

        # select the position embedings accordingly
        sel_encoder_pos_embed = self.encoder_pos_embed_local.expand(batch_size, -1, -1).gather(dim=1, index=sel_indices[:, :,
                                                                                                      None].expand(-1,
                                                                                                                   -1,
                                                                                                                   args.encoder_embed_dim))

        # forward encoder & proj to decoder dimension
        sel_x = self.encoder(sel_x, sel_encoder_pos_embed)  # (8, 5, 768)

        # Nuclear-norm
        nnm_loss = self.nuclear_norm(sel_x)

        sel_x = self.encoder_to_decoder(sel_x)

        # combine the selected tokens and mask tokens in the shuffled order
        all_x = torch.cat([sel_x, self.mask_token.expand(batch_size, msk_length, -1)], dim=1)
        # shuffle all the decoder positional encoding
        shuffled_decoder_pos_embed = self.decoder_pos_embed_local.expand(batch_size, -1, -1).gather(dim=1,
                                                                                              index=shuffle_indices[:,
                                                                                                    :, None].expand(-1,
                                                                                                                    -1,
                                                                                                                    args.decoder_embed_dim))
        # add the shuffled positional embedings to encoder output tokens
        all_x[:, 1:, :] += shuffled_decoder_pos_embed

        # forward decoder
        all_x = self.decoder(all_x)

        # loss
        local_loss = self.criterion(input=all_x[:, -msk_length:, :], target=self.patch_norm(msk_x.detach()))

        return local_loss, nnm_loss


    def forward(self, x, return_image=False):
        '''
        :param x: (8, 1, 96, 96, 96)
        :param return_image: True or False
        :return: loss
        '''
        args = self.args
        batch_size = x.size(0)
        in_chans = x.size(1)
        assert in_chans == args.in_chans
        out_chans = self.out_chans

        # local reconstruction
        sub_column = self.local_sampling(x)
        local_loss, nnm_loss = self.local_forward(sub_column)
        # print(local_loss)

        x = patchify_image(x, self.patch_size)  # [B,gh*gw*gd,ph*pw*pd*C]

        # compute length for selected and masked
        length = np.prod(self.grid_size)
        sel_length = int(length * (1 - args.mask_ratio))
        msk_length = length - sel_length

        # generate batched shuffle indices
        shuffle_indices = batched_shuffle_indices(batch_size, length, device=x.device)
        unshuffle_indices = shuffle_indices.argsort(dim=1)

        # select and mask the input patches
        print(shuffle_indices[:, :, None].expand(-1, -1, out_chans).shape)
        shuffled_x = x.gather(dim=1, index=shuffle_indices[:, :, None].expand(-1, -1, out_chans))
        sel_x = shuffled_x[:, :sel_length, :]
        msk_x = shuffled_x[:, -msk_length:, :]
        # select and mask the indices
        # shuffle_indices = F.pad(shuffle_indices + 1, pad=(1, 0), mode='constant', value=0)
        sel_indices = shuffle_indices[:, :sel_length]
        # msk_indices = shuffle_indices[:, -msk_length:]

        # select the position embedings accordingly
        sel_encoder_pos_embed = self.encoder_pos_embed.expand(batch_size, -1, -1).gather(dim=1, index=sel_indices[:, :, None].expand(-1, -1, args.encoder_embed_dim))

        # forward encoder & proj to decoder dimension
        sel_x = self.encoder(sel_x, sel_encoder_pos_embed)
        sel_x = self.encoder_to_decoder(sel_x)

        # combine the selected tokens and mask tokens in the shuffled order
        all_x = torch.cat([sel_x, self.mask_token.expand(batch_size, msk_length, -1)], dim=1)
        # shuffle all the decoder positional encoding
        shuffled_decoder_pos_embed = self.decoder_pos_embed.expand(batch_size, -1, -1).gather(dim=1, index=shuffle_indices[:, :, None].expand(-1, -1, args.decoder_embed_dim))
        # add the shuffled positional embedings to encoder output tokens
        all_x[:, 1:, :] += shuffled_decoder_pos_embed

        # forward decoder
        all_x = self.decoder(all_x)

        # loss
        loss = self.criterion(input=all_x[:, -msk_length:, :], target=self.patch_norm(msk_x.detach())) + 0.3 * local_loss + 0.3 * nnm_loss

        if return_image:
            # unshuffled all the tokens
            masked_x = torch.cat([shuffled_x[:, :sel_length, :], 0. * torch.ones(batch_size, msk_length, out_chans).to(x.device)], dim=1).gather(dim=1, index=unshuffle_indices[:, :, None].expand(-1, -1, out_chans))
            recon = all_x[:, 1:, :].gather(dim=1, index=unshuffle_indices[:, :, None].expand(-1, -1, out_chans))
            recon = recon * (x.var(dim=-1, unbiased=True, keepdim=True).sqrt() + 1e-6) + x.mean(dim=-1, keepdim=True)
            return loss, x.detach(), recon.detach(), masked_x.detach()
        else:
            return loss
