import os
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from functools import partial
from collections import OrderedDict

from mmcv.runner import _load_checkpoint, load_state_dict
from mmdet.utils import get_root_logger
from mmdet.models.builder import HEADS
from .bbox_head import BBoxHead
from models.vision_transformer import Block, trunc_normal_
from ...utils.positional_encoding import get_2d_sincos_pos_embed


@HEADS.register_module()
class MAEBBoxHead(BBoxHead):
    def __init__(self,
                 in_channels,
                 img_size=224,
                 patch_size=16, 
                 embed_dim=256, 
                 depth=4,
                 num_heads=8, 
                 mlp_ratio=4., 
                 qkv_bias=True, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 use_checkpoint=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.use_checkpoint = use_checkpoint
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        num_patches = (img_size // patch_size) ** 2
        self.num_patches = num_patches

        # MAE decoder specifics
        self.norm = norm_layer(in_channels)
        self.decoder_embed = nn.Linear(in_channels, embed_dim, bias=True)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.decoder_blocks = nn.ModuleList([
            Block(
                embed_dim, num_heads, mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.decoder_box_norm = norm_layer(embed_dim)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            self.fc_cls = nn.Linear(embed_dim, self.num_classes + 1)
        if self.with_reg:
            out_dim_reg = 4 if self.reg_class_agnostic else 4 * self.num_classes
            self.fc_reg = nn.Linear(embed_dim, out_dim_reg)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

    def init_weights(self, pretrained):
        logger = get_root_logger()
        if os.path.isfile(pretrained):
            logger.info('loading checkpoint for {}'.format(self.__class__))
            checkpoint = _load_checkpoint(pretrained, map_location='cpu')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            # TODO: match the decoder blocks, norm and head in the state_dict due to the different prefix
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('patch_embed') or k.startswith('blocks'):
                    continue
                elif k in ['pos_embed']:
                    continue
                else:
                    new_state_dict[k] = v
            load_state_dict(self, new_state_dict, strict=False, logger=logger)
        else:
            raise ValueError(f"checkpoint path {pretrained} is invalid")

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.decoder_pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.decoder_pos_embed
        class_pos_embed = self.decoder_pos_embed[:, 0]
        patch_pos_embed = self.decoder_pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
    
    def forward(self, x):
        B, C, W, H = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = self.decoder_embed(x)

        x = x + self.interpolate_pos_encoding(x, W * self.patch_size, H * self.patch_size)[:, 1:, :]
        for blk in self.decoder_blocks:
            if self.use_checkpoint:
                x = checkpoint(blk, x)
            else:
                x = blk(x)
        x = self.decoder_box_norm(x.mean(dim=1))
        cls_score = self.fc_cls(x) if self.with_cls else None
        bbox_pred = self.fc_reg(x) if self.with_reg else None
        return cls_score, bbox_pred
