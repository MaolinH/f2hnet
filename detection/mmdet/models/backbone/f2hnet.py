# --------------------------------------------------------
# written By Maolin Huang
# based on mmdetection
# --------------------------------------------------------
import warnings
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmengine.logging import MMLogger
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import constant_init, trunc_normal_,trunc_normal_init
from mmengine.runner.checkpoint import CheckpointLoader
from mmengine.utils import to_2tuple
from timm.layers import DropPath
from mmdet.registry import MODELS


def img2seq(x, dilation,seq_size):
    B, H, W, C = x.shape
    dh,dw = dilation
    sh,sw = seq_size
    rh,rw = H//(dh*sh), W//(dw*sw)
    x = x.view(B, rh,sh,dh,rw,sw,dw, C)
    x = x.permute(0, 1, 4, 3, 6, 2, 5, 7)           # (B,rh,rw,dh,dw,sh,sw,C)
    x = x.reshape(-1, sh*sw, C)
    return x  # (B*nS, L,C)


def seq2img(x,dilation, seq_size, H: int, W: int):
    B = int(x.shape[0] / (H * W / seq_size[0] / seq_size[1]))
    dh,dw = dilation
    sh,sw = seq_size
    rh,rw = H//(dh*sh), W//(dw*sw)
    x = x.reshape(B,rh,rw,dh,dw,sh,sw,-1)   # (B,rh,rw,dh,dw,sh,sw,C)
    x = x.permute(0, 1, 5, 3, 2, 6, 4, 7)
    x = x.reshape(B, H, W, -1)
    return x  # (B,L,C)

class MLP(BaseModule):
    def __init__(self, in_dim, hide_dim, activation=nn.GELU, dropout=0.,
                 with_cp=False, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.in_dim = in_dim
        self.hide_dim = hide_dim
        self.activation = activation()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(in_dim, hide_dim)
        self.fc2 = nn.Linear(hide_dim, in_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Scale(BaseModule):
    def __init__(self, dim=64, layer_scale=1e-6, format = 'blc',with_cp=False, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.layer_scale = nn.Parameter(layer_scale*torch.ones(dim), requires_grad=True)
        self.format = format

    def forward(self, x):
        if self.format == 'blc':
            x = self.layer_scale*x
        elif self.format == 'bchw':
            x = self.layer_scale[:, None, None] * x
        return x

class FocalBlock(BaseModule):
    def __init__(self, dim=64, kernel_size=3, groups=1, ratio=4, path_drop=0.,
                 layer_scale=None,with_cp=False, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.with_cp = with_cp
        self.dim = dim
        self.out_dim = ratio * dim
        self.kernel_size = kernel_size
        self.groups = groups or dim
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=groups),
            nn.BatchNorm2d(dim,eps=1e-6),
            nn.Conv2d(dim, self.out_dim, 1),
            nn.GELU(),
            nn.Conv2d(self.out_dim, dim, 1),
           )
        self.path_drop = DropPath(path_drop) if path_drop > 0 else nn.Identity()
        self.scale = nn.Identity()
        if layer_scale is not None:
            self.scale = Scale(dim, layer_scale, format='bchw')


    def forward(self, x, hw_shape):
        def _inner_forward(x,):
            H, W = hw_shape
            x = x.reshape(-1, H, W, self.dim).permute(0, 3, 1, 2)
            x = self.scale(x) + self.path_drop(self.conv(x))
            x = x.flatten(2).transpose(1, 2)
            return x
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x

class Attention(BaseModule):
    def __init__(self,
                 seq_size,
                 dim=96,
                 num_heads=4,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 qkv_bias=True,
                 rpe=True,
                 with_cp=False,
                 init_cfg=None
                ):
        super().__init__(init_cfg=init_cfg)
        self.with_cp = with_cp
        self.dim = dim
        self.num_heads = num_heads
        hide_dim = dim // num_heads
        self.qk_scale = qk_scale or hide_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.rpe = rpe
        if rpe:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * seq_size[0] - 1) * (2 * seq_size[1] - 1), num_heads))
            Sh, Sw = seq_size
            rel_index_coords = self.double_step_seq(2 * Sw - 1, Sh, 1, Sw)
            rel_position_index = rel_index_coords + rel_index_coords.T
            rel_position_index = rel_position_index.flip(1).contiguous()
            self.register_buffer('relative_position_index', rel_position_index)
            trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, mask=None):
        B_, L, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B_, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # (3,B_,nH,L,head_dim)
        q, k, v = qkv
        # # -------------------------------------------------------------------------------------------------
        # attn = q @ k.transpose(-1, -2) * self.qk_scale
        # if self.rpe:
        #     relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(L, L,
        #                                                                                                            -1)  # Wh*Ww,Wh*Ww,nH
        #     relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        #     attn = attn + relative_position_bias.unsqueeze(0)
        # if mask is not None:
        #     nS = mask.shape[0]
        #     N = attn.shape[-1]
        #     attn = attn.view(B_//nS,nS,self.num_heads,N,N)+mask[None,:,None,:,:]
        #     attn = attn.reshape(-1,self.num_heads,N,N)
        # attn = self.attn_drop(attn)
        # attn = self.softmax(attn)
        # attn = (attn @ v).transpose(1, 2).reshape(B_, L, C)
        # -------------------------------------------------------------------------------------------------
        if self.rpe:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(L, L,
                                                                                                                   -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)  # nH, Wh*Ww, Wh*Ww
        else:
            relative_position_bias=None
        attn = F.scaled_dot_product_attention(query=q, key=k, value=v, attn_mask=relative_position_bias)
        attn = attn.transpose(1, 2).reshape(B_, L, C)
        # -------------------------------------------------------------------------------------------------
        x = self.proj(attn)
        x = self.proj_drop(x)
        # -------------------------------------------------------------------------------------------------
        return x

    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)

class Transformer(BaseModule):
    def __init__(self,
                 dilation,
                 seq_size,
                 dim=64,
                 num_heads=2,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 mlp_drop=0.,
                 path_drop=0.,
                 qkv_bias=True,
                 mlp_ratio=4,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 layer_scale=None,
                 rpe=True,
                 shift = False,
                 with_cp=False,
                 init_cfg=None
                 ):
        super().__init__(init_cfg=init_cfg)
        self.with_cp = with_cp
        self.dilation = dilation
        self.seq_size = seq_size
        self.shift = shift
        self.attn = Attention(
            seq_size=seq_size,
            dim=dim,
            num_heads=num_heads,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            qkv_bias=qkv_bias,
            rpe=rpe
            )
        self.dim = dim
        self.mlp_hide_dim = mlp_ratio * dim
        self.mlp = MLP(in_dim=dim, hide_dim=self.mlp_hide_dim, activation=act_layer, dropout=mlp_drop)

        self.norm_1 = norm_layer(dim)
        self.norm_2 = norm_layer(dim)
        self.path_drop = DropPath(path_drop) if path_drop > 0 else nn.Identity()

        if layer_scale is not None:
            self.scale_attn = Scale(dim, layer_scale=layer_scale, format='blc')
            self.scale_mlp = Scale(dim, layer_scale=layer_scale, format='blc')
        else:
            self.scale_attn = nn.Identity()
            self.scale_mlp = nn.Identity()

    def forward(self, x, hw_shape):
        def _inner_forward(x):
            B, L, C = x.shape
            H, W = hw_shape
            shortcut = x

            x = x.view(B, H, W, C)
            # pad feature maps to multiples of window size

            h_div = self.seq_size[0]*self.dilation[0]
            w_div = self.seq_size[1]*self.dilation[1]
            pad_h = (h_div - H % h_div) % h_div
            pad_w = (w_div- W % w_div) % w_div
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            H_pad, W_pad = x.shape[1], x.shape[2]
            mask = torch.zeros((1,H_pad, W_pad,1), device=x.device)
            
            if pad_h > 0 or pad_w>0:
                if pad_h>0:
                    mask[:,-pad_h:, :,:]=-1
                if pad_w>0:
                    mask[:, :,-pad_w:,:] = -1
                mask = img2seq(mask, self.dilation,self.seq_size)   # (nS, L, 1)
                mask = mask.squeeze(-1).unsqueeze(1)        # (nS,1, L)
                sh,sw = self.seq_size
                nS = H_pad*W_pad//(sh*sw)
                attn_mask = torch.zeros((nS,sh*sw, sh*sw),device=x.device)
                attn_mask = attn_mask.masked_fill(mask < 0, -100000)
            else:
                attn_mask = None

            x = self.norm_1(x)
            # -------------------------------------------------------------------------------------------------
            seq_x = img2seq(x, self.dilation, self.seq_size)
            attn = self.attn(seq_x, mask=attn_mask)
            x = seq2img(attn, self.dilation, self.seq_size, H=H_pad, W=W_pad)
            if pad_h > 0 or pad_w:
                x = x[:, :H, :W, :]

            x = x.reshape(B,L,C)
            x = self.scale_attn(shortcut) + self.path_drop(x)  # (B,L,C)
            # -------------------------------------------------------------------------------------------------
            x = self.scale_mlp(x) + self.path_drop(self.mlp(self.norm_2(x)))
            # -------------------------------------------------------------------------------------------------
            return x  # (B,L,C)
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x

class FCMixer(BaseModule):
    def __init__(self,
                 dilation,
                 seq_size,
                 dim=64,
                 num_heads=2,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 mlp_drop=0.,
                 path_drop=0.,
                 qkv_bias=True,
                 mlp_ratio=4,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 kernel_size=3,
                 layer_scale=None,
                 rpe=True,
                 shift = False,
                 groups=1,
                 focal=False,
                 coarse=False,
                 with_cp=False,
                 init_cfg=None
                 ):
        super().__init__(init_cfg)
        self.with_cp = with_cp
        self.focal = None
        if focal:
            self.focal = FocalBlock(
                dim=dim,
                kernel_size=kernel_size,
                groups=groups,
                ratio=mlp_ratio,
                path_drop=path_drop,
                layer_scale=layer_scale,
            )
        self.coarse = None
        if coarse:
            self.coarse = Transformer(
                                                 dilation=dilation,
                                                 seq_size=seq_size,
                                                 dim=dim,
                                                 num_heads=num_heads,
                                                 qk_scale=qk_scale,
                                                 attn_drop=attn_drop,
                                                 proj_drop=proj_drop,
                                                 mlp_drop=mlp_drop,
                                                 path_drop=path_drop,
                                                 qkv_bias=qkv_bias,
                                                 mlp_ratio=mlp_ratio,
                                                 act_layer=act_layer,
                                                 norm_layer=norm_layer,
                                                layer_scale=layer_scale,
                                                rpe=rpe,
                                                shift=shift
                                                  )


    def forward(self, x, hw_shape):
        def _inner_forward(x):
            if self.focal is not None:
                x = self.focal(x,hw_shape)
            if self.coarse is not None:
                x = self.coarse(x,hw_shape)
            return x  # (B,L,C)
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x

class PatchMerge(BaseModule):
    def __init__(self, in_dim=64, out_dim=128, norm=nn.LayerNorm,
                 with_cp=False, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.norm = norm(in_dim)
        self.down = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x, hw_shape):
        B, L, C = x.shape
        H, W = hw_shape
        x = self.norm(x)
        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 1, 2)
        x = self.down(x)
        hw_shape = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        return x, hw_shape  # (B,L/4,2C)

class BasicLayer(BaseModule):
    def __init__(self,
                 depth=3,
                 dilation=4,
                 seq_size=7,
                 dim=64,
                 out_dim=64,
                 kernel_size=7,
                 num_heads=4,
                 mlp_ratio=4,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 path_drop=(0., 0.),
                 mlp_drop=0.,
                 qkv_bias=True,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 layer_scale=1e-6,
                 rpe=True,
                 is_first=False,
                 is2=False,
                 is3=False,
                 with_cp=False,
                 init_cfg=None
                 ):
        super().__init__()
        self.dim = dim
        self.dilation = to_2tuple(dilation)
        self.seq_size = to_2tuple(seq_size)
        self.mlp_ratio = mlp_ratio
        self.is_first = is_first
        self.blocks = nn.ModuleList([])
        self.depth = depth
        for i in range(depth):
            if is_first:
                need_focal = True
            elif is2:
                need_focal = not (i + 1) % 4 == 0
            elif is3:
                need_focal = i % 4 == 0
            else:
                need_focal = False
            need_coarse = not need_focal
            self.blocks.append(
            FCMixer(
                            self.dilation,
                            self.seq_size,
                            dim = dim,
                            num_heads = num_heads,
                            qk_scale = qk_scale,
                            attn_drop = attn_drop,
                            proj_drop = proj_drop,
                            mlp_drop = mlp_drop,
                            path_drop = path_drop[i] if isinstance(path_drop, (list, tuple)) else path_drop,
                            qkv_bias = qkv_bias,
                            mlp_ratio = mlp_ratio,
                            act_layer = act_layer,
                            norm_layer = norm_layer,
                            kernel_size = kernel_size,
                            layer_scale=layer_scale,
                            rpe=rpe,
                            shift = i%2==0,
                            groups = 1 if is_first else num_heads,
                            focal = need_focal,
                            coarse = need_coarse,
                            with_cp=with_cp,
                            init_cfg=init_cfg
                            )
            )
        if out_dim is not None:
            self.downsample = PatchMerge(in_dim=dim, out_dim=out_dim, norm=norm_layer,
                                         with_cp=with_cp,init_cfg=init_cfg)
        else:
            self.downsample = None

    def forward(self, x, hw_shape):
        # x (B,L,C)
        for blk in self.blocks:
            x = blk(x, hw_shape)
        if self.downsample is not None:
            out = x
            out_hw_shape = hw_shape
            x, hw_shape = self.downsample(x, hw_shape)
        else:
            out = x
            out_hw_shape = hw_shape
        return x, hw_shape, out, out_hw_shape


class Patchify(BaseModule):
    def __init__(self,
                 in_dim=3,
                 embed_dim=96,
                 patch_size=4,
                 norm_layer=nn.LayerNorm,
                 init_cfg=None,
                 ):
        super().__init__(init_cfg=init_cfg)
        self.in_dim = in_dim
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.norm = norm_layer(embed_dim)
        self.patchify = nn.Conv2d(in_dim, embed_dim, kernel_size=8, stride=patch_size, padding=2)

    def forward(self, x):
        # x(B,C,H,W)
        H,W = x.shape[2:]
        patches = self.patchify(x)
        patches = patches.flatten(2).transpose(1, 2)
        patches = self.norm(patches)
        return patches, (H//self.patch_size, W//self.patch_size)

@MODELS.register_module()
class F2hNet(BaseModule):
    def __init__(self,
                 pretrained_img_size=224,
                 patch_size=4,
                 dilation = (8,4,2,1),
                 seq_size=(7,7,14,7),
                 depths=(3,4,12,4),
                 in_dim=3,
                 dims=(64,128,320,512),
                 kernel_sizes=(3,3,3,3),
                 num_heads=(2,4,10,16),
                 mlp_ratio=(4,4,4,4),
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 mlp_drop=0. ,
                 drop_path_rate=0.1,
                 qkv_bias=True,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 layer_scale=(None,None, 1., 1.),
                 rpe=True,
                 out_indices=(0, 1, 2, 3),
                 with_cp=False,
                 frozen_stages=-1,
                 init_cfg=None
                 ):
        super(F2hNet,self).__init__(init_cfg=init_cfg)
        self.frozen_stages = frozen_stages
        self.dims=dims
        self.out_indices = out_indices
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.stem = Patchify(patch_size=patch_size, in_dim=in_dim,
                             embed_dim=dims[0], norm_layer=norm_layer,init_cfg=init_cfg)
        self.layers = nn.ModuleList()
        cur = 0
        for i in range(len(depths)):
            self.layers.append(
                BasicLayer(
                    depth=depths[i],
                    dilation=dilation[i],
                    seq_size=seq_size[i],
                    dim=dims[i],
                    out_dim=dims[i+1] if i < len(depths) - 1 else None,
                    kernel_size=kernel_sizes[i],
                    num_heads=num_heads[i],
                    mlp_ratio=mlp_ratio[i],
                    qk_scale=qk_scale,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    mlp_drop=mlp_drop,
                    path_drop=dpr[cur: cur+depths[i]],
                    qkv_bias=qkv_bias,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    layer_scale=layer_scale[i],
                    rpe=rpe,
                    is_first= i==0,
                    is2= i==1,
                    is3 = i==2,
                    with_cp=with_cp,
                    init_cfg=init_cfg
                )
            )
            cur += depths[i]
        for i in out_indices:
            layer = norm_layer(dims[i])
            layer_name = f'norm{i}'
            self.add_module(layer_name, layer)

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(F2hNet, self).train(mode)
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patchify.eval()
            for param in self.patchify.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):

            if (i - 1) in self.out_indices:
                norm_layer = getattr(self, f'norm{i - 1}')
                norm_layer.eval()
                for param in norm_layer.parameters():
                    param.requires_grad = False

            m = self.stages[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self):
        logger = MMLogger.get_current_instance()
        if self.init_cfg is None:
            logger.warning(f'No pre-trained weights for '
                           f'{self.__class__.__name__}, '
                           f'training start from scratch')
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            ckpt = CheckpointLoader.load_checkpoint(
                self.init_cfg.checkpoint, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = OrderedDict()
            for k, v in _state_dict.items():
                if k.startswith('backbone.'):
                    state_dict[k[9:]] = v  # for checkpoints
                else:
                    state_dict[k] = v  # for pretrained in imagenet

            # strip prefix of state_dict
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            # interpolate position bias table if needed
            relative_position_bias_table_keys = [
                k for k in state_dict.keys()
                if 'relative_position_bias_table' in k
            ]
            for table_key in relative_position_bias_table_keys:
                table_pretrained = state_dict[table_key]
                table_current = self.state_dict()[table_key]
                L1, nH1 = table_pretrained.size()
                L2, nH2 = table_current.size()
                if nH1 != nH2:
                    logger.warning(f'Error in loading {table_key}, pass')
                elif L1 != L2:
                    S1 = int(L1 ** 0.5)
                    S2 = int(L2 ** 0.5)
                    table_pretrained_resized = F.interpolate(
                        table_pretrained.permute(1, 0).reshape(1, nH1, S1, S1),
                        size=(S2, S2),
                        mode='bicubic')
                    state_dict[table_key] = table_pretrained_resized.view(
                        nH2, L2).permute(1, 0).contiguous()

            # load state_dict
            self.load_state_dict(state_dict, False)

    def forward(self, x):
        x, hw_shape = self.stem(x)

        outs = []
        for i, layer in enumerate(self.layers):
            x, hw_shape, out, out_hw_shape = layer(x, hw_shape)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(out)
                out = out.view(-1, *out_hw_shape, self.dims[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        return outs

if __name__ == '__main__':
    x = torch.randn(1, 3, 1280, 800).cuda()
    model = F2hNet().cuda()
    y = model(x)
    print(y[-1].shape)
