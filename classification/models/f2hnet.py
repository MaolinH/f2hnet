# --------------------------------------------------------
# written By Maolin Huang
# --------------------------------------------------------
import torch
import torch.nn as nn
from timm.layers import to_2tuple, trunc_normal_, DropPath
from torch.nn.functional import scaled_dot_product_attention

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
    x = x.reshape(B, H*W, -1)
    return x  # (B,L,C)

class MLP(nn.Module):
    def __init__(self, in_dim, hide_dim, activation=nn.GELU, dropout=0.):
        super().__init__()
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

class Scale(nn.Module):
    def __init__(self, dim=64, layer_scale=1e-6, format = 'blc'):
        super().__init__()
        self.layer_scale = nn.Parameter(layer_scale*torch.ones(dim), requires_grad=True)
        self.format = format

    def forward(self, x):
        if self.format == 'blc':
            x = self.layer_scale*x
        elif self.format == 'bchw':
            x = self.layer_scale[:, None, None] * x
        return x


class FocalBlock(nn.Module):
    def __init__(self, resolution, dim=64, kernel_size=3, groups=1, ratio=4, path_drop=0., layer_scale=None):
        super().__init__()
        self.resolution = resolution
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

    def forward(self, x):
        H, W = self.resolution
        x = x.reshape(-1, H, W, self.dim).permute(0, 3, 1, 2)
        x = self.scale(x) + self.path_drop(self.conv(x))
        x = x.flatten(2).transpose(1, 2)
        return x

    def flops(self):
        H, W = self.resolution
        flops = 0
        flops += self.kernel_size ** 2 * self.dim * self.dim * H * W // self.groups
        flops += 2 * self.out_dim * self.dim * H * W
        flops += (2 * self.dim + self.out_dim) * H * W
        return flops


class Attention(nn.Module):
    def __init__(self,
                 seq_size,
                 dim=96,
                 num_heads=4,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 qkv_bias=True,
                 rpe=True
                ):
        super().__init__()
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

    def forward(self, x):
        B_, L, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B_, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # (3,B_,nH,L,head_dim)
        q, k, v = qkv
        # Training the model in this way might be faster
        #     # ----------------------------------------hand-written attention--------------------------------------------------
        #     attn = q @ k.transpose(-1, -2) * self.qk_scale
        #     if self.rpe:
        #         relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(L, L,
        #                                                                                                                -1)  # Wh*Ww,Wh*Ww,nH
        #         relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        #         attn = attn + relative_position_bias.unsqueeze(0)
    
        #     attn = self.attn_drop(attn)
        #     attn = self.softmax(attn)
        #     attn = (attn @ v).transpose(1, 2).reshape(B_, L, C)
        # ----------------------------------------FlashAttention--------------------------------------------------
        if self.rpe:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(L, L,
                                                                                                                   -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)  # nH, Wh*Ww, Wh*Ww
        else:
            relative_position_bias=None
        attn = scaled_dot_product_attention(query=q, key=k, value=v, attn_mask=relative_position_bias)
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

    def flops(self, N):
        flops = 0
        flops += 3 * self.dim * self.dim  # qkv
        flops += 2 * N * self.dim  # attn
        flops += self.dim  # softmax
        flops += self.dim * self.dim  # proj
        return flops

class Transformer(nn.Module):
    def __init__(self,
                 resolution,
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
                 rpe=True
                 ):
        super().__init__()
        self.resolution = resolution
        self.dilation = dilation
        self.seq_size = seq_size
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

    def forward(self, x):
        B, L, C = x.shape
        H, W = self.resolution
        shortcut = x
        x = self.norm_1(x)
        # -------------------------------------------------------------------------------------------------
        if H * W > self.seq_size[0] * self.seq_size[1]:
            x = x.view(B, H, W, C)
            seq_x = img2seq(x, self.dilation, self.seq_size)
            attn = self.attn(seq_x)
            x = seq2img(attn, self.dilation, self.seq_size,H=H, W=W)
        else:
            x = self.attn(x)
        x = self.scale_attn(shortcut) + self.path_drop(x)  # (B,L,C)
        # -------------------------------------------------------------------------------------------------
        x = self.scale_mlp(x) + self.path_drop(self.mlp(self.norm_2(x)))
        # -------------------------------------------------------------------------------------------------
        return x  # (B,L,C)

    def flops(self):
        H, W = self.resolution
        N = self.seq_size[0] * self.seq_size[1]  # number of windows
        flops = 0
        flops += self.attn.flops(N)  # Attention
        flops += 2 * self.mlp_hide_dim * self.dim  # MLP
        flops += 2 * self.dim  # LayerNorm
        return flops * H * W

class FCMixer(nn.Module):
    def __init__(self,
                 resolution,
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
                 groups=1,
                 focal=False,
                 coarse=False,

                 ):
        super().__init__()
        self.focal = None
        if focal:
            self.focal = FocalBlock(
                resolution=resolution,
                dim=dim,
                kernel_size=kernel_size,
                groups=groups,
                ratio=mlp_ratio,
                path_drop=path_drop,
                layer_scale=layer_scale,
            )
        self.coarse = None
        if coarse:
            self.coarse = Transformer(resolution=resolution,
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
                                                rpe=rpe
                                                  )


    def forward(self, x):
        if self.focal is not None:
            x = self.focal(x)
        if self.coarse is not None:
            x = self.coarse(x)
        return x  # (B,L,C)

    def flops(self):
        flops = 0
        if self.focal is not None:
            flops += self.focal.flops()
        if self.coarse is not None:
            flops += self.coarse.flops()
        return flops


class PatchMerge(nn.Module):
    def __init__(self, resolution, in_dim=64, out_dim=128, norm=nn.LayerNorm, ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.resolution = resolution
        self.norm = norm(in_dim)
        self.down = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        B, L, C = x.shape
        H, W = self.resolution
        x = self.norm(x)
        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 1, 2)
        x = self.down(x)
        x = x.flatten(2).transpose(1, 2)
        return x  # (B,L/4,2C)

    def flops(self):
        H, W = self.resolution
        L = (H // 2) * (W // 2)
        flops = 0
        flops += self.in_dim * H * W  # Norm
        flops += 9 * self.in_dim * self.out_dim * L  # down
        return flops


class BasicLayer(nn.Module):
    def __init__(self,
                 depth=3,
                 img_size=56,
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
                 ):
        super().__init__()
        self.resolution = to_2tuple(img_size)
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
                            self.resolution,
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
                            groups = 1 if is_first else num_heads,
                            focal = need_focal,
                            coarse = need_coarse,   )
            )
        if out_dim is not None:
            self.downsample = PatchMerge(resolution=self.resolution, in_dim=dim, out_dim=out_dim, norm=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        # x (B,L,C)
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x  # (B,L/4, 2C)

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class Patchify(nn.Module):
    def __init__(self,
                 img_size=224,
                 in_dim=3,
                 embed_dim=96,
                 patch_size=4,
                 norm_layer=nn.LayerNorm
                 ):
        super().__init__()
        self.in_dim = in_dim
        self.patch_size = patch_size
        self.img_size = img_size//patch_size
        self.embed_dim = embed_dim
        self.norm = norm_layer(embed_dim)
        self.patchify = nn.Conv2d(in_dim, embed_dim, kernel_size=8, stride=patch_size,padding=2)

    def forward(self, x):
        # x(B,C,H,W)
        patches = self.patchify(x)
        patches = patches.flatten(2).transpose(1, 2)
        patches = self.norm(patches)
        return patches  # (B,L,C)

    def flops(self):
        flops = 0
        flops +=  self.in_dim * self.embed_dim * 64
        flops += self.embed_dim
        return flops*self.img_size**2



class F2hNet(nn.Module):
    def __init__(self,
                 img_size=224,
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
                 layer_scale=(None, None, 1., 1.),
                 rpe=True,
                 num_classes=1000
                 ):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.stem = Patchify(img_size=img_size, patch_size=patch_size, in_dim=in_dim, embed_dim=dims[0], norm_layer=norm_layer)
        img_size = self.stem.img_size
        self.layers = nn.ModuleList()
        cur = 0
        for i in range(len(depths)):
            self.layers.append(
                BasicLayer(
                    depth=depths[i],
                    dilation=dilation[i],
                    img_size=img_size,
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
                )
            )
            cur += depths[i]
            img_size = img_size//2

        self.last_dim, self.num_cls = dims[-1], num_classes
        self.norm_head = norm_layer(dims[-1])
        self.cls_head = nn.Linear(dims[-1], num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.2)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        x = self.stem(x)
        for layer in self.layers:
            x = layer(x)
        return x

    def forward_head(self, x):
        x = self.norm_head(x)
        x = torch.mean(x, dim=1)
        x = self.cls_head(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.stem.flops()
        for layer in self.layers:
            flops += layer.flops()
        flops += self.last_dim * self.num_cls  # head
        return flops


def f2hnet_tiny(**kwargs):
    model = F2hNet(dims=(64, 128, 320, 512), num_heads=(2, 4, 10, 16), depths=(2, 4, 12, 4), 
                    dilation = (8,4,1,1), seq_size=(7,7,14,7), drop_path_rate=0.15,
                    **kwargs)
    return model

def f2hnet_small(**kwargs):
    model = F2hNet(dims=(64, 128, 320, 512), num_heads=(2, 4, 10, 16), depths=(3, 12, 24, 4), 
                    dilation = (8,4,1,1), seq_size=(7,7,14,7), drop_path_rate=0.3,
                    **kwargs)
    return model

def f2hnet_base(**kwargs):
    model = F2hNet(dims=(96, 192, 384, 768), num_heads=(3, 6, 12, 24), depths=(3, 12, 24, 4),
                    dilation = (8,4,1,1), seq_size=(7,7,14,7), drop_path_rate=0.5,
                    **kwargs)
    return model

if __name__ == '__main__':
    x = torch.randn(1, 3, 224, 224).cuda()
    model = f2hnet_tiny().cuda()
    print(model)
    s = 0
    for p in model.parameters():
        s += p.numel()
    print(f'total params: {s }')
    print(f'{model.flops()}')
    y = model(x)
