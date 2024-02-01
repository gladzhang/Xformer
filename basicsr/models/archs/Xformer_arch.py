## Pytorch implementation of Xformer. This version use the redundant params 'input_resolution', which does ont affect results.
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange

from timm.models.layers import to_2tuple, trunc_normal_



def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

def drop_path(x, drop_prob: float = 0., training: bool = False):

    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows, window_size, h, w):
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        #  relative position encoding
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


## Spatial-wise window-based Transformer block (STB in this paper)
class SpatialTransformerBlock(nn.Module):
    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size=8,
                 shift_size=0,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, 'shift_size must in 0-window_size'

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer('attn_mask', attn_mask)

    def calculate_mask(self, x_size):
        # calculate mask for shift
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))  # 1 h w 1
        h_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x):
        b, c, h, w = x.shape
    
        x = to_3d(x)
        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)
        # padding
        size_par = self.window_size
        pad_l = pad_t = 0
        pad_r = (size_par - w % size_par) % size_par
        pad_b = (size_par - h % size_par) % size_par
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hd, Wd, _ = x.shape
        x_size = (Hd, Wd)

        if min(x_size) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(x_size)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)

        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, Hd, Wd)  # b h' w' c

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        # remove padding
        if pad_r > 0 or pad_b > 0:
            x = x[:, :h, :w, :].contiguous()
        x = x.view(b, h * w, c)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = to_4d(x, h, w)

        return x


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



## Channel-wise cross-covariance Transformer block (CTB in this paper)
class ChannelTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(ChannelTransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

    
    
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x



class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class XFormer(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3,
        img_size = 128,
        dim = 48,
        num_blocks = [2,4,4], 
        spatial_num_blocks = [2,4,4,6],
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        window_size=[16,16,16,16],
        drop_path_rate=0.1,
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False
    ):

        super(XFormer, self).__init__()
        self.alpha = 1
        self.beta = 1

        #########################################  Bidirectional connection unit (BCU)  ##################################### 
        self.Convs = nn.ModuleList()
        self.Convs.append(nn.Conv2d(dim * 2, dim * 2, kernel_size=3,padding=1,stride=1))
        self.Convs.append(nn.Conv2d(dim * 2 ** 2, dim * 2 ** 2, kernel_size=3,padding=1,stride=1))
        self.Convs.append(nn.Conv2d(dim * 2, dim * 2, kernel_size=3,padding=1,stride=1))
        self.Convs.append(nn.Conv2d(dim, dim, kernel_size=3,padding=1,stride=1))
        self.DWconvs = nn.ModuleList()
        self.DWconvs.append(nn.Conv2d(dim * 2, dim * 2, kernel_size=3,padding=1,stride=1,groups=dim * 2))
        self.DWconvs.append(nn.Conv2d(dim * 2 ** 2, dim * 2 ** 2, kernel_size=3,padding=1,stride=1,groups=dim * 2 ** 2))
        self.DWconvs.append(nn.Conv2d(dim * 2, dim * 2, kernel_size=3,padding=1,stride=1,groups=dim * 2))
        self.DWconvs.append(nn.Conv2d(dim, dim, kernel_size=3,padding=1,stride=1,groups=dim))
        #########################################  end  ##################################### 

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(spatial_num_blocks))]  # stochastic depth decay rule

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        #####################################  channel-wise branch  ##################################### 
        self.encoder_level1 = nn.Sequential(*[ChannelTransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[ChannelTransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[ChannelTransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[ChannelTransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[ChannelTransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1 
        self.reduce_chan_level1 = nn.Conv2d(int(dim*2), int(dim), kernel_size=1, bias=bias)
        self.decoder_level1 = nn.Sequential(*[ChannelTransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        #####################################  end  ##################################### 

        #####################################  spatial-wise branch  ##################################### 
        self.encoder1 = nn.Sequential(*[
            SpatialTransformerBlock(dim=dim, input_resolution=(img_size, img_size),
                             num_heads=heads[0], window_size=window_size[0], shift_size=0 if (i % 2 == 0) else window_size[0] // 2,
                             mlp_ratio=ffn_expansion_factor,
                             drop_path=dpr[sum(spatial_num_blocks[:0]):sum(spatial_num_blocks[:1])][i]
                             ) for i in range(spatial_num_blocks[0])])

        self.d1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder2 = nn.Sequential(*[
            SpatialTransformerBlock(dim=int(dim * 2 ** 1), input_resolution=(img_size//2, img_size//2),
                             num_heads=heads[1], window_size=window_size[1], shift_size=0 if (i % 2 == 0) else window_size[1] // 2,
                             mlp_ratio=ffn_expansion_factor,
                             drop_path=dpr[sum(spatial_num_blocks[:1]):sum(spatial_num_blocks[:2])][i]) for i in range(spatial_num_blocks[1])])

        self.d2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder3 = nn.Sequential(*[
            SpatialTransformerBlock(dim=int(dim * 2 ** 2),input_resolution=(img_size//4, img_size//4), num_heads=heads[2], window_size=window_size[2], shift_size=0 if (i % 2 == 0) else window_size[2] // 2,
                             mlp_ratio=ffn_expansion_factor,
                             drop_path=dpr[sum(spatial_num_blocks[:2]):sum(spatial_num_blocks[:3])][i]) for i in range(spatial_num_blocks[2])])

        self.d3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.s_latent = nn.Sequential(*[
            SpatialTransformerBlock(dim=int(dim * 2 ** 3), input_resolution=(img_size//8, img_size//8),num_heads=heads[3], window_size=window_size[3], shift_size=0 if (i % 2 == 0) else window_size[3] // 2,
                             mlp_ratio=ffn_expansion_factor,
                             drop_path=dpr[sum(spatial_num_blocks[:3]):sum(spatial_num_blocks[:4])][i]) for i in range(spatial_num_blocks[3])])

        self.u4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder3 = nn.Sequential(*[
            SpatialTransformerBlock(dim=int(dim * 2 ** 2), input_resolution=(img_size//4, img_size//4),num_heads=heads[2], window_size=window_size[2], shift_size=0 if (i % 2 == 0) else window_size[2] // 2,
                             mlp_ratio=ffn_expansion_factor,
                             drop_path=dpr[sum(spatial_num_blocks[:2]):sum(spatial_num_blocks[:3])][i]) for i in range(spatial_num_blocks[2])])

        self.u3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder2 = nn.Sequential(*[
            SpatialTransformerBlock(dim=int(dim * 2 ** 1), input_resolution=(img_size//2, img_size//2),num_heads=heads[1], window_size=window_size[1], shift_size=0 if (i % 2 == 0) else window_size[1] // 2,
                             mlp_ratio=ffn_expansion_factor,
                             drop_path=dpr[sum(spatial_num_blocks[:1]):sum(spatial_num_blocks[:2])][i]) for i in range(spatial_num_blocks[1])])

        self.u2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1 
        self.reduce1 = nn.Conv2d(int(dim * 2), int(dim), kernel_size=1, bias=bias)
        self.decoder1 = nn.Sequential(*[
            SpatialTransformerBlock(dim=int(dim), input_resolution=(img_size, img_size),num_heads=heads[0], window_size=window_size[0], shift_size=0 if (i % 2 == 0) else window_size[0] // 2,
                             mlp_ratio=ffn_expansion_factor,
                             drop_path=dpr[sum(spatial_num_blocks[:0]):sum(spatial_num_blocks[:1])][i]) for i in range(spatial_num_blocks[0])])
        #####################################  end  ##################################### 


        #####################################  refinement stage  ##################################### 
        self.refinement = nn.Sequential(*[ChannelTransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim*2**1), kernel_size=1, bias=bias)
            
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)


    def forward(self, inp_img):

        inp = self.patch_embed(inp_img)

        out_enc_level1 = self.encoder_level1(inp)
        out_enc1 = self.encoder1(inp)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        inp_enc2 = self.d1_2(out_enc1)
        shortcut = inp_enc_level2
        inp_enc_level2 = inp_enc_level2 + self.alpha * self.DWconvs[0](inp_enc2) # information fusion
        inp_enc2 = inp_enc2 + self.beta * self.Convs[0](shortcut) # information fusion
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        out_enc2 = self.encoder2(inp_enc2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        inp_enc3 = self.d2_3(out_enc2)
        shortcut = inp_enc_level3
        inp_enc_level3 = inp_enc_level3 + self.alpha * self.DWconvs[1](inp_enc3) # information fusion
        inp_enc3 = inp_enc3 + self.beta * self.Convs[1](shortcut) # information fusion
        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        out_enc3 = self.encoder3(inp_enc3)
        
        inp_enc_level4 = self.down3_4(out_enc_level3)
        inp_enc4 = self.d3_4(out_enc3)
        c_latent = self.s_latent(inp_enc_level4) 
        s_latent = self.s_latent(inp_enc4) 

        inp_dec_level3 = self.up4_3(c_latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        inp_dec3 = self.u4_3(s_latent)
        inp_dec3 = torch.cat([inp_dec3, out_enc3], 1)
        inp_dec3 = self.reduce3(inp_dec3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 
        out_dec3 = self.decoder3(inp_dec3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        inp_dec2 = self.u3_2(out_dec3)
        inp_dec2 = torch.cat([inp_dec2, out_enc2], 1)
        inp_dec2 = self.reduce2(inp_dec2)
        shortcut = inp_dec_level2
        inp_dec_level2 = inp_dec_level2 + self.alpha * self.DWconvs[2](inp_dec2) # information fusion
        inp_dec2 = inp_dec2 + self.beta * self.Convs[2](shortcut) # information fusion
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 
        out_dec2 = self.decoder2(inp_dec2)
        
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)
        inp_dec1 = self.u2_1(out_dec2)
        inp_dec1 = torch.cat([inp_dec1, out_enc1], 1)
        inp_dec1 = self.reduce1(inp_dec1)
        shortcut = inp_dec_level1
        inp_dec_level1 = inp_dec_level1 + self.alpha * self.DWconvs[3](inp_dec1) # information fusion
        inp_dec1 = inp_dec1 + self.beta * self.Convs[3](shortcut) # information fusion
        out_dec_level1 = self.decoder_level1(inp_dec_level1) 
        out_dec1 = self.decoder1(inp_dec1)
        
        x = torch.cat([out_dec_level1, out_dec1], 1)

        
        res = self.refinement(x)

        if self.dual_pixel_task:
            res = res + self.skip_conv(inp)
            res = self.output(res)
        else:
            res = self.output(res) + inp_img


        return res

