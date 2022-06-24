import torch
from torch import nn, einsum
import numpy as np
from einops import rearrange, repeat


class CyclicShift(nn.Module):
    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def forward(self, x):
        return torch.roll(x, shifts=(self.displacement, self.displacement), dims=(1, 2))


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        #print('before res:',x.shape)
        x1 = self.fn(x, **kwargs)
        #print('after res:' , x1.shape)
        x1 = rearrange(x1, 'b h_n w_n (c p1 p2) ->b c (h_n p1) (w_n p2)', p1=1, p2=1)


        return x1 + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.norm(self.fn(x, **kwargs))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        x = rearrange(x, 'b c (h_n p1) (w_n p2) ->b h_n w_n (c p1 p2)', p1=1, p2=1)
        return self.net(x)


def create_mask(window_size, displacement, upper_lower, left_right):
    mask = torch.zeros(window_size ** 2, window_size ** 2)

    if upper_lower:
        mask[-displacement * window_size:, :-displacement * window_size] = float('-inf')
        mask[:-displacement * window_size, -displacement * window_size:] = float('-inf')

    if left_right:
        mask = rearrange(mask, '(h1 w1) (h2 w2) -> h1 w1 h2 w2', h1=window_size, h2=window_size)
        mask[:, -displacement:, :, :-displacement] = float('-inf')
        mask[:, :-displacement, :, -displacement:] = float('-inf')
        mask = rearrange(mask, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')

    return mask


def get_relative_distances(window_size):
    indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
    distances = indices[None, :, :] - indices[:, None, :]
    return distances

class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels=32, downscaling_factor=1):
        super().__init__()
        # self.conv1 = nn.Conv2d(in_channels, 16, 3, 1, 1)
        # self.relu = nn.ReLU(inplace=True)
        # self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        # self.conv3 = nn.Conv2d(32, out_channels*3, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.conv1_1 = nn.Conv2d(in_channels, 64, 3, 1, 1)
        self.conv2_1 = nn.Conv2d(64, 32, 3, 1, 1)
        self.conv3_1 = nn.Conv2d(32, out_channels, 3, 1, 1)
        self.conv1_2 = nn.Conv2d(in_channels, 64, 3, 1, 1)
        self.conv2_2 = nn.Conv2d(64, 32, 3, 1, 1)
        self.conv3_2 = nn.Conv2d(32, out_channels, 3, 1, 1)
        self.conv1_3 = nn.Conv2d(in_channels, 64, 3, 1, 1)
        self.conv2_3 = nn.Conv2d(64, 32, 3, 1, 1)
        self.conv3_3 = nn.Conv2d(32, out_channels, 3, 1, 1)
        self.patch_merge = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor, padding=0)
        self.downscaling_factor = downscaling_factor
    def forward(self, x):
        b, c, h, w = x.shape
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
        x1 = self.conv3_1(self.relu(self.conv2_1(self.relu(self.conv1_1(x)))))
        x2 = self.conv3_2(self.relu(self.conv2_2(self.relu(self.conv1_2(x)))))
        x3 = self.conv3_3(self.relu(self.conv2_3(self.relu(self.conv1_3(x)))))
        x = torch.cat((x1,x2,x3),1)
        x = x.permute(0, 2, 3, 1)
        return x

class WindowAttention(nn.Module):
    def __init__(self, in_channels, dim, heads, head_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        inner_dim = head_dim * heads
        self.patch_partition = PatchMerging(in_channels=in_channels, out_channels=dim)

        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted

        if self.shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift(-displacement)
            self.cyclic_back_shift = CyclicShift(displacement)
            self.upper_lower_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                             upper_lower=True, left_right=False), requires_grad=False)
            self.left_right_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                            upper_lower=False, left_right=True), requires_grad=False)

        #self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(window_size) + window_size - 1
            self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_size ** 2, window_size ** 2))

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        #经过cnn，得到qkv
        x = self.patch_partition(x)
        if self.shifted:
            x = self.cyclic_shift(x)
        b, n_h, n_w, _, h = *x.shape, self.heads

        qkv = x.chunk(3, dim=-1)
        nw_h = n_h // self.window_size
        nw_w = n_w // self.window_size

        q, k, v = map(
            lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d',
                                h=h, w_h=self.window_size, w_w=self.window_size), qkv)

        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale

        if self.relative_pos_embedding:
            dots += self.pos_embedding[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]
        else:
            dots += self.pos_embedding

        if self.shifted:
            dots[:, :, -nw_w:] += self.upper_lower_mask
            dots[:, :, nw_w - 1::nw_w] += self.left_right_mask

        attn = dots.softmax(dim=-1)

        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)
        out = rearrange(out, 'b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)',
                        h=h, w_h=self.window_size, w_w=self.window_size, nw_h=nw_h, nw_w=nw_w)
        out = self.to_out(out)

        if self.shifted:
            out = self.cyclic_back_shift(out)


        return out



class SwinBlock(nn.Module):
    def __init__(self, in_channels, dim, heads, head_dim, mlp_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        self.attention_block = Residual(PreNorm(dim, WindowAttention(in_channels=in_channels,
                                                                     dim=dim,
                                                                     heads=heads,
                                                                     head_dim=head_dim,
                                                                     shifted=shifted,
                                                                     window_size=window_size,
                                                                     relative_pos_embedding=relative_pos_embedding)))
        #self.mlp_block = Residual(PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim)))

    def forward(self, x):
        #print('before swinblock:',x.shape)
        x = self.attention_block(x)
        #print("after_attrntion:",x.shape)
        #x = self.mlp_block(x)
        #print("after_swin:", x.shape)
        return x





class StageModule(nn.Module):
    def __init__(self, in_channels, hidden_dimension, layers, downscaling_factor, num_heads, head_dim, window_size,
                 relative_pos_embedding):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'
        self.downscaling_factor = downscaling_factor
        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(nn.ModuleList([
                SwinBlock(in_channels=in_channels,dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
                SwinBlock(in_channels=in_channels,dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
            ]))

    def forward(self, x):
        for regular_block, shifted_block in self.layers:
            x = regular_block(x)
            x = shifted_block(x)
        return x
class merage(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x, y):
        x = torch.cat([x, y], axis=1)
        #x = torch.add(x, y).permute(0, 2, 3, 1)

        x = self.relu(self.conv(x))
        return x

class ResBlock(nn.Module):

    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=5, stride=1,
                               padding=2, bias=False)
        # self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=5, stride=1,
                               padding=2, bias=False)
        # self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)

        out += residual
        # out = self.relu(out)

        return out

class Head(nn.Module):
    """ Head consisting of convolution layers
    Extract features from corrupted images, mapping N3HW images into NCHW feature map.
    """

    def __init__(self, in_channels, out_channels=64, channels=32):
        super(Head, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(out_channels) if task_id in [0, 1, 5] else nn.Identity()
        # self.relu = nn.ReLU(inplace=True)
        self.resblock1 = ResBlock(out_channels)
        self.resblock2 = ResBlock(out_channels)
        self.conv2 = nn.Conv2d(out_channels, channels, kernel_size=1, stride=1,
                               padding=0, bias=False)

    def forward(self, x):
        out = self.conv1(x)
        # out = self.bn1(out)
        # out = self.relu(out)

        out = self.resblock1(out)
        out = self.resblock2(out)
        out = self.conv2(out)
        return out

class CURTransformer(nn.Module):
    def __init__(self, *, hidden_dim, layers, heads, channels=32, num_classes=1000, head_dim=8, window_size=28,
                 downscaling_factors=1, relative_pos_embedding=True,scale_factor=0):
        super().__init__()
        self.sr = scale_factor
        self.stage1 = StageModule(in_channels=channels, hidden_dimension=hidden_dim, layers=2,
                                  downscaling_factor=downscaling_factors, num_heads=heads, head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage2 = StageModule(in_channels=channels, hidden_dimension=hidden_dim, layers=2,
                                  downscaling_factor=downscaling_factors, num_heads=heads, head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage3 = StageModule(in_channels=channels, hidden_dimension=hidden_dim, layers=2,
                                  downscaling_factor=downscaling_factors, num_heads=heads, head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage4 = StageModule(in_channels=channels, hidden_dimension=hidden_dim, layers=2,
                                  downscaling_factor=downscaling_factors, num_heads=heads, head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)

        self.merage1 = merage(in_channels=channels * 2, out_channels=channels)
        self.merage2 = merage(in_channels=channels * 2, out_channels=channels)
        self.merage3 = merage(in_channels=channels * 2, out_channels=channels)
        self.up_stage1 = StageModule(in_channels=channels, hidden_dimension=hidden_dim, layers=2,
                                  downscaling_factor=downscaling_factors, num_heads=heads, head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.up_stage2 = StageModule(in_channels=channels, hidden_dimension=hidden_dim, layers=2,
                                  downscaling_factor=downscaling_factors, num_heads=heads, head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.up_stage3 = StageModule(in_channels=channels, hidden_dimension=hidden_dim, layers=2,
                                  downscaling_factor=downscaling_factors, num_heads=heads, head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.headsets = Head(3, 64, channels)
        self.tailsets = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, stride=1,
                           padding=1, bias=False),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(16, 3, kernel_size=3, stride=1,
                                                padding=1, bias=False)
                                      )
        #self.mlp_head = nn.Linear(96, 48)

        # up-sampling
        #assert 2 <= scale_factor <= 4
        # if scale_factor == 2 or scale_factor == 4:
        #     self.upscale = []
        #     for _ in range(scale_factor // 2):
        #         self.upscale.extend([nn.Conv2d(hidden_dim,hidden_dim* (2 ** 2), kernel_size=3, padding=1),
        #                              nn.PixelShuffle(2)])
        #     self.upscale = nn.Sequential(*self.upscale)
        # elif scale_factor == 3 :
        #     self.upscale = nn.Sequential(
        #         nn.Conv2d(hidden_dim, hidden_dim * (scale_factor ** 2), kernel_size=3, padding=1),
        #         nn.PixelShuffle(scale_factor)
        #     )

        self.conv2 = nn.Conv2d(channels, 3, 3, 1, 1)

    def forward(self, img):
        #b 3 hw->b 16 hw
        record = img
        x = self.headsets(img)
        #print(x.shape)
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        #print(x1.shape, x2.shape, x3.shape, x4.shape, x.shape)
        x = self.merage1(x4, x3)
        #print(x.shape)
        x = self.up_stage1(x)
        #print(x.shape)
        #print(x.shape)
        x = self.merage2(x, x2)
        #print(x.shape)
        x = self.up_stage2(x)
        #print(x.shape)
        x = self.merage3(x, x1)

        x = self.up_stage3(x)
        # if self.sr != 0:
        #     x = self.upscale(x)
        x = self.conv2(x)
        return x+img