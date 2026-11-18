import torch
from torch import nn
import torch.nn.functional as F


class DownSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels * s_factor, 1, stride=1, padding=0))

    def forward(self, x):
        x = self.down(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=s_factor, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels, in_channels // 2, 1, stride=1, padding=0))

    def forward(self, x):
        x = self.up(x)
        return x


class Focus(nn.Module):
    def __init__(self, in_channels, n_feats, ksize=1, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels * 4, n_feats, ksize, stride, padding=ksize // 2)

    def forward(self, x):
        patch_top_left = x[..., ::2, ::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat((patch_top_left, patch_bot_left, patch_top_right, patch_bot_right,), dim=1, )
        return self.conv(x)


class ChannelAttention(nn.Module):
    def __init__(self, n_feats, reduction):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(n_feats, n_feats // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats // reduction, n_feats, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.ca(self.avg_pool(x))
        return x * y


class CAB(nn.Module):
    """MPRNet CAB: PReLU body, ReLU channel attention, residual addition."""
    def __init__(self, n_feats, reduction):
        super().__init__()
        self.conv_1 = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1, bias=False)
        self.act = nn.PReLU()
        self.conv_2 = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1, bias=False)
        self.ca = ChannelAttention(n_feats, reduction)

    def forward(self, x):
        residual = self.conv_2(self.act(self.conv_1(x)))
        return x + self.ca(residual)


class Encoder(nn.Module):
    def __init__(self, in_channels=3, n_feats=32, reduction=8):
        super().__init__()
        self.num_blocks = [1, 1, 1, 1, 1]
        # 640x640
        self.emb_in = Focus(in_channels, n_feats, ksize=3, stride=1)
        # 320x320
        self.e1 = nn.Sequential(*[CAB(n_feats, reduction) for _ in range(self.num_blocks[0])])
        self.down2 = DownSample(n_feats, s_factor=2)
        # 160x160
        self.e2 = nn.Sequential(*[CAB(n_feats * 2, reduction) for _ in range(self.num_blocks[1])])
        self.down3 = DownSample(n_feats * 2, s_factor=2)
        # 80x80
        self.e3 = nn.Sequential(*[CAB(n_feats * 4, reduction) for _ in range(self.num_blocks[2])])
        self.down4 = DownSample(n_feats * 4, s_factor=2)
        # 40x40
        self.e4 = nn.Sequential(*[CAB(n_feats * 8, reduction) for _ in range(self.num_blocks[3])])
        self.down5 = DownSample(n_feats * 8, s_factor=2)
        # 20x20
        self.e5 = nn.Sequential(*[CAB(n_feats * 16, reduction) for _ in range(self.num_blocks[4])])

    def forward(self, x):
        e1 = self.e1(self.emb_in(x))
        e2 = self.e2(self.down2(e1))
        e3 = self.e3(self.down3(e2))
        e4 = self.e4(self.down4(e3))
        e5 = self.e5(self.down5(e4))
        return e1, e2, e3, e4, e5


class MultiScalePromptPool(nn.Module):
    """DSPM: 1 global, 4 regional and 16 local learnable blocks."""
    def __init__(self, embed_dim, prompt_dim, prompt_size):
        super().__init__()
        if prompt_size < 4 or prompt_size % 4:
            raise ValueError('prompt_size must be a positive multiple of four.')
        self.global_param = nn.Parameter(torch.rand(1, 1, prompt_dim, prompt_size, prompt_size))
        self.region_param = nn.Parameter(torch.rand(1, 4, prompt_dim, prompt_size // 2, prompt_size // 2))
        self.local_param = nn.Parameter(torch.rand(1, 16, prompt_dim, prompt_size // 4, prompt_size // 4))
        self.linear_layer = nn.Linear(embed_dim, 21)
        self.conv3x3 = nn.Conv2d(prompt_dim * 3 + embed_dim, embed_dim, 3, padding=1, bias=False)

    @staticmethod
    def assemble(blocks, grid):
        b, _, c, h, w = blocks.shape
        return blocks.reshape(b, grid, grid, c, h, w).permute(0, 3, 1, 4, 2, 5).reshape(b, c, grid*h, grid*w)

    def forward(self, x):
        if x.shape[-2:] != self.global_param.shape[-2:]:
            raise ValueError('Feature spatial size must match the native prompt resolution; build the model with the matching input_size.')
        weights = self.linear_layer(x.mean(dim=(-2, -1))).softmax(dim=-1)
        weights = weights[:, :, None, None, None]
        maps = [self.assemble(weights[:, :1] * self.global_param, 1),
                self.assemble(weights[:, 1:5] * self.region_param, 2),
                self.assemble(weights[:, 5:] * self.local_param, 4)]
        return self.conv3x3(torch.cat([*maps, x], dim=1))


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class PromptFusionAttention(nn.Module):
    """Spatial soft queries attend to four hard-prompt key/value tokens."""
    def __init__(self, embed_dim, hard_dim=512, num_attr=4, num_heads=4):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError('Prompt channels must be divisible by attention heads.')
        self.num_attr, self.num_heads = num_attr, num_heads
        self.scale = (embed_dim // num_heads) ** -0.5
        self.to_kv = nn.Linear(hard_dim, embed_dim * num_attr * 2)
        self.norm_kv = nn.LayerNorm(embed_dim * 2)
        self.norm_q = ChannelLayerNorm(embed_dim)
        self.q_pointwise = nn.Conv2d(embed_dim, embed_dim, 1)
        self.q_dwconv = nn.Conv2d(embed_dim, embed_dim, 3, padding=1, groups=embed_dim)

    def forward(self, soft_prompt, hard_prompt):
        b, c, h, w = soft_prompt.shape
        q = self.q_dwconv(self.q_pointwise(self.norm_q(soft_prompt)))
        q = q.reshape(b, self.num_heads, c // self.num_heads, h*w).transpose(-2, -1)
        kv = self.norm_kv(self.to_kv(hard_prompt).reshape(b, self.num_attr, 2*c))
        k, v = kv.chunk(2, dim=-1)
        k = k.reshape(b, self.num_attr, self.num_heads, c // self.num_heads).transpose(1, 2)
        v = v.reshape(b, self.num_attr, self.num_heads, c // self.num_heads).transpose(1, 2)
        attention = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        return (attention @ v).transpose(-2, -1).reshape(b, c, h, w)


class GDFN(nn.Module):
    """Gated depthwise-convolution feed-forward refinement."""
    def __init__(self, channels, expansion=2.66):
        super().__init__()
        hidden = int(channels * expansion)
        self.project_in = nn.Conv2d(channels, 2*hidden, 1, bias=False)
        self.dwconv = nn.Conv2d(2*hidden, 2*hidden, 3, padding=1, groups=2*hidden, bias=False)
        self.project_out = nn.Conv2d(hidden, channels, 1, bias=False)

    def forward(self, x):
        left, right = self.dwconv(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(left) * right)


class PromptFusionBlock(nn.Module):
    def __init__(self, embed_dim, hard_dim=512, num_attr=4, num_heads=4):
        super().__init__()
        self.attn = PromptFusionAttention(embed_dim, hard_dim, num_attr, num_heads)
        self.ffn = GDFN(embed_dim)

    def forward(self, soft_prompt, hard_prompt):
        return self.ffn(soft_prompt + self.attn(soft_prompt, hard_prompt))


class Decoder(nn.Module):
    def __init__(self, out_channel=3, n_feats=32, reduction=8, use_prompt_fusion=False, input_size=640):
        super().__init__()
        self.use_prompt_fusion = use_prompt_fusion
        self.up3 = UpSample(n_feats * 4, 2)
        self.d2 = CAB(n_feats * 2, reduction)
        self.up2 = UpSample(n_feats * 2, 2)
        self.d1 = CAB(n_feats, reduction)
        self.up1 = UpSample(n_feats, 2)
        self.emb_out = nn.Sequential(CAB(n_feats // 2, reduction), nn.Conv2d(n_feats // 2, out_channel, 3, padding=1))
        if use_prompt_fusion:
            self.prompt_pool1 = MultiScalePromptPool(n_feats * 2, n_feats * 2, input_size // 4)
            self.prompt_pool2 = MultiScalePromptPool(n_feats, n_feats, input_size // 2)
            self.fusion_block1 = PromptFusionBlock(n_feats * 2)
            self.fusion_block2 = PromptFusionBlock(n_feats)

    def prepare_features(self, encoder_features, hard_prompt):
        if not self.use_prompt_fusion:
            return tuple(encoder_features), []
        soft1 = self.prompt_pool1(encoder_features[1])
        soft2 = self.prompt_pool2(encoder_features[0])
        guided = list(encoder_features)
        guided[1] = guided[1] + self.fusion_block1(soft1, hard_prompt)
        guided[0] = guided[0] + self.fusion_block2(soft2, hard_prompt)
        return tuple(guided), [soft1, soft2]

    def forward(self, x, guided_features, image):
        d2 = self.d2(self.up3(x) + guided_features[1])
        d1 = self.d1(self.up2(d2) + guided_features[0])
        return self.emb_out(self.up1(d1)) + image
