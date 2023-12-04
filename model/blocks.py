from kornia.filters import filter2d
from torch import nn
import torch
class Posemb_LW(nn.Module):
    """
    Lightweight Posemb. We may not need a posemb as complex as before.
    """

    def __init__(self):
        super().__init__()
        self.posemb = nn.Sequential(
            nn.Conv2d(5, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LayerNorm([16, 7, 7]),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LayerNorm([32, 4, 4]),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        :param x: [n_scales x 18, 5, 56, 56]
        :return: [n_scales x 18, 32, 4, 4]
        """
        return self.posemb(x)


class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)

    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f[None, :, None]
        return filter2d(x, f, normalized=True)


class UpsampleBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, norm_layer, apply_blur=False, upsample=True):
        """"
        input: b c h w t
        If we use resnet features, h, w = 7, 7
        """
        super().__init__()
        blur = Blur() if apply_blur else nn.Identity()
        UpSample = nn.Upsample(scale_factor=(2, 2, 1), mode='trilinear') if upsample else nn.Identity()

        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=(3, 3, 1), stride=1, padding=(1, 1, 0)),
            norm_layer,
            nn.ReLU(inplace=True),
            blur,
            UpSample,
        )

    def forward(self, x):
        return self.block(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm_layer=nn.Identity(), apply_blur=False, upsample=True):
        super().__init__()
        blur = Blur() if apply_blur else nn.Identity()
        UpSample = nn.Upsample(scale_factor=(2, 2), mode='bilinear') if upsample else nn.Identity()

        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            norm_layer,
            nn.ReLU(inplace=True),
            blur,
            UpSample,
        )

    def forward(self, x):
        return self.block(x)
