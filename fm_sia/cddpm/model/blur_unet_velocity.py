"""A convolutional velocity U-Net with low-pass filtering before downsampling."""
import torch
from torch import nn
from torch.nn import functional as F

from model.conditional_fno import ConditionalFNO


class BlurPool(nn.Module):
    def __init__(self, channels):
        super().__init__()
        taps = torch.tensor([1., 2., 1.]) / 4
        self.register_buffer("kernel", torch.outer(taps, taps).expand(channels, 1, 3, 3).contiguous())

    def forward(self, x):
        return F.conv2d(x, self.kernel, stride=2, padding=1, groups=x.shape[1])


class ConvPair(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.SiLU(),
                         nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.SiLU())


def haar_split(x):
    """Return cell means and full-grid detail; border cells use their available pixels."""
    low = F.avg_pool2d(x, 2, stride=2, ceil_mode=True, count_include_pad=False)
    constant = low.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
    return low, x - constant[..., :x.shape[-2], :x.shape[-1]]


class BlurUNetVelocity(nn.Module):
    def __init__(self, width, haar_details=False):
        super().__init__()
        self.haar_details = haar_details
        self.fine = ConvPair(65, width)
        if not haar_details:
            self.down_fine = BlurPool(width)
            self.down_middle = BlurPool(2 * width)
        self.middle = ConvPair(width, 2 * width)
        self.coarse = ConvPair(2 * width, 4 * width)
        self.decode_middle = ConvPair(6 * width, 2 * width)
        self.decode_fine = ConvPair(3 * width, width)
        self.output = nn.Conv2d(width, 1, 1)

    @staticmethod
    def with_skip(coarse, skip):
        up = F.interpolate(coarse, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat((skip, up), dim=1)

    def forward(self, x):
        fine = self.fine(x)
        low, fine = haar_split(fine) if self.haar_details else (self.down_fine(fine), fine)
        middle = self.middle(low)
        low, middle = haar_split(middle) if self.haar_details else (self.down_middle(middle), middle)
        coarse = self.coarse(low)
        middle = self.decode_middle(self.with_skip(coarse, middle))
        return self.output(self.decode_fine(self.with_skip(middle, fine)))


class ConditionedBlurUNet(ConditionalFNO):
    def __init__(self, *, hidden_channels, haar_details=False, **kwargs):
        super().__init__(hidden_channels=hidden_channels, **kwargs)
        # Construct the parent first to match the random stream for its SIA head.
        del self.fno
        with torch.random.fork_rng(devices=[]):
            self.fno = BlurUNetVelocity(hidden_channels, haar_details=haar_details)
