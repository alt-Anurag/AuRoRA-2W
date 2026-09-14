"""Local rotated sampling prior retained from Phase 1, with bounded refinement.

This rotates convolution taps, NOT the image or a global feature coordinate frame.
roll_rad is the measured scene rotation in camera pixels: x right, y down,
positive clockwise. It must already include camera/device extrinsics and sync.
"""

import torch
from torch import nn
from torch.nn import functional as F


def portable_deform_conv3x3(x, offsets, weight):
    """Exact single-group stride-1, padding-1 DCN using standard ONNX ops.

    x is BCHW, offsets are Bx18xHxW interleaved (dy, dx), weight is C_out x
    C_in x 3 x 3. Bilinear sampling uses zero padding and pixel-center grids.
    Each tap is sampled and accumulated separately to avoid materializing a
    Bx9xCxHxW stack. The runtime may still schedule independent taps concurrently.
    This is an inference portability implementation, not an acceleration claim.
    """
    height, width = x.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(height, dtype=x.dtype, device=x.device),
                            torch.arange(width, dtype=x.dtype, device=x.device), indexing="ij")
    result = None
    for index in range(9):
        ky, kx = index // 3, index % 3
        sample_y = yy + (ky - 1) + offsets[:, 2 * index]
        sample_x = xx + (kx - 1) + offsets[:, 2 * index + 1]
        # align_corners=False also works for spatial dimensions equal to one.
        grid = torch.stack(((sample_x + .5) * (2. / width) - 1.,
                            (sample_y + .5) * (2. / height) - 1.), dim=-1)
        sampled = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        contribution = F.conv2d(sampled, weight[:, :, ky:ky+1, kx:kx+1])
        result = contribution if result is None else result + contribution
    return result


class RollConditionedConv(nn.Module):
    def __init__(self, channels, deformable=True, residual_bound=0.5):
        super().__init__()
        self.deformable = deformable
        self.portable = False  # Nonpersistent export switch; checkpoint keys stay compatible.
        self.residual_bound = float(residual_bound)
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(channels)
        if deformable:
            self.refine = nn.Sequential(
                nn.Conv2d(channels + 18, 32, 1), nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=1, groups=32), nn.ReLU(),
                nn.Conv2d(32, 18, 1),
            )
            # No outer model-wide reinitialization may overwrite this.
            nn.init.zeros_(self.refine[-1].weight)
            nn.init.zeros_(self.refine[-1].bias)
            y, x = torch.meshgrid(torch.arange(-1., 2.), torch.arange(-1., 2.), indexing="ij")
            self.register_buffer("tap_x", x.reshape(1, 9))
            self.register_buffer("tap_y", y.reshape(1, 9))

    def offsets(self, x, imu):
        """imu: Bx2 [roll_rad, confidence]; offsets: Bx18xHxW (dy,dx)."""
        angle = imu[:, :1].to(x)
        confidence = imu[:, 1:2].to(x).clamp(0, 1).reshape(-1, 1, 1, 1)
        c, s = angle.cos(), angle.sin()
        dx = self.tap_x * c - self.tap_y * s - self.tap_x
        dy = self.tap_x * s + self.tap_y * c - self.tap_y
        prior = torch.stack((dy, dx), dim=2).flatten(1).to(x)
        prior = prior[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
        residual = self.residual_bound * torch.tanh(self.refine(torch.cat((x, prior), dim=1)))
        return confidence * (prior + residual)

    def forward(self, x, imu):
        if self.deformable:
            # Float32 DCN is intentional for consistent offset/weight dtype under AMP.
            with torch.autocast(device_type=x.device.type, enabled=False):
                offsets = self.offsets(x.float(), imu.float())
                if self.portable:
                    y = portable_deform_conv3x3(x.float(), offsets, self.conv.weight.float())
                else:
                    from torchvision.ops import deform_conv2d
                    y = deform_conv2d(x.float(), offsets, self.conv.weight.float(), padding=(1, 1))
            y = y.to(x.dtype)
        else:
            y = self.conv(x)
        return F.relu(x + self.norm(y))
