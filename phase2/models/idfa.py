"""
IDFAModule — Image Deformation Field Attention
================================================
Ported unchanged from Phase 1: models/PIDNet/models/idfa.py

Generates roll-compensated geometric offset fields for Deformable Convolutions.
Takes absolute roll angle from the IMU, calculates a closed-form inverse
rotation coordinate offset grid, and refines it with a lightweight residual
convolutional network to handle lens distortion, pitch, and vibrations.
"""

import torch
import torch.nn as nn
import math


class IDFAModule(nn.Module):
    """
    Image Deformation Field Attention (IDFA)

    Generates roll-compensated geometric offset fields for Deformable Convolutions.
    It takes an absolute roll angle from the IMU, calculates a closed-form inverse
    rotation coordinate offset grid, and refines it with a lightweight residual
    convolutional network to handle lens distortion, pitch, and vibrations.
    """
    def __init__(self, in_channels, kernel_size=3):
        super(IDFAModule, self).__init__()
        self.kernel_size = kernel_size
        self.num_offsets = 2 * (kernel_size ** 2)  # 18 for a 3x3 kernel

        # Residual Refinement Network
        # Concatenates input image features (C) + geometric prior offsets (18)
        self.refinement_conv = nn.Sequential(
            nn.Conv2d(in_channels + self.num_offsets, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.num_offsets, kernel_size=1, bias=True)
        )

        # Initialize the final conv layer to zero so it initially relies purely on the math prior
        nn.init.constant_(self.refinement_conv[-1].weight, 0)
        nn.init.constant_(self.refinement_conv[-1].bias, 0)

    def _generate_geometric_prior(self, b, h, w, roll_angles, device):
        """
        Calculates the exact per-pixel (dx, dy) coordinate shifts needed to
        counteract the camera's physical roll angle.

        roll_angles: Tensor of shape (B,) in degrees
        """
        # Create a meshgrid representing pixel coordinates
        y, x = torch.meshgrid(
            torch.arange(h, device=device, dtype=torch.float32),
            torch.arange(w, device=device, dtype=torch.float32),
            indexing='ij'
        )

        # Center the coordinates (Origin at image center)
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        x_centered = x - cx
        y_centered = y - cy

        # Convert roll angles to radians (shape: B, 1, 1)
        theta = roll_angles.view(b, 1, 1).to(device) * (math.pi / 180.0)

        # Precompute cos and sin
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        # Calculate rotated coordinates
        # x' = x * cos(theta) - y * sin(theta)
        # y' = x * sin(theta) + y * cos(theta)
        x_rot = x_centered.unsqueeze(0) * cos_t - y_centered.unsqueeze(0) * sin_t
        y_rot = x_centered.unsqueeze(0) * sin_t + y_centered.unsqueeze(0) * cos_t

        # The offset is the difference between rotated and original coordinates
        # Delta = Target - Source
        dx = x_rot - x_centered.unsqueeze(0)
        dy = y_rot - y_centered.unsqueeze(0)

        # Stack into shape (B, 2, H, W)
        base_offsets = torch.cat([dx.unsqueeze(1), dy.unsqueeze(1)], dim=1)

        # DCN expects offsets for every point in the kernel.
        # By repeating the central offset 9 times, we shift the entire 3x3 kernel
        # cleanly to the un-rotated physical location.
        # Shape becomes (B, 18, H, W)
        geom_offsets = base_offsets.repeat(1, self.kernel_size ** 2, 1, 1)

        return geom_offsets

    def forward(self, x, roll_angles):
        """
        x: Input feature map (B, C, H, W)
        roll_angles: Absolute IMU roll angles in degrees (B,)

        Returns: Final deformable convolution offset field (B, 18, H, W)
        """
        b, c, h, w = x.shape
        device = x.device

        # 1. Closed-Form Physical Math Prior
        geom_offsets = self._generate_geometric_prior(b, h, w, roll_angles, device)

        # 2. Artificial Intelligence Residual Refinement
        # Concatenate image features and geometric prior to understand context
        features_with_prior = torch.cat([x, geom_offsets], dim=1)
        residual_offsets = self.refinement_conv(features_with_prior)

        # 3. Final Output Field
        final_offsets = geom_offsets + residual_offsets

        return final_offsets


# ---------------------------------------------------------
# Test block to validate the math locally
# ---------------------------------------------------------
if __name__ == "__main__":
    b, c, h, w = 2, 40, 64, 64
    dummy_x = torch.randn(b, c, h, w)
    dummy_angles = torch.tensor([0.0, 30.0])  # 0 degrees and 30 degrees lean

    idfa = IDFAModule(in_channels=c)
    offsets = idfa(dummy_x, dummy_angles)

    print(f"Input shape: {dummy_x.shape}")
    print(f"Output offsets shape: {offsets.shape} (Expected: B, 18, H, W)")

    # Validation: If lean is 0, the geometric offset should be 0 everywhere
    max_offset_0 = offsets[0].abs().max().item()
    print(f"Max offset for 0 degree lean: {max_offset_0:.4f}")
    assert max_offset_0 < 1e-5, "Zero degree lean should have zero geometric offset!"
    print("[OK] IDFA Module mathematical test passed.")
