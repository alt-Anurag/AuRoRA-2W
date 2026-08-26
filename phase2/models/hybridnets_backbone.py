"""
AuRoRA-2W-IUR Backbone — EfficientNet-B3 with IDFA Tilt-Compensation Injection
================================================================================
Based on HybridNets' EfficientNet-B3 backbone.

KEY MODIFICATION:
  The standard EfficientNet processes features without any awareness of the
  camera's physical orientation. For a motorcycle that leans ±30°, this causes
  segmentation to drift off the road.

  We intercept the feature map at the END of Stage 2 (MBConv blocks, stride=8
  spatial resolution) — the same resolution used by the PIDNet P-Branch in Phase 1
  — and compute a Deformable Convolution offset field using the IDFAModule.

  This offset field is then used by a DCN layer to geometrically un-rotate the
  features before they are passed to deeper stages and the three task heads.

Architecture flow:
  Image (B, 3, H, W)
      ↓ stem_conv
      ↓ stage1 (stride 2)
      ↓ stage2 (stride 2)  →  [features at H/4, W/4, C=40]
      ↓ IDFA(roll_angles)  →  offset field (B, 18, H/4, W/4)
      ↓ DCN_layer          →  roll-compensated features
      ↓ stage3-7           →  continue normally
      ↓ BiFPN neck
      ↓ [Detection head | Drivable Area Seg head | Lane Seg head]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

from .idfa import IDFAModule


# ─────────────────────────────────────────────────────────────────
# EfficientNet-B3 building blocks (minimal, self-contained)
# ─────────────────────────────────────────────────────────────────

def _round_filters(filters, width_coeff):
    """Scale number of filters by width coefficient."""
    return int(filters * width_coeff)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_channels):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_channels, 1),
            Swish(),
            nn.Conv2d(reduced_channels, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class MBConv(nn.Module):
    """Mobile Inverted Bottleneck Conv (core EfficientNet block)."""
    def __init__(self, in_ch, out_ch, kernel_size, stride, expand_ratio, se_ratio=0.25):
        super().__init__()
        self.use_residual = (stride == 1 and in_ch == out_ch)
        mid_ch = in_ch * expand_ratio
        self.expand = (expand_ratio != 1)

        layers = []
        if self.expand:
            layers += [
                nn.Conv2d(in_ch, mid_ch, 1, bias=False),
                nn.BatchNorm2d(mid_ch, momentum=0.01, eps=1e-3),
                Swish()
            ]
        layers += [
            # Depthwise conv
            nn.Conv2d(mid_ch, mid_ch, kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch, momentum=0.01, eps=1e-3),
            Swish(),
            # SE
            SqueezeExcitation(mid_ch, max(1, int(in_ch * se_ratio))),
            # Pointwise
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch, momentum=0.01, eps=1e-3),
        ]
        self.block = nn.Sequential(*layers)

        # Stochastic depth (drop_path) placeholder — set rate in training
        self.drop_rate = 0.0

    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            if self.training and self.drop_rate > 0:
                # Stochastic depth
                keep = torch.rand(x.size(0), 1, 1, 1, device=x.device) >= self.drop_rate
                out = out * keep.float()
            out = out + x
        return out


def _make_stage(in_ch, out_ch, kernel_size, stride, expand_ratio, num_blocks):
    blocks = [MBConv(in_ch, out_ch, kernel_size, stride, expand_ratio)]
    for _ in range(num_blocks - 1):
        blocks.append(MBConv(out_ch, out_ch, kernel_size, 1, expand_ratio))
    return nn.Sequential(*blocks)


# ─────────────────────────────────────────────────────────────────
# AuRoRA-2W-IUR Backbone
# ─────────────────────────────────────────────────────────────────

# EfficientNet-B3 block config: (kernel, in_ch, out_ch, expand, num_blocks, stride)
# Channels after width_coeff=1.2 scaling
B3_CONFIG = [
    # stage1
    (3,  24,  24,  1, 2, 1),
    # stage2  ← IDFA injection point (output of this stage)
    (3,  24,  32,  6, 3, 2),
    # stage3
    (5,  32,  48,  6, 3, 2),
    # stage4
    (3,  48,  96,  6, 5, 2),
    # stage5
    (5,  96, 136,  6, 5, 1),
    # stage6
    (5, 136, 232,  6, 6, 2),
    # stage7
    (3, 232, 384,  6, 2, 1),
]

# BiFPN feature pyramid extraction points (C3, C4, C5 in HybridNets)
# These correspond to the outputs of stage3 (H/8), stage5 (H/16), stage7 (H/32)
BIFPN_STAGES = [2, 4, 6]  # 0-indexed into B3_CONFIG after stem


class AuRoRA2W_IUR_Backbone(nn.Module):
    """
    EfficientNet-B3 backbone with IDFA roll-compensation injection.

    Returns three feature maps at different scales for the BiFPN neck:
      P3: (B, 48,  H/8,  W/8)
      P4: (B, 136, H/16, W/16)
      P5: (B, 384, H/32, W/32)
    """

    def __init__(self, pretrained_weights=None):
        super().__init__()

        # ── Stem (stride 2) ─────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv2d(3, 40, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(40, momentum=0.01, eps=1e-3),
            Swish()
        )
        # After stem: (B, 40, H/2, W/2)

        # ── Stage 1 — no stride change (stride=1) ───────────────
        # in: (B, 40, H/2, W/2)  out: (B, 24, H/2, W/2)
        self.stage1 = _make_stage(40, 24, 3, 1, 1, 2)

        # ── Stage 2 — stride 2 ──────────────────────────────────
        # in: (B, 24, H/2, W/2)  out: (B, 32, H/4, W/4)  ← IDFA INJECTION HERE
        self.stage2 = _make_stage(24, 32, 3, 2, 6, 3)

        # ── IDFA Module (operates at H/4, W/4 resolution) ───────
        # in_channels=32 matches stage2 output
        self.idfa = IDFAModule(in_channels=32, kernel_size=3)

        # ── DCN layer to apply the IDFA offset ──────────────────
        self.dcn_post_idfa = DeformConv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.dcn_bn = nn.BatchNorm2d(32, momentum=0.01, eps=1e-3)
        self.dcn_act = Swish()

        # ── Stage 3 — stride 2 (P3 output at H/8) ───────────────
        # in: (B, 32, H/4, W/4)  out: (B, 48, H/8, W/8)
        self.stage3 = _make_stage(32, 48, 5, 2, 6, 3)

        # ── Stage 4 — stride 2 ──────────────────────────────────
        # in: (B, 48, H/8, W/8)  out: (B, 96, H/16, W/16)
        self.stage4 = _make_stage(48, 96, 3, 2, 6, 5)

        # ── Stage 5 — stride 1 (P4 output at H/16) ──────────────
        # in: (B, 96, H/16, W/16)  out: (B, 136, H/16, W/16)
        self.stage5 = _make_stage(96, 136, 5, 1, 6, 5)

        # ── Stage 6 — stride 2 ──────────────────────────────────
        # in: (B, 136, H/16, W/16)  out: (B, 232, H/32, W/32)
        self.stage6 = _make_stage(136, 232, 5, 2, 6, 6)

        # ── Stage 7 — stride 1 (P5 output at H/32) ──────────────
        # in: (B, 232, H/32, W/32)  out: (B, 384, H/32, W/32)
        self.stage7 = _make_stage(232, 384, 3, 1, 6, 2)

        self._init_weights()

        if pretrained_weights is not None:
            self._load_pretrained(pretrained_weights)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # DCN init
        nn.init.kaiming_normal_(self.dcn_post_idfa.weight, mode='fan_out')

    def _load_pretrained(self, weights_path):
        """Load EfficientNet-B3 ImageNet pretrained weights (shape-matched)."""
        state = torch.load(weights_path, map_location='cpu', weights_only=False)
        if 'state_dict' in state:
            state = state['state_dict']
        model_dict = self.state_dict()
        matched = {k: v for k, v in state.items()
                   if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(matched)
        self.load_state_dict(model_dict, strict=False)
        print(f"[Backbone] Loaded {len(matched)}/{len(model_dict)} pretrained parameters.")

    def forward(self, x, roll_angles=None):
        """
        Args:
            x: Input image tensor (B, 3, H, W)
            roll_angles: IMU roll angles in degrees (B,). Defaults to zeros.

        Returns:
            p3: Feature map at H/8  — fed to BiFPN
            p4: Feature map at H/16 — fed to BiFPN
            p5: Feature map at H/32 — fed to BiFPN
        """
        B = x.size(0)
        if roll_angles is None:
            roll_angles = torch.zeros(B, device=x.device)

        # ── Stem ────────────────────────────────────────────────
        x = self.stem(x)          # (B, 40, H/2, W/2)

        # ── Stage 1 ─────────────────────────────────────────────
        x = self.stage1(x)        # (B, 24, H/2, W/2)

        # ── Stage 2 ─────────────────────────────────────────────
        x = self.stage2(x)        # (B, 32, H/4, W/4)

        # ── IDFA Tilt Compensation ───────────────────────────────
        offsets = self.idfa(x, roll_angles)          # (B, 18, H/4, W/4)
        x = self.dcn_post_idfa(x, offsets)           # (B, 32, H/4, W/4)
        x = self.dcn_act(self.dcn_bn(x))

        # ── Stage 3 → P3 ────────────────────────────────────────
        x = self.stage3(x)        # (B, 48, H/8, W/8)
        p3 = x                    # ← BiFPN input

        # ── Stage 4 ─────────────────────────────────────────────
        x = self.stage4(x)        # (B, 96, H/16, W/16)

        # ── Stage 5 → P4 ────────────────────────────────────────
        x = self.stage5(x)        # (B, 136, H/16, W/16)
        p4 = x                    # ← BiFPN input

        # ── Stage 6 ─────────────────────────────────────────────
        x = self.stage6(x)        # (B, 232, H/32, W/32)

        # ── Stage 7 → P5 ────────────────────────────────────────
        x = self.stage7(x)        # (B, 384, H/32, W/32)
        p5 = x                    # ← BiFPN input

        return p3, p4, p5


if __name__ == '__main__':
    # Quick shape test — no GPU needed
    model = AuRoRA2W_IUR_Backbone()
    model.eval()

    dummy_img = torch.randn(2, 3, 384, 640)
    dummy_roll = torch.tensor([0.0, 15.0])  # 0 and 15 degree lean

    with torch.no_grad():
        p3, p4, p5 = model(dummy_img, dummy_roll)

    print(f"P3 (H/8):  {p3.shape}")   # Expected: (2, 48, 48, 80)
    print(f"P4 (H/16): {p4.shape}")   # Expected: (2, 136, 24, 40)
    print(f"P5 (H/32): {p5.shape}")   # Expected: (2, 384, 12, 20)
    print("[OK] Backbone forward pass successful.")
