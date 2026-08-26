"""
AuRoRA-2W-IUR Multi-Task Heads
================================
Three parallel heads that share the BiFPN feature pyramid:

  1. Detection Head    — YOLO-style anchored bounding box prediction
                         Classes: vehicle, pothole, speed_breaker
  2. Drivable Area Head — Binary/multi-class pixel segmentation
  3. Lane Line Head    — Binary pixel segmentation (lane vs. background)

BiFPN Neck:
  Takes P3 (H/8), P4 (H/16), P5 (H/32) from the backbone and
  fuses them bi-directionally for 3 iterations.

Input channel sizes (from EfficientNet-B3 backbone):
  P3: 48 channels  @ H/8
  P4: 136 channels @ H/16
  P5: 384 channels @ H/32
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────
# BiFPN Neck
# ─────────────────────────────────────────────────────────────────

class BiFPNLayer(nn.Module):
    """Single BiFPN iteration (fast normalised fusion)."""

    def __init__(self, fpn_ch):
        super().__init__()
        self.fpn_ch = fpn_ch
        eps = 1e-4

        # TD (top-down) weighted fusion weights
        self.w_td_p4 = nn.Parameter(torch.ones(2))
        self.w_td_p3 = nn.Parameter(torch.ones(2))

        # BU (bottom-up) weighted fusion weights
        self.w_bu_p4 = nn.Parameter(torch.ones(3))
        self.w_bu_p5 = nn.Parameter(torch.ones(2))

        self.eps = eps
        self.act = nn.ReLU()

        # Depthwise separable convs after each fusion
        self.conv_td_p4 = self._dw_conv(fpn_ch)
        self.conv_td_p3 = self._dw_conv(fpn_ch)
        self.conv_bu_p4 = self._dw_conv(fpn_ch)
        self.conv_bu_p5 = self._dw_conv(fpn_ch)

    @staticmethod
    def _dw_conv(ch):
        return nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False),
            nn.Conv2d(ch, ch, 1, bias=False),
            nn.BatchNorm2d(ch, momentum=0.01, eps=1e-3),
            nn.ReLU(inplace=True),
        )

    def _fast_norm(self, w):
        w = self.act(w)
        return w / (w.sum() + self.eps)

    def forward(self, p3, p4, p5):
        # ── Top-Down ────────────────────────────────────────────
        # P4_td = (w1*P4 + w2*resize(P5)) / (w1+w2)
        w = self._fast_norm(self.w_td_p4)
        p4_td = self.conv_td_p4(
            w[0] * p4 + w[1] * F.interpolate(p5, size=p4.shape[-2:], mode='nearest')
        )
        # P3_td = (w1*P3 + w2*resize(P4_td)) / (w1+w2)
        w = self._fast_norm(self.w_td_p3)
        p3_out = self.conv_td_p3(
            w[0] * p3 + w[1] * F.interpolate(p4_td, size=p3.shape[-2:], mode='nearest')
        )

        # ── Bottom-Up ───────────────────────────────────────────
        # P4_out = (w1*P4 + w2*P4_td + w3*resize(P3_out)) / sum
        w = self._fast_norm(self.w_bu_p4)
        p4_out = self.conv_bu_p4(
            w[0] * p4 + w[1] * p4_td +
            w[2] * F.interpolate(p3_out, size=p4.shape[-2:], mode='nearest')
        )
        # P5_out = (w1*P5 + w2*resize(P4_out)) / sum
        w = self._fast_norm(self.w_bu_p5)
        p5_out = self.conv_bu_p5(
            w[0] * p5 + w[1] * F.interpolate(p4_out, size=p5.shape[-2:], mode='nearest')
        )

        return p3_out, p4_out, p5_out


class BiFPN(nn.Module):
    """3-iteration BiFPN with lateral convs to project backbone channels → fpn_ch."""

    def __init__(self, in_channels=(48, 136, 384), fpn_ch=64, num_iters=3):
        super().__init__()

        # Lateral projections: map backbone channels to uniform fpn_ch
        self.lat_p3 = self._lat_conv(in_channels[0], fpn_ch)
        self.lat_p4 = self._lat_conv(in_channels[1], fpn_ch)
        self.lat_p5 = self._lat_conv(in_channels[2], fpn_ch)

        self.layers = nn.ModuleList([BiFPNLayer(fpn_ch) for _ in range(num_iters)])

    @staticmethod
    def _lat_conv(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch, momentum=0.01, eps=1e-3),
            nn.ReLU(inplace=True)
        )

    def forward(self, p3, p4, p5):
        p3 = self.lat_p3(p3)
        p4 = self.lat_p4(p4)
        p5 = self.lat_p5(p5)
        for layer in self.layers:
            p3, p4, p5 = layer(p3, p4, p5)
        return p3, p4, p5


# ─────────────────────────────────────────────────────────────────
# Segmentation Head (shared design for Drivable Area + Lane Lines)
# ─────────────────────────────────────────────────────────────────

class SegmentationHead(nn.Module):
    """
    Upsamples P3 (H/8) → full resolution using bilinear upsampling + conv.
    Output: (B, num_classes, H, W)
    """

    def __init__(self, fpn_ch, num_classes, upsample_scale=8):
        super().__init__()
        self.upsample_scale = upsample_scale

        self.decode = nn.Sequential(
            # Feature refinement at P3 resolution
            nn.Conv2d(fpn_ch, fpn_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_ch, fpn_ch // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_ch // 2),
            nn.ReLU(inplace=True),
            # Final class logits
            nn.Conv2d(fpn_ch // 2, num_classes, 1),
        )

    def forward(self, p3, target_size):
        """
        p3: (B, fpn_ch, H/8, W/8)
        target_size: (H, W) of the original input image
        Returns: (B, num_classes, H, W)
        """
        x = self.decode(p3)
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x


# ─────────────────────────────────────────────────────────────────
# Detection Head (YOLO-style)
# ─────────────────────────────────────────────────────────────────

class DetectionHead(nn.Module):
    """
    Single-scale anchor-free detection head operating on all three FPN levels.
    Output per level: (B, num_anchors * (5 + num_classes), H_level, W_level)
      where 5 = [tx, ty, tw, th, obj_conf]

    For simplicity we use 3 anchors per level (standard YOLO).

    Classes:
      0: vehicle
      1: pothole
      2: speed_breaker
    """

    def __init__(self, fpn_ch, num_classes=3, num_anchors=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        out_ch = num_anchors * (5 + num_classes)

        # Shared convolutional feature extractor per scale
        self.shared_conv = nn.Sequential(
            nn.Conv2d(fpn_ch, fpn_ch * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_ch * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_ch * 2, fpn_ch * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_ch * 2),
            nn.ReLU(inplace=True),
        )

        # One prediction conv per FPN level (P3, P4, P5)
        self.pred_p3 = nn.Conv2d(fpn_ch * 2, out_ch, 1)
        self.pred_p4 = nn.Conv2d(fpn_ch * 2, out_ch, 1)
        self.pred_p5 = nn.Conv2d(fpn_ch * 2, out_ch, 1)

    def forward(self, p3, p4, p5):
        """
        Returns:
            preds: list of 3 tensors, each (B, num_anchors*(5+C), H_level, W_level)
        """
        f3 = self.shared_conv(p3)
        f4 = self.shared_conv(p4)
        f5 = self.shared_conv(p5)

        return [self.pred_p3(f3), self.pred_p4(f4), self.pred_p5(f5)]


# ─────────────────────────────────────────────────────────────────
# Full Multi-Task Model
# ─────────────────────────────────────────────────────────────────

class AuRoRA2W_IUR(nn.Module):
    """
    Full AuRoRA-2W-IUR multi-task model.

    Inputs:
        x: (B, 3, H, W) image
        roll_angles: (B,) IMU roll angles in degrees

    Outputs (dict):
        'det':      list of 3 tensors — detection predictions per FPN scale
        'drive_seg': (B, num_drive_classes, H, W) — drivable area logits
        'lane_seg':  (B, 2, H, W)               — lane line logits (binary)
    """

    # Default anchor sizes (in pixels) for [P3, P4, P5]
    # Tuned for 640×384 input (IDD/BDD100K standard)
    ANCHORS = [
        [(10, 13), (16, 30), (33, 23)],    # P3 — small objects (signs, potholes)
        [(30, 61), (62, 45), (59, 119)],   # P4 — medium objects (bikes, cars)
        [(116, 90), (156, 198), (373, 326)] # P5 — large objects (trucks, buses)
    ]

    def __init__(
        self,
        num_det_classes=3,      # vehicle, pothole, speed_breaker
        num_drive_classes=3,    # background, drivable, alt-drivable
        fpn_ch=64,
        pretrained_backbone=None,
    ):
        super().__init__()

        from .hybridnets_backbone import AuRoRA2W_IUR_Backbone
        self.backbone = AuRoRA2W_IUR_Backbone(pretrained_weights=pretrained_backbone)
        self.neck = BiFPN(in_channels=(48, 136, 384), fpn_ch=fpn_ch, num_iters=3)
        self.det_head = DetectionHead(fpn_ch, num_classes=num_det_classes)
        self.drive_seg_head = SegmentationHead(fpn_ch, num_classes=num_drive_classes)
        self.lane_seg_head = SegmentationHead(fpn_ch, num_classes=2)  # lane / no-lane

    def forward(self, x, roll_angles=None):
        H, W = x.shape[-2], x.shape[-1]

        # ── Backbone (with IDFA tilt comp) ──────────────────────
        p3, p4, p5 = self.backbone(x, roll_angles)

        # ── BiFPN Neck ──────────────────────────────────────────
        p3, p4, p5 = self.neck(p3, p4, p5)

        # ── Three Task Heads ─────────────────────────────────────
        det_preds = self.det_head(p3, p4, p5)
        drive_logits = self.drive_seg_head(p3, target_size=(H, W))
        lane_logits = self.lane_seg_head(p3, target_size=(H, W))

        return {
            'det': det_preds,
            'drive_seg': drive_logits,
            'lane_seg': lane_logits,
        }


def get_aurora2w_iur_model(
    num_det_classes=3,
    num_drive_classes=3,
    fpn_ch=64,
    pretrained_backbone=None,
):
    """Factory function — call this from training and inference scripts."""
    return AuRoRA2W_IUR(
        num_det_classes=num_det_classes,
        num_drive_classes=num_drive_classes,
        fpn_ch=fpn_ch,
        pretrained_backbone=pretrained_backbone,
    )


if __name__ == '__main__':
    # Shape validation — no GPU needed
    model = get_aurora2w_iur_model()
    model.eval()

    dummy = torch.randn(2, 3, 384, 640)
    rolls = torch.tensor([0.0, -20.0])

    with torch.no_grad():
        out = model(dummy, rolls)

    print("Detection predictions:")
    for i, d in enumerate(out['det']):
        print(f"  P{i+3}: {d.shape}")
    print(f"Drivable area: {out['drive_seg'].shape}")
    print(f"Lane lines:    {out['lane_seg'].shape}")
    print("[OK] Full AuRoRA-2W-IUR forward pass successful.")
