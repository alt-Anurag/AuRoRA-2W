"""Compact DrivableNets-inspired model; not a reproduction of its E-ELAN model.

MobileNetV3-Large -> P3/P4 local IDFA -> bidirectional pyramid -> FCOS-style
boxes and a shared segmentation decoder with independent road/lane logits.
"""

import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

from .config import CLASS_NAMES
from .idfa import RollConditionedConv


def conv_bn(cin, cout, kernel=1):
    return nn.Sequential(nn.Conv2d(cin, cout, kernel, padding=kernel // 2, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU())


class SeparableConv(nn.Sequential):
    def __init__(self, channels):
        super().__init__(nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
                         nn.Conv2d(channels, channels, 1, bias=False),
                         nn.BatchNorm2d(channels), nn.ReLU())


class Fusion(nn.Module):
    def __init__(self, channels, count):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(count))
        self.conv = SeparableConv(channels)

    def forward(self, *features):
        weights = F.relu(self.weights)
        weights = weights / (weights.sum() + 1e-4)
        return self.conv(sum(w * f for w, f in zip(weights.unbind(), features)))


class BiPyramid(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.top4, self.top3 = Fusion(channels, 2), Fusion(channels, 2)
        self.bottom4, self.bottom5 = Fusion(channels, 3), Fusion(channels, 2)

    def forward(self, features):
        p3, p4, p5 = features
        t4 = self.top4(p4, F.interpolate(p5, size=p4.shape[-2:], mode="nearest"))
        t3 = self.top3(p3, F.interpolate(t4, size=p3.shape[-2:], mode="nearest"))
        o4 = self.bottom4(p4, t4, F.max_pool2d(t3, 2))
        o5 = self.bottom5(p5, F.max_pool2d(o4, 2))
        return t3, o4, o5


class DetectionHead(nn.Module):
    def __init__(self, channels, num_classes):
        super().__init__()
        self.cls_tower = nn.Sequential(SeparableConv(channels), SeparableConv(channels))
        self.box_tower = nn.Sequential(SeparableConv(channels), SeparableConv(channels))
        self.classifier = nn.Conv2d(channels, num_classes, 1)
        self.regressor = nn.Conv2d(channels, 4, 1)
        self.center = nn.Conv2d(channels, 1, 1)
        nn.init.constant_(self.classifier.bias, -math.log(99))

    def forward(self, x, stride):
        c, r = self.cls_tower(x), self.box_tower(x)
        return {"class_logits": self.classifier(c),
                "distances": F.softplus(self.regressor(r)) * stride,
                "centerness": self.center(r)}


class MultiTaskNet(nn.Module):
    def __init__(self, config, pretrained=None):
        super().__init__()
        self.config = dict(config)
        use_pretrained = config["pretrained_backbone"] if pretrained is None else pretrained
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if use_pretrained else None
        self.backbone = mobilenet_v3_large(weights=weights).features
        c = config["channels"]
        self.projections = nn.ModuleList(conv_bn(n, c) for n in (24, 40, 112, 960))
        # Both ablations contain the same learned 3x3 block at each selected level.
        self.alignment = nn.ModuleDict({str(level): RollConditionedConv(
            c, config["alignment"] == "idfa", config["residual_bound"])
            for level in config["idfa_levels"]})
        self.neck = nn.ModuleList(BiPyramid(c) for _ in range(config["neck_repeats"]))
        self.detector = DetectionHead(c, len(CLASS_NAMES))
        self.seg_decoder = nn.Sequential(conv_bn(4 * c, c), SeparableConv(c), SeparableConv(c))
        self.road_head, self.lane_head = nn.Conv2d(c, 1, 1), nn.Conv2d(c, 1, 1)

    def forward(self, image, imu):
        """RGB ImageNet-normalized BCHW; imu Bx2; outputs stay in image coordinates."""
        features = []
        for i, block in enumerate(self.backbone):
            image = block(image)
            if i in (3, 6, 12, 16):
                features.append(self.projections[len(features)](image))
        p2 = features[0]
        for level, block in self.alignment.items():
            features[int(level) - 2] = block(features[int(level) - 2], imu)
        pyramid = tuple(features[1:])
        for block in self.neck:
            pyramid = block(pyramid)
        seg = self.seg_decoder(torch.cat([p2] + [
            F.interpolate(f, size=p2.shape[-2:], mode="bilinear", align_corners=False)
            for f in pyramid], dim=1))
        return {"road_logits": self.road_head(seg), "lane_logits": self.lane_head(seg),
                "detection": [self.detector(f, s) for f, s in zip(pyramid, (8, 16, 32))]}


def flatten_detection(outputs):
    """Return class BxNxC, distances BxNx4, centerness BxN, centers Nx2, strides N."""
    classes, distances, centers, points, strides = [], [], [], [], []
    for output, stride in zip(outputs, (8, 16, 32)):
        logits = output["class_logits"]
        h, w = logits.shape[-2:]
        y, x = torch.meshgrid(torch.arange(h, device=logits.device, dtype=logits.dtype),
                              torch.arange(w, device=logits.device, dtype=logits.dtype), indexing="ij")
        points.append(torch.stack((x + .5, y + .5), -1).reshape(-1, 2) * stride)
        strides.append(logits.new_full((h * w,), stride))
        classes.append(logits.flatten(2).transpose(1, 2))
        distances.append(output["distances"].flatten(2).transpose(1, 2))
        centers.append(output["centerness"].flatten(1))
    return (torch.cat(classes, 1), torch.cat(distances, 1), torch.cat(centers, 1),
            torch.cat(points), torch.cat(strides))


def distance_to_boxes(points, distances):
    return torch.cat((points - distances[..., :2], points + distances[..., 2:]), dim=-1)


class ExportModel(nn.Module):
    """Static adapter; explicit portable IDFA keeps every learned sampling weight.

    Deep copying prevents export from changing a training model's runtime.
    """
    def __init__(self, model, portable_idfa=False):
        super().__init__()
        self.uses_imu = model.config["alignment"] == "idfa"
        if self.uses_imu and not portable_idfa:
            raise ValueError("IDFA custom backend requires explicit portable_idfa=True and export parity checks")
        self.model = copy.deepcopy(model)
        if self.uses_imu:
            for block in self.model.alignment.values():
                block.portable = True

    def forward(self, image, imu=None):
        if self.uses_imu and imu is None:
            raise ValueError("IDFA inference requires imu Bx2 [camera_scene_roll_rad, confidence]")
        if not self.uses_imu:
            imu = image.new_zeros((image.shape[0], 2))
        out = self.model(image, imu)
        cls, distances, center, points, _ = flatten_detection(out["detection"])
        return out["road_logits"], out["lane_logits"], cls, distance_to_boxes(points, distances), center
