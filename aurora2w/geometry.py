"""Pixel-space transforms shared by images, masks, boxes, and telemetry."""

import math
import numpy as np
import cv2


def transform_boxes(boxes, matrix, width, height):
    """XYXY boxes with continuous pixel edges; transform ALL four corners."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if len(boxes) == 0:
        return boxes.copy(), np.zeros(0, dtype=bool)
    corners = boxes[:, [0, 1, 2, 1, 2, 3, 0, 3]].reshape(-1, 4, 2)
    points = np.concatenate((corners, np.ones((*corners.shape[:-1], 1))), axis=-1)
    mapped = points @ np.asarray(matrix).T
    if mapped.shape[-1] == 3:
        mapped = mapped[..., :2] / mapped[..., 2:3]
    low, high = mapped.min(axis=1), mapped.max(axis=1)
    result = np.concatenate((low, high), axis=1).astype(np.float32)
    result[:, [0, 2]] = result[:, [0, 2]].clip(0, width)
    result[:, [1, 3]] = result[:, [1, 3]].clip(0, height)
    keep = (result[:, 2] - result[:, 0] >= 1) & (result[:, 3] - result[:, 1] >= 1)
    return result, keep


def letterbox(image, size):
    h, w = size
    ih, iw = image.shape[:2]
    scale = min(w / iw, h / ih)
    nw, nh = max(1, round(iw * scale)), max(1, round(ih * scale))
    left, top = (w - nw) // 2, (h - nh) // 2
    canvas = np.full((h, w, 3), 114, dtype=np.uint8)
    canvas[top:top + nh, left:left + nw] = cv2.resize(image, (nw, nh))
    valid = np.zeros((h, w), dtype=np.uint8)
    valid[top:top + nh, left:left + nw] = 1
    matrix = np.array([[nw / iw, 0, left], [0, nh / ih, top], [0, 0, 1]], np.float32)
    return canvas, valid, matrix


def roll_matrix(width, height, angle_rad):
    """Positive clockwise image rotation, around pixel-edge center."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    cx, cy = width / 2, height / 2
    return np.array([[c, -s, cx - c * cx + s * cy],
                     [s, c, cy - s * cx - c * cy], [0, 0, 1]], np.float32)


def pixel_center_matrix(edge_matrix):
    """Convert pixel-edge geometry to OpenCV integer pixel-center coordinates."""
    shift = np.array([[1, 0, .5], [0, 1, .5], [0, 0, 1]], np.float32)
    return np.linalg.inv(shift) @ edge_matrix @ shift


def warp_mask(mask, matrix, size, fill=255):
    return cv2.warpPerspective(mask, pixel_center_matrix(matrix), (size[1], size[0]),
                               flags=cv2.INTER_NEAREST, borderValue=fill)


def normalize_image(image):
    image = image.astype(np.float32) / 255
    image = (image - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
    return np.ascontiguousarray(image.transpose(2, 0, 1))
