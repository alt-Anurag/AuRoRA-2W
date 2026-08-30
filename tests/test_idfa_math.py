import os
import sys
import math
import torch
import pytest
import torchvision.ops as ops

# Add the root directory and models/PIDNet to path to import IDFAModule
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'PIDNet'))
from models.idfa import IDFAModule

@pytest.fixture
def idfa_module():
    return IDFAModule(in_channels=32, kernel_size=3)

def test_boundedness(idfa_module):
    """
    Offsets never exceed the closed-form max for a 3x3 kernel at up to 30 degrees roll.
    For a 3x3 kernel, the max local offset (corner pixel) at 30 degrees is approx 0.732 pixels.
    We assert that no offset exceeds 2.0.
    """
    b, h, w = 1, 64, 64
    angles = torch.tensor([30.0])
    geom_offsets = idfa_module._generate_geometric_prior(b, h, w, angles, 'cpu')
    
    max_offset = geom_offsets.abs().max().item()
    assert max_offset <= 2.0, f"Expected offset <= 2.0 for a local 3x3 kernel deformation, but got {max_offset}"

def test_spatial_uniformity(idfa_module):
    """
    The geometric prior offset vector must be IDENTICAL at every (i, j) pixel location for a fixed roll angle.
    Since kernel rotation is a local operation, the offset from the grid sampling point should not depend
    on the global (x, y) location of the kernel.
    """
    b, h, w = 1, 64, 64
    angles = torch.tensor([15.0])
    geom_offsets = idfa_module._generate_geometric_prior(b, h, w, angles, 'cpu')
    
    # Compare everything to the top-left corner (0, 0)
    top_left_offset = geom_offsets[..., 0, 0].unsqueeze(-1).unsqueeze(-1)
    diff = (geom_offsets - top_left_offset).abs().max().item()
    assert diff < 1e-5, f"Offsets are not spatially uniform! Max diff across spatial locations: {diff}"

def test_zero_angle_identity(idfa_module):
    """
    0 degree roll must give exactly zero offset.
    """
    b, h, w = 1, 64, 64
    angles = torch.tensor([0.0])
    geom_offsets = idfa_module._generate_geometric_prior(b, h, w, angles, 'cpu')
    
    max_offset = geom_offsets.abs().max().item()
    assert max_offset < 1e-5, f"Expected zero offset for 0 degree roll, got {max_offset}"

def test_handedness_against_known_rotation(idfa_module):
    """
    Build an independent, manual ground-truth rotation using basic trigonometry 
    and confirm IDFA's first kernel tap offset matches it.
    """
    b, h, w = 1, 64, 64
    angle = 30.0
    theta = angle * math.pi / 180.0
    geom_offsets = idfa_module._generate_geometric_prior(b, h, w, torch.tensor([angle]), 'cpu')
    
    # First tap in PyTorch 3x3 kernel is (-1, -1) [top-left]
    # Expected target location after rotation around origin:
    expected_x_prime = -1 * math.cos(theta) - (-1) * math.sin(theta)
    expected_y_prime = -1 * math.sin(theta) + (-1) * math.cos(theta)
    
    expected_dx = expected_x_prime - (-1)
    expected_dy = expected_y_prime - (-1)
    
    # Channel 0 is dy, Channel 1 is dx for the first tap
    # We check the center of the image
    dy = geom_offsets[0, 0, h//2, w//2].item()
    dx = geom_offsets[0, 1, h//2, w//2].item()
    
    error_dy = abs(dy - expected_dy)
    error_dx = abs(dx - expected_dx)
    
    assert error_dy < 1e-5 and error_dx < 1e-5, (
        f"Manual tap 0 offset: (dy={expected_dy:.4f}, dx={expected_dx:.4f}), "
        f"IDFA returned: (dy={dy:.4f}, dx={dx:.4f})"
    )

def test_offset_grouping_matches_torchvision_convention(idfa_module):
    """
    Apply the generated offset field via torchvision.ops.deform_conv2d to a single bright pixel 
    on a black background with a near-identity kernel, and check that the sampled position moves 
    in the geometrically correct direction for a known roll angle.
    """
    b, c, h, w = 1, 1, 15, 15
    input_tensor = torch.zeros(b, c, h, w)
    
    # -------------------------------------------------------------------------
    # EMPIRICAL CONFIRMATION COMMENT:
    # PyTorch's torchvision.ops.deform_conv2d expects the offset channels in the
    # order [dy_0, dx_0, dy_1, dx_1, ... dy_8, dx_8].
    # By running deform_conv2d with IDFA's offsets, we can empirically confirm this.
    # We will evaluate the top-left tap (index 0), which is normally at (y-1, x-1).
    # At 90 degrees roll:
    # x_rot = x*cos(90) - y*sin(90) = -(-1) = 1
    # y_rot = x*sin(90) + y*cos(90) = -1
    # dx = x_rot - x = 2
    # dy = y_rot - y = 0
    # PyTorch deform_conv2d sampling position: p = p0 + p_n + \Delta p_n
    # p_n = (-1, -1). \Delta p_n = (dy=0, dx=2)
    # Target p = p0 + (-1, -1) + (0, 2) = p0 + (-1, 1)
    # -------------------------------------------------------------------------
    
    p0_y, p0_x = 7, 7
    target_y = p0_y - 1
    target_x = p0_x + 1
    
    # Place a bright pixel at the target location
    input_tensor[0, 0, target_y, target_x] = 1.0
    
    # Set weight for the top-left tap (index 0, i.e., weight[0, 0, 0, 0]) to 1.0
    weight = torch.zeros(1, 1, 3, 3)
    weight[0, 0, 0, 0] = 1.0 
    
    # 90 degrees roll
    angles = torch.tensor([90.0])
    geom_offsets = idfa_module._generate_geometric_prior(b, h, w, angles, 'cpu')
    
    out = ops.deform_conv2d(input_tensor, geom_offsets, weight, padding=1)
    
    val = out[0, 0, p0_y, p0_x].item()
    assert val > 0.9, f"Expected deform_conv2d to sample from (y-1, x+1) (val > 0.9), got {val}. Channel order [dy, dx] might be wrong."
