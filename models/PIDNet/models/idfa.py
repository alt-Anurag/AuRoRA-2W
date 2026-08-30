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
        Calculates the exact per-pixel (dy, dx) coordinate shifts needed to 
        counteract the camera's physical roll angle, purely based on local 
        kernel tap rotation around its own center.
        
        roll_angles: Tensor of shape (B,) in degrees
        """
        # Convert roll angles to radians (shape: B, 1)
        theta = roll_angles.view(b, 1).to(device) * (math.pi / 180.0)
        
        # Precompute cos and sin
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        
        # Generate the k x k sampling grid around local kernel center (0,0)
        k = self.kernel_size
        pad = k // 2
        
        grid_y, grid_x = torch.meshgrid(
            torch.arange(-pad, pad + 1, device=device, dtype=torch.float32),
            torch.arange(-pad, pad + 1, device=device, dtype=torch.float32),
            indexing='ij'
        )
        
        # Flatten local grid to shape (1, k*k)
        grid_y_flat = grid_y.flatten().unsqueeze(0)
        grid_x_flat = grid_x.flatten().unsqueeze(0)
        
        # Calculate rotated kernel coordinates
        # x' = x * cos(theta) - y * sin(theta)
        # y' = x * sin(theta) + y * cos(theta)
        x_rot = grid_x_flat * cos_t - grid_y_flat * sin_t
        y_rot = grid_x_flat * sin_t + grid_y_flat * cos_t
        
        # The offset is the difference between rotated and original local coordinates
        dx = x_rot - grid_x_flat
        dy = y_rot - grid_y_flat
        
        # Interleave dy and dx for each tap.
        # Stage 1 empirical tests confirmed torchvision expects [dy, dx] order.
        offsets = torch.zeros(b, 2 * k * k, device=device, dtype=torch.float32)
        offsets[:, 0::2] = dy
        offsets[:, 1::2] = dx
        
        # Broadcast the fixed local offsets across all spatial locations (H, W)
        geom_offsets = offsets.view(b, 2 * k * k, 1, 1).expand(b, 2 * k * k, h, w)
        
        offsets = geom_offsets.contiguous()
        return offsets

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
    b, c, h, w = 2, 32, 64, 64
    dummy_x = torch.randn(b, c, h, w)
    dummy_angles = torch.tensor([0.0, 30.0]) # 0 degrees and 30 degrees lean
    
    idfa = IDFAModule(in_channels=c)
    offsets = idfa(dummy_x, dummy_angles)
    
    print(f"Input shape: {dummy_x.shape}")
    print(f"Output offsets shape: {offsets.shape} (Expected: B, 18, H, W)")
    
    # Validation 1: Zero-angle identity
    max_offset_0 = offsets[0].abs().max().item()
    print(f"Max offset for 0 degree lean: {max_offset_0:.4f}")
    assert max_offset_0 < 1e-5, "Zero degree lean should have zero geometric offset!"
    
    # Validation 2: Boundedness at 30 degrees
    max_offset_30 = offsets[1].abs().max().item()
    print(f"Max offset for 30 degree lean: {max_offset_30:.4f}")
    assert max_offset_30 < 1.5, f"Expected offset < 1.5px at 30deg, got {max_offset_30:.4f}"
    
    # Validation 3: Spatial uniformity
    corner_offset = offsets[1, :, 0, 0]
    center_offset = offsets[1, :, h//2, w//2]
    diff = (corner_offset - center_offset).abs().max().item()
    assert diff < 1e-5, "Offsets are not spatially uniform!"
    
    print("[OK] IDFA Module mathematical tests passed.")