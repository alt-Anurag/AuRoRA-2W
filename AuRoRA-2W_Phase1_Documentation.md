# AuRoRA-2W: Phase 1 Complete Documentation & Architecture Overview
*Autonomous Robotics on Rough Areas for 2-Wheelers*

## 1. Phase 1 Objective & Achievements
We successfully converted a standard flat-horizon semantic segmentation network into a robust, tilt-compensating, motorcycle ADAS architecture. 

**Achievements:**
* **Architectural Breakthrough (IDFAModule):** We debugged and perfected the **Image Deformation Field Attention (IDFA)** module. We discovered that the original code mistakenly attempted to apply a global spatial shift to the image. We rewrote the mathematical prior to calculate **local kernel rotation offsets**. This allows the Deformable Convolutions (DCN) to dynamically rotate their 3x3 receptive fields at every pixel based on the IMU roll angle, extracting tilt-invariant features while preserving the structural layout of the image!
* **Synthetic Roll Augmentation:** Designed dynamic data augmentation to artificially tilt training images up to 30 degrees to simulate motorcycle cornering physics.
* **OHEM Loss Implementation:** Upgraded the training pipeline to use Online Hard Example Mining (OHEM) to hyper-focus the network on challenging pixels (like lane edges).

## 2. Model Specifications & Training Pipeline
* **Base Architecture**: PIDNet-Small injected with custom IDFAModule blocks.
* **Task**: Semantic Segmentation (19 classes, highly focused on Drivable Area, Vehicles, and Lanes).
* **Training Dataset**: Cityscapes (High-resolution urban street scenes).
* **Batch Size**: 2.
* **Hardware & Cores**: 
  * **GPU**: 1x NVIDIA GPU (cuda:0). We utilize PyTorch \mp.autocast\ (Mixed Precision) for maximum computational speed and VRAM efficiency.
  * **CPU Cores**: \
um_workers=0\. We restrict dataloading to a single core (the main thread) to bypass severe Windows OS multiprocessing overhead constraints.

## 3. The Core Challenge: Tilt, Shake, and Roll Compensation
A car camera stays perfectly horizontal, but a motorcycle aggressively leans into turns. If a standard Convolutional Neural Network (CNN) is fed a tilted image, its structural priors fail.

Our solution is the **IDFAModule**. Instead of relying on a black-box neural network to figure out how to handle tilt, we inject a hardcoded mathematical prior into the Deformable Convolutions. 

### Exact Code Implementation: \models/PIDNet/models/idfa.py\
We rewrote \_generate_geometric_prior\ to properly calculate local kernel rotation offsets for the 3x3 grid:

\\\python
    def _generate_geometric_prior(self, b, h, w, roll_angles, device):
        # 1. Define the standard 3x3 convolution grid (relative to center)
        k = self.kernel_size
        grid_y, grid_x = torch.meshgrid(
            torch.arange(-(k//2), k//2 + 1, device=device, dtype=torch.float32),
            torch.arange(-(k//2), k//2 + 1, device=device, dtype=torch.float32),
            indexing='ij'
        )
        
        # Flatten to shape (9,)
        grid_x = grid_x.reshape(-1)
        grid_y = grid_y.reshape(-1)
        
        # Convert roll angles to radians (shape: B, 1)
        theta = roll_angles.view(b, 1).to(device) * (math.pi / 180.0)
        
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        
        # 2. Rotate the local kernel grid
        x_rot = grid_x.unsqueeze(0) * cos_t - grid_y.unsqueeze(0) * sin_t
        y_rot = grid_x.unsqueeze(0) * sin_t + grid_y.unsqueeze(0) * cos_t
        
        # 3. Calculate the local offsets (Target - Source)
        dx = x_rot - grid_x.unsqueeze(0)
        dy = y_rot - grid_y.unsqueeze(0)
        
        # 4. Interleave dx and dy for the 18 channels (B, 18)
        offsets = torch.stack([dx, dy], dim=2).view(b, self.num_offsets)
        
        # 5. Broadcast to the entire spatial dimension (B, 18, H, W)
        geom_offsets = offsets.view(b, self.num_offsets, 1, 1).expand(b, self.num_offsets, h, w)
        
        return geom_offsets
\\\

## Why this architecture preserves research novelty
By combining hardcoded Affine Matrix Math with the learnable offsets of Deformable Convolutions, the network learns to adapt its receptive field dynamically without losing spatial layout. This is a massive leap forward from standard CNNs, completely preserving the core novelty of the research paper.
