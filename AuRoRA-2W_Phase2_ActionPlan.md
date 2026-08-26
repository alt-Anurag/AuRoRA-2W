# AuRoRA-2W: Phase 2 Action Plan
*(Autonomous Robotics on Rough Areas for 2-Wheelers)*

## ?? The Objective
Build the ultimate **AuRoRA-2W** architecture: A Multi-Task ADAS network designed specifically for unstructured Indian roads and 2-wheeler physics. 

**Required Outputs (Simultaneous):**
1. Drivable Area Segmentation (Pixel-level)
2. Lane Line Segmentation (Pixel-level)
3. Object Detection Bounding Boxes (Vehicles, **Potholes**, and **Speed Breakers**)

## ??? The Architecture Strategy
Since the proprietary *DrivableNets* source code is inaccessible, we will use **YOLOPv2** or **HybridNets** (both Open Source) as our foundational architecture. These models inherently support all three tasks simultaneously. 

Your job is to fork their repository and inject our custom **Tilt-Compensation Logic (IDFAModule)** into their backbone, upgrading it from a flat-horizon 4-wheeler model into the **AuRoRA-2W** motorcycle model.

---

## ?? Key Reference Files (From Current Phase 1 Repo)
To port the tilt-compensation logic into YOLOPv2, study these exact files in the current repository. **We have recently applied major bug fixes to these files that you MUST replicate in Phase 2:**

1. **\models/PIDNet/models/idfa.py\ (THE MOST CRITICAL FILE)**
   * **What to look at:** The \IDFAModule\ class.
   * **Why it matters:** This contains the core 2-wheeler tilt-compensation math. It takes the incoming IMU \oll_angles\ and calculates a geometric offset field which it feeds into Deformable Convolution (DCN) layers. 
   * **CRITICAL FIX:** The original code mistakenly calculated a *global coordinate shift*. We have rewritten \_generate_geometric_prior\ to properly calculate **local kernel rotation offsets**. This dynamically rotates the 3x3 receptive field at every pixel, allowing the network to extract rotation-invariant features while preserving the structural layout of the image!

2. **\scripts/train_aurora2w.py\ (Synthetic Augmentation & OHEM Loss)**
   * **What to look at:** The \AuRoRA2W_FullModel\ wrapper class.
   * **Why it matters:** This contains the math for our **Synthetic Roll Augmentation** (\±30 degrees\). 

---

## ?? Execution Steps (For the Teammate)

### Step 1: Establish the Baseline
* Clone the open-source **YOLOPv2** repository.
* Map the **IDD Dataset** (Indian Driving Dataset) to the YOLO format. Make sure to isolate/map the classes for Potholes, Speed Breakers, and Vehicles.

### Step 2: Inject Tilt-Compensation into the Backbone
* Modify the \orward()\ method of YOLOPv2's backbone to accept the \oll_angles\ argument.
* Implement the rotational feature extraction logic referenced in \urora2w.py\ and the fixed \IDFAModule\.

### Step 3: Implement Synthetic Roll Augmentation (?? CRITICAL WARNING)
* Port the augmentation logic from \	rain_aurora2w.py\ into the YOLO training loop. 
* **CRITICAL DIFFERENCE:** In Phase 1, we only had to rotate images and semantic masks. In Phase 2, YOLOPv2 uses **Bounding Boxes**. You cannot simply rotate a bounding box image mask. You must apply 2D rotational matrix math to the actual \[x_center, y_center, width, height]\ coordinate tensors of the YOLO labels.

### Step 4: Lane Segmentation Focus
* The user explicitly requested that Lane Segmentation cannot be skipped. 
* Heavily rotating images often destroys thin lane lines. When applying \TF.rotate\ to the Lane Line ground-truth masks, ensure you use \InterpolationMode.NEAREST\.

### Step 5: Train and Validate
* Train the modified AuRoRA-2W model on the IDD dataset using OHEM Loss.
* Validate the inference by feeding it raw motorcycle footage combined with actual IMU roll data.
