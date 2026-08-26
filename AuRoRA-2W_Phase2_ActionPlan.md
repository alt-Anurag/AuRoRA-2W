# AuRoRA-2W: Phase 2 Action Plan
*(Autonomous Robotics on Rough Areas for 2-Wheelers)*

## 🎯 The Objective
Build the ultimate **AuRoRA-2W** architecture: A Multi-Task ADAS network designed specifically for unstructured Indian roads and 2-wheeler physics. 

**Required Outputs (Simultaneous):**
1. Drivable Area Segmentation (Pixel-level)
2. Lane Line Segmentation (Pixel-level)
3. Object Detection Bounding Boxes (Vehicles, **Potholes**, and **Speed Breakers**)

## 🏗️ The Architecture Strategy
Since the proprietary *DrivableNets* source code is inaccessible, we will use **YOLOPv2** or **HybridNets** (both Open Source) as our foundational architecture. These models inherently support all three tasks simultaneously. 

Your job is to fork their repository and inject our custom **Tilt-Compensation Logic** into their backbone, upgrading it from a flat-horizon 4-wheeler model into the **AuRoRA-2W** motorcycle model.

---

## 📂 Key Reference Files (From Current Phase 1 Repo)
To port the tilt-compensation logic into YOLOPv2, study these exact files in the current repository. **We have recently applied major bug fixes to these files that you MUST replicate in Phase 2:**

1. **`models/PIDNet/models/idfa.py` (THE MOST CRITICAL FILE)**
   * **What to look at:** The `IDFAModule` class.
   * **Why it matters:** This contains the core 2-wheeler tilt-compensation math. It takes the incoming IMU `roll_angles` and calculates a geometric offset field (inverse rotation coordinate grid) which it feeds into Deformable Convolution (DCN) layers to un-rotate the features. Your teammate MUST port this module into YOLOPv2's backbone.

2. **`scripts/train_aurora2w.py` (Synthetic Augmentation & OHEM Loss)**
   * **What to look at:** The `AuRoRA2W_FullModel` wrapper class, `forward()` function, and the loss initialization.
   * **Why it matters:** This contains the math for our **Synthetic Roll Augmentation** (`±30 degrees`). 
   * **🚨 CRITICAL PHASE 1 FIXES YOU MUST PORT:** 
     1. **The Padding Bug:** When using `TF.rotate` on the ground truth labels, you MUST set `fill=[255]` (the `ignore_index`). If you fill the empty rotated corners with `0`, the model hallucinates drivable road on the edges.
     2. **OHEM Loss:** We replaced standard Cross Entropy with **OhemCrossEntropyLoss**. This hyper-focuses on the hardest, most complex edge pixels (like curbs and lane lines). YOLOPv2's segmentation head must use this.

3. **`models/PIDNet/models/aurora2w.py` (Backbone Injection)**
   * **What to look at:** The modified `forward()` pass of the model architecture.
   * **Why it matters:** This demonstrates how we intercept the scalar `roll_angles` tensor and inject it directly into the feature-extraction backbone so the model can mathematically correlate the visual horizon with the IMU data.

4. **`scripts/test_video.py` (Inference & OpenCV Colors)**
   * **What to look at:** The inference loop and the `color[::-1]` patch.
   * **Why it matters:** Shows how real IMU data is passed as a tensor.
   * **🚨 CRITICAL PHASE 1 FIX YOU MUST PORT:** Our color palette is defined in **RGB**, but OpenCV (`cv2`) inherently uses **BGR**. When drawing segmentation maps over the video, you MUST reverse the color tuple `color[::-1]`. Without this, your sky will turn yellow and cars will turn red!

---

## 🚀 Execution Steps (For the Teammate)

### Step 1: Establish the Baseline
* Clone the open-source **YOLOPv2** repository.
* Map the **IDD Dataset** (Indian Driving Dataset) to the YOLO format. Make sure to isolate/map the classes for Potholes, Speed Breakers, and Vehicles.
* Run a short baseline training test on IDD (without tilt logic) just to ensure the loss decreases and the environment is stable.

### Step 2: Inject Tilt-Compensation into the Backbone
* Modify the `forward()` method of YOLOPv2's backbone (CSPDarknet/E-ELAN) to accept the `roll_angles` argument.
* Implement the rotational feature extraction logic referenced in `aurora2w.py` and the `IDFAModule`.

### Step 3: Implement Synthetic Roll Augmentation (⚠️ CRITICAL WARNING)
* Port the augmentation logic from `train_aurora2w.py` into the YOLO training loop. 
* **CRITICAL DIFFERENCE:** In Phase 1, we only had to rotate images and semantic masks. In Phase 2, YOLOPv2 uses **Bounding Boxes**. You cannot simply rotate a bounding box image mask. You must apply 2D rotational matrix math to the actual `[x_center, y_center, width, height]` coordinate tensors of the YOLO labels. If you fail to mathematically rotate the bounding box coordinates to match the rotated image, the Object Detection loss will explode.

### Step 4: Lane Segmentation Focus
* The user explicitly requested that Lane Segmentation cannot be skipped. 
* Heavily rotating images often destroys thin lane lines due to pixel interpolation loss. When applying `TF.rotate` to the Lane Line ground-truth masks, ensure you use `InterpolationMode.NEAREST` and potentially apply a morphological dilation/thickening step to the lane lines before rotating them, so they don't vanish.

### Step 5: Train and Validate
* Train the modified AuRoRA-2W model on the IDD dataset using OHEM Loss.
* Validate the inference by feeding it raw motorcycle footage combined with actual IMU roll data (from the NISER dataset) to prove the bounding boxes and segmentation stay glued to the road even when the bike leans 30 degrees into a turn.
