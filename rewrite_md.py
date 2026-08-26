new_content = \"\"\"# AuRoRA-2W: Phase 2 Action Plan
*(Autonomous Robotics on Rough Areas for 2-Wheelers)*

## ?? The Objective
Build the ultimate **AuRoRA-2W** architecture: A Multi-Task ADAS network designed specifically for unstructured Indian roads and 2-wheeler physics. 

**Required Outputs (Simultaneous):**
1. Drivable Area Segmentation (Pixel-level)
2. Lane Line Segmentation (Pixel-level)
3. Object Detection Bounding Boxes (Vehicles, **Potholes**, and **Speed Breakers**)

## ??? The Architecture Strategy
Since the proprietary *DrivableNets* source code is inaccessible, we will use **YOLOPv2** or **HybridNets** (both Open Source) as our foundational architecture. These models inherently support all three tasks simultaneously. 

Your job is to fork their repository and wrap their architecture in our custom **STN Tilt-Compensation Pipeline**, upgrading it from a flat-horizon 4-wheeler model into the **AuRoRA-2W** motorcycle model.

---

## ?? Key Reference Files (From Current Phase 1 Repo)
To port the tilt-compensation logic into YOLOPv2, study these exact files in the current repository. **We have recently applied major architectural shifts that you MUST replicate in Phase 2:**

1. **\scripts/train_aurora2w.py\ and \scripts/test_video.py\ (THE STN WRAPPER)**
   * **What to look at:** The \AuRoRA2W_FullModel\ wrapper class and the test inference loop.
   * **Why it matters:** We discovered that using Deformable Convolutions (DCN) internally for rotation causes catastrophic loss misalignment. Instead, we use a **Spatial Transformer Network (STN)** wrapper around the entire model. The pipeline is: 
     1. Un-rotate the incoming camera frame by \-roll_angle\ using STN matrix math (\TF.rotate\ or \grid_sample\).
     2. Pass the perfectly horizontal frame to the standard YOLO model.
     3. Take the YOLO output masks/boxes and mathematically re-rotate them by \+roll_angle\ to overlay onto the tilted video.
   * **CRITICAL FIX:** When STN rotates the image, the empty padded corners MUST be masked out or filled with \255\ (ignore index) in the loss function, otherwise the model will hallucinate classes in the black corners.

2. **OHEM Loss:** We replaced standard Cross Entropy with **OhemCrossEntropyLoss**. This hyper-focuses on the hardest, most complex edge pixels (like curbs and lane lines). YOLOPv2's segmentation head must use this.

---

## ?? Execution Steps (For the Teammate)

### Step 1: Establish the Baseline
* Clone the open-source **YOLOPv2** repository.
* Map the **IDD Dataset** (Indian Driving Dataset) to the YOLO format. Make sure to isolate/map the classes for Potholes, Speed Breakers, and Vehicles.
* Run a short baseline training test on IDD (without tilt logic) just to ensure the loss decreases and the environment is stable.

### Step 2: Implement the STN Tilt-Compensation Wrapper (CRITICAL)
* DO NOT attempt to modify the YOLO backbone with custom convolution layers for tilt! 
* Instead, write a wrapper class around the YOLO model. 
* During training (Synthetic Augmentation): Randomly generate a roll angle, rotate the labels, pass the image through the wrapper which un-rotates the image, runs YOLO, re-rotates the predictions, and calculates the loss against the tilted labels.
* **CRITICAL DIFFERENCE FOR YOLO:** YOLOPv2 uses **Bounding Boxes**. You cannot simply rotate a bounding box mask. You must apply 2D rotational matrix math to the actual \[x_center, y_center, width, height]\ coordinate tensors of the YOLO labels before calculating the detection loss!

### Step 3: Lane Segmentation Focus
* Heavily rotating images often destroys thin lane lines due to pixel interpolation loss. When applying \TF.rotate\ to the Lane Line ground-truth masks, ensure you use \InterpolationMode.NEAREST\ and potentially apply a morphological dilation/thickening step to the lane lines before rotating them, so they don't vanish.

### Step 4: Train and Validate
* Train the modified AuRoRA-2W model on the IDD dataset using OHEM Loss.
* Validate the inference by feeding it raw motorcycle footage combined with actual IMU roll data to prove the bounding boxes and segmentation stay glued to the road even when the bike leans 30 degrees into a turn.

---

## ??? Resolving Phase 2 Doubts: How did DrivableNets do it?

**Doubt:** *If the Indian Driving Dataset (IDD) is used, how do we train the model to segment lane lines and detect potholes, since IDD lacks annotations for these?*

**The Concrete Reality:**
You must build a **Composite Dataset**:
1. **Potholes (The RDD2022 Solution):** Download the **RDD2022 India Subset** (available on Kaggle/Roboflow) and merge its bounding box annotations into the IDD dataset pool during YOLO training.
2. **Lane Segmentation (BDD100K):** Inject the **BDD100K Lane Marking Dataset** into the training mix. YOLOPv2 was natively built for BDD100K. 

By mixing BDD100K (for lane lines) + IDD (for Indian vehicles) + RDD2022 (for Indian potholes), the model will learn to generalize and project lane boundaries even on unstructured Indian roads.

---

## ?? Future Outlook: Phase 3 (Mobile Edge Deployment)

**The Vision:** Running the final AuRoRA-2W model natively on a smartphone mounted on the bike. The app feeds live camera frames + real-time smartphone IMU data directly into the model.

**Is it feasible? YES, absolutely.** Here is the technical breakdown:

1. **Model Size & Parameters:**
   * **Storage Size:** By applying **INT8 Quantization** to YOLOPv2, the model size drops to **~38 - 40 MB**, which is incredibly lightweight for an Android app payload.

2. **Real-Time Performance:**
   * An INT8 quantized YOLOPv2 can comfortably run at **30+ FPS** on a modern smartphone NPU (Snapdragon / Apple A-series).

3. **The Sensor Fusion Integration:**
   * Smartphone APIs provide Gyroscope data at 100Hz+. 
   * The app will calculate the smartphone's real-time **Roll Angle**.
   * **Why the STN Architecture is perfect:** Because we shifted to the STN approach (instead of DCN), the model is **100% natively compatible** with Android and iOS NPUs. The Android app simply uses standard Android Bitmap Matrix APIs to mathematically un-rotate the camera frame by \-roll_angle\, passes the flat image to the TFLite YOLO model, and rotates the output masks back. It requires zero custom C++ operators!
\"\"\"
with open('AuRoRA-2W_Phase2_ActionPlan.md', 'w', encoding='utf-8') as f:
    f.write(new_content)
