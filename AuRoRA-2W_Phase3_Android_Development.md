# AuRoRA-2W: Phase 3 Android Application Development Plan
*(Mobile Edge Deployment & Real-Time Sensor Fusion)*

## 📱 The Ultimate Vision
The final phase of AuRoRA-2W is to deploy the trained YOLOPv2 Tilt-Compensated model natively onto a smartphone (Android/iOS) mounted on a 2-wheeler. The application will process live camera frames alongside real-time IMU data to output drivable area, lane segmentation, and pothole detection at 30+ FPS directly on the edge.

---

## 🧠 1. Model Preparation & Quantization
Running a heavy, multi-task neural network on a smartphone requires strict optimization.
* **Base Parameters:** The YOLOPv2 baseline contains roughly **38 Million parameters** (~150 MB in FP32 format).
* **Quantization:** We will apply **INT8 Post-Training Quantization (PTQ)** during export to `.tflite` (Android) or CoreML (iOS). 
* **Payload Size:** This will shrink the model to a highly efficient **~38-40 MB** payload.
* **NPU Execution:** Modern smartphones (Snapdragon NPUs, Apple Neural Engine) have dedicated hardware for INT8 matrix multiplication, allowing this 40MB model to comfortably execute at 30+ FPS without draining the battery.

---

## 🚨 2. The Architectural Requirement (STN vs DCN)
This is the most critical constraint for Phase 3 deployment.
* **The Problem:** In Phase 1, our `IDFAModule` utilized Deformable Convolutions (DCN) to un-rotate features. DCN operations rely on dynamically generated offsets, which are notoriously incompatible with edge runtimes like TFLite and CoreML. Attempting to export a DCN-based model will result in operator fallback or outright conversion failure.
* **The Solution:** The model architecture **must** implement tilt-compensation using **Spatial Transformer Networks (STN)** (specifically `torch.nn.functional.grid_sample` with affine matrices). STNs perform the exact same rotational un-warping mathematically, but because they are standard affine transformations, they are **100% natively supported** by all mobile NPUs.

---

## ⏱️ 3. The Real-Time Application Loop

The application will leverage OS-level hardware APIs to bypass manual sensor fusion math, making the inference loop incredibly clean and efficient.

### Step A: The Sensor API (Zero-Math Roll Calculation)
You do not need to manually write complex Kalman filters to merge raw gyroscope and accelerometer data. 
* On **Android**, use the `Sensor.TYPE_ROTATION_VECTOR` API. 
* On **iOS**, use `CoreMotion`'s `CMAttitude`.
These OS-level APIs use optimized hardware sensor fusion to instantly hand you the gravity-aligned **Roll Angle** of the device as a clean floating-point scalar.

### Step B: The Sync Loop
At 30 frames per second, the application will simultaneously capture:
1. The current live camera frame (converted to a Tensor).
2. The current Roll Angle from the Rotation Vector API.

### Step C: The Inference
Both data points are injected directly into the TFLite runtime as parallel inputs:
* **Input 0:** `[1, 3, H, W]` (The image tensor)
* **Input 1:** `[1, 1]` (The scalar roll angle)

### Step D: The Output
The model’s STN layers will natively apply the affine rotation using the injected roll angle. The app will immediately receive perfectly leveled segmentation masks and bounding boxes for potholes, vehicles, and lanes, which are then rendered as a UI overlay on the live camera preview.
