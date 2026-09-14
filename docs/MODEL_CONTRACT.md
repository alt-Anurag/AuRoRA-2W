# Aurora Android model bundle, version 2

The Android application imports a ZIP containing exactly `model.onnx` and
`model.json` through Android's document picker. The ONNX model is static float32
opset 17. The app validates the file SHA-256, taxonomy, tensor names and shapes,
then executes a dry inference before activating it. Weights are generated or imported separately. The optional YOLOP starter is
BDD pretrained and supports only road and lane masks.

## Export after training

```powershell
.\.venv\Scripts\python.exe -m aurora2w.cli export --checkpoint output\idfa\best.pt --output artifacts\aurora-idfa.zip
```

`alignment=none` exports the independently trained visual baseline. `alignment=idfa`
exports the actual roll-conditioned sampling network with its IMU input and
learned weights preserved. Export does not substitute image rectification or
remove the alignment layers. `ExportModel(..., portable_idfa=True)` explicitly
selects the portable sampling implementation on a deep copy.

The same fully convolutional weights can be exported at a smaller fixed input
size for a phone profile, without changing the checkpoint or its trained weights:

```powershell
.\.venv\Scripts\python.exe -m aurora2w.cli export --checkpoint output\idfa\best.pt --image-size 256 448 --output artifacts\aurora-idfa-256x448.zip
```

Both dimensions must be integer multiples of 32 and at least 64. All native,
portable and ONNX parity cases execute at the requested export resolution.
Metadata records `training_image_shape`, `export_image_shape`, `resolution_changed`
and `requires_resolution_reevaluation`. The existing validation metrics remain
explicitly attached to `validation_image_shape`; changing resolution can alter
small-object recall and segmentation accuracy, so those metrics do not transfer
automatically. Measure the exported profile's task accuracy and device speed
before selecting it. Smaller input dimensions alone do not establish real-time
operation on a particular phone.

Synthetic/unknown checkpoints require `--allow-untrained` for integration exports.
This flag does not promote their training status. Android refuses to activate
them for live perception. Newly trained checkpoints carry explicit
`trained_on_real_data` provenance; legacy checkpoints with no such field remain
unknown. A model marked trained has completed a real-data training epoch, not
established safety, sufficient accuracy, or real-time performance.

Checkpoint and bundle metadata declare `supervised_tasks` (road and/or lane) and
`supervised_detection_classes` (the actually supervised detector class names).
Android renders only those segmentation heads and filters detection classes
before top-k selection and NMS. A pothole-only RDD training run therefore enables
pothole boxes while suppressing the untrained road/lane heads and other classes.
The dashboard identifies partial model scope. Unknown or empty supervision
declarations cannot activate live inference even if the checkpoint claims a
real-data training epoch. Declared supervision is an annotation/training scope,
not evidence of sufficient accuracy. Detector live scope requires observed
training positives and exhaustive coverage; positive-only training does not
enable a class for live use under the current conservative rule.

## Tensors

All tensors are float32 with batch size one. Input height H and width W are fixed
by the training configuration and are multiples of 32.

| Name | Shape | Meaning |
| --- | --- | --- |
| `image` | `[1,3,H,W]` | RGB, NCHW, normalized letterboxed pixels |
| `imu` | `[1,2]` | IDFA only: scene roll radians, confidence |
| `road_logits` | `[1,1,H/4,W/4]` | Independent road foreground logits |
| `lane_logits` | `[1,1,H/4,W/4]` | Independent lane-marking logits |
| `class_logits` | `[1,N,9]` | Sigmoid classes, ordered below |
| `boxes_xyxy` | `[1,N,4]` | Continuous input-image pixel edges, may extend outside input |
| `centerness_logits` | `[1,N]` | Sigmoid centerness logits |

`N=(H/8)(W/8)+(H/16)(W/16)+(H/32)(W/32)`. Classes are exactly:
person, rider, bicycle, motorcycle, autorickshaw, car, bus, truck, pothole.
Potholes are boxes. Road/lane segmentation is pixel based; there is no pothole
mask output in this version. An unmarked road need not have a lane-marking mask.

## Pixels and rendering

For a source frame of width iw and height ih, `scale=min(W/iw,H/ih)`.
`nw=max(1,round(iw*scale))`, `nh=max(1,round(ih*scale))`, with round-to-nearest
ties-to-even. `left=(W-nw)//2`, `top=(H-nh)//2`. Fill padding with RGB
`[114,114,114]`. Bilinear half-pixel resize places source centers at
`(destination+.5)*source_size/resized_size-.5`, clamping at the source edges.
Training uses OpenCV `INTER_LINEAR` on uint8; Android float interpolation can
differ slightly because of uint8 rounding and OpenCV's fixed-point coefficients.
That preprocessing difference must be included in device parity measurements.

Normalize `(RGB/255-mean)/std`, with mean `[.485,.456,.406]` and std
`[.229,.224,.225]`. This operates after any camera-frame display rotation; sensor
roll must use the same resulting image axes. The actual geometric scales are
`sx=nw/iw`, `sy=nh/ih`, which may differ slightly after dimension rounding.

For segmentation: bilinearly resize logits to HxW, sigmoid, remove letterbox
padding, and bilinearly resize probabilities to the source frame; threshold at
0.5 for overlays. For boxes use score
`sqrt(sigmoid(class_logits)*sigmoid(centerness_logits))`; threshold 0.3, classwise
NMS IoU 0.5, at most 100 boxes. Undo padding and scale independently in x/y,
clip to source bounds, discard empty boxes. Predictions remain aligned with the
camera frame; IDFA does not globally rotate the image.

## Phone sensor contract

`imu=[camera_scene_roll_rad,confidence]`. Optical axes are x right, y down,
z forward. Positive scene roll is clockwise in the model input image. Confidence
is a validity gate in `[0,1]`, not a calibrated probability of accuracy.
Missing, stale, unreliable, uncalibrated, or incompatible-stabilization samples
must use `[0,0]`. Roll is derived from calibrated device-to-camera axes and
timestamped device-to-world attitude, rather than a raw Android Euler component.

Python quaternions are normalized `[w,x,y,z]`, device to world, world z up.
`R_device_camera` maps optical camera axes into sensor-device axes. Gravity in
camera axes is `(R_world_device @ R_device_camera).T @ [0,0,-1]`; roll is
`atan2(-gravity_x,gravity_y)`. If the optical axis is within the implemented
degenerate horizon threshold (`hypot(gx,gy)<0.1`), confidence is zero.

Use camera exposure midpoint in a verified common monotonic timebase. Android
camera timestamps with an unknown source are insufficient for trusted fusion
until calibrated. Device/camera frame rotation and any image stabilization must
be accounted for. Current IDFA explicitly conditions only local roll; it does
not correct pitch, yaw, translation, rolling shutter, or exposure blur.

Offline synchronization defaults to a maximum bracketing sample gap of 50 ms:

```powershell
.\.venv\Scripts\python.exe -m aurora2w.cli sync --quaternions data\attitude.csv --frames data\frames.csv --calibration data\calibration.json --max-gap-s 0.12 --output data\camera_roll.csv
```

The 0.12 example intentionally permits interpolation of a 10 Hz attitude stream.
It cannot recover high-frequency vibration or make raw gyroscope samples into
calibrated attitude. No extrapolation occurs; larger gaps are rejected. Exact
measured samples remain valid at either edge of a dropped-sample gap.

## Portable IDFA and validation scope

Native training uses torchvision deformable convolution with float32 sampling
inside an AMP-disabled region. Export uses nine bilinear `GridSample` operators,
each followed by a 1x1 convolution with the corresponding original 3x3 kernel
slice. These contributions are summed. Offsets preserve `(dy,dx)` ordering,
zero padding, the rotated prior, confidence gate, and bounded learned residual.
Weights and checkpoint keys are unchanged. This is the same local sampling
operation, not a replacement model.

ONNX GridSample is standardized from opset 16, including half-pixel grids and
zero padding. ONNX Runtime 1.22 lists a float32 CPU kernel for opsets 16–19.
The full Android ORT package is used, with CPU as the compatibility path;
custom reduced builds must retain GridSample. NNAPI/NPU delegation of this
graph is not asserted. Sources: [ONNX GridSample](https://onnx.ai/onnx/operators/onnx__GridSample.html),
[ORT 1.22 CPU kernel table](https://raw.githubusercontent.com/microsoft/onnxruntime/v1.22.0/docs/OperatorKernels.md),
[ORT Android build](https://onnxruntime.ai/docs/build/android.html).

Each export compares native PyTorch, portable PyTorch, and ONNX Runtime CPU on
fixed varied tensors; IDFA also varies roll (including pi), fractional confidence,
and confidence zero. Unit tests additionally exercise nonzero learned refinement,
single-pixel dimensions, out-of-bounds taps and independent ONNX sampling parity.
The destination is published only after comparisons pass. Metadata records the
per-output maximum absolute errors and tolerances, versions and test count.
These are infrastructure checks. Export does not measure held-out accuracy,
camera preprocessing parity, Android latency, battery draw or thermal throttling.

Verification in the isolated D-drive environment used PyTorch 2.8.0+cu128,
torchvision 0.23.0+cu128, ONNX 1.18.0 and ONNX Runtime 1.22.0. The initial
export/sensor suite passed 34 checks; an additional RTX 4050 test passed native
CUDA versus portable CPU parity and verified finite FP16 AMP gradients through
IDFA. These results establish implementation checks, not trained-road accuracy.

For Android instrumentation, regenerate the ignored test-only graph and native
expected tensors before building the test APK:

```powershell
.\.venv\Scripts\python.exe -m aurora2w.android_fixture --ensure
```

The generator source is `aurora2w/android_fixture.py`. It writes only
`android/app/src/androidTest/assets/untrained_idfa_fixture/` (never main assets).
`--ensure` verifies hashes or rebuilds its three generated files. The synthetic
64x96 model uses three roll/confidence cases and full expected output arrays;
synthetic batch-normalization statistics make the roll input's effect observable.
The test constructs an inference engine directly; production model import must
still reject the fixture. Actual emulator/device test results must be reported
separately from desktop ONNX parity and physical-phone performance.

The implementation avoids an explicit nine-feature stack. At P3 with 384x640
input and 64 channels, one float32 sampled feature map is 0.94 MiB. Nine taps
still multiply sampling work and add separate convolution nodes; runtime
scheduling and graph optimization determine actual peak memory and speed.
Confidence zero currently disables offsets numerically but still runs IDFA.

## Metadata

The authoritative field definitions are in `aurora2w/mobile_contract.py`.
Required importer fields include `format_version`, `model_sha256`, `alignment`,
`image_shape`, `input_names`, `input_shapes`, `outputs`, `output_shapes`, `classes`,
`dtype`, `layout`, `color`, `mean`, `std`, `pixel_scale`, `letterbox`, and the
training provenance fields. For live activation all of these must agree:
`training_status=="trained"`, `trained_on_real_data==true`,
`synthetic_smoke==false`, `live_inference_allowed==true`.
At least one explicitly declared supervised segmentation task or detector class
is also required; only that declared scope is rendered.

Metadata hashes verify accidental file mismatch, not authorship. Import only
models produced from the intended project training run. Held-out research
results live in the training/evaluation reports and must accompany any later
comparison with DrivableNets, HybridNets, or other baselines.
