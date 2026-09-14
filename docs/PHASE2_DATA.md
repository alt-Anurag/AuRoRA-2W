# Phase 2 data preparation

## Selected strategy

IDD provides Indian-road appearance and traffic-object supervision. BDD100K
provides lane/road supervision and optional object pretraining. RDD2022's Indian
subset supplies explicit pothole boxes. These complement
each other; they are not interchangeable annotation sets. Final evaluation needs
jointly labeled, synchronized Indian two-wheeler rides.

No dataset is downloaded automatically. Public paired two-wheeler sources include
I2WDD and MOTOR, but neither has verified complete road/lane/box labels for this
taxonomy. The repository currently contains Phase 1
Cityscapes, 15 downloaded MOTOR front-video/telemetry clip pairs, and one independent
phone ride. None has the complete Phase 2 perception annotations. MOTOR's published
labels target rider behavior, and its stabilized video does not make raw lean a
validated residual image-roll label. Uncalibrated sensor samples use confidence zero.
Record each source URL, release, license, class definitions, and annotation scope.
Follow `PHASE2_TRAINING_DATA.md` for the source-by-source training allocation.

## Canonical JSONL manifest

One line per image; paths are relative to the manifest or absolute. Example:

```json
{"id":"ride01_000120","source":"own_rides","sequence":"ride01","split":"train","image":"images/000120.jpg","road_mask":"road/000120.png","lane_mask":"lane/000120.png","boxes":[{"class":"pothole","xyxy":[100,220,150,245]}],"detection_classes":["pothole"],"imu":{"roll_rad":0.17,"confidence":1.0}}
```

- Masks: single-channel, same dimensions as the original image. Values 0 negative,
  1 positive, 255 unknown. Missing masks mean unknown. **A conventional 0/255
  foreground mask must be explicitly converted**; 255 means ignore here.
- Boxes: continuous original-image pixel edges `[xmin,ymin,xmax,ymax]`, class names
  from the fixed nine-class taxonomy. Empty boxes mean no positives were supplied.
- `detection_classes`: classes exhaustively labeled on this image, including their
  absence. Only these classes receive negative classification supervision.
  Positive boxes can be used without declaring exhaustive coverage. If a class is
  positive-only everywhere, add audited negatives before expecting useful precision.
- `ignore_boxes`: optional XYXY regions excluded from detection supervision/metrics.
- `imu`: optional calibrated camera **scene** roll, radians, clockwise in x-right/
  y-down coordinates. Confidence is `[0,1]`. Missing sensors default to confidence 0.
- `upright_reference: true`: explicit curator assertion for an upright source image;
  permits synthetic roll supervision. Do not enable automatically for all IDD/BDD.
- `split`: train, val, or test. A source sequence must stay in one split. Keep official
  splits; group additional data by ride/location/device before training.

Do not duplicate an image as separate road/detection samples. Join its available
annotations into one record. Identify cross-source copies and near duplicates
before selecting test data. The audit detects identical decoded pixels and exact
sequence leakage, but not every near duplicate or shared location.

## Source converter

Edit a copy of `configs/phase2_sources.example.json`. It is a schema template, not
a claim that those datasets or exact class names are present. Remove unused
sources. Each source uses an explicit JSONL index; example index lines:

```json
{"id":"0001","sequence":"ride_07","split":"train","image":"images/0001.jpg","annotation":"labels/0001.txt"}
{"id":"0002","sequence":"ride_08","split":"val","image":"images/0002.jpg","road_mask":"road/0002.png","lane_mask":"lane/0002.png"}
```

Image, annotation, and mask paths are relative to the source `root`. `root` and
`index` paths are relative to the source spec. Preserve sequence metadata when
creating indexes; do not assign individual neighboring frames randomly to splits.

Supported formats:

| Format | Input and rules |
| --- | --- |
| `idd_label_ids` | Original IDD **id** encoding, not `level3Id`, `csId`, or `csTrainId`. Default road positives 0 and 2; parking 1 ignored. Configure mapping after label review. Other known labels 3–34 are road negatives, void stays unknown. Does not invent lane/pothole labels or extract boxes from semantic masks. |
| `voc` | Pascal VOC XML, explicit source-name -> canonical-name map. Default 1-based inclusive minima converted to zero-based edges; set `voc_one_based: false` only when source documentation requires it. Difficult/unmapped boxes become ignored regions. RDD `D40` maps to pothole. |
| `yolo` | Five columns: class index, normalized center x/y, width/height. Explicit class map; missing label files fail instead of being assumed negative. Segmentation-style polygon text is rejected. |
| `masks` | Prepared binary road/lane masks and optional canonical `boxes`/`ignore_boxes` in the index. Use this for BDD/own annotations after native conversion. |

BDD100K native lane polylines/curves are **not parsed by this converter**. Use the
[official BDD100K mask conversion tools](https://github.com/bdd100k/bdd100k/blob/master/bdd100k/label/to_mask.py)
first, retaining the lane-mask width policy and
image resolution in experiment metadata. Do not replace lanes with road contours.
For a verified binary source whose foreground is 255, set:

```json
"mask_value_map": {"lane": {"0": 0, "255": 1}, "road": {"0": 0, "255": 1}}
```

The IDD detection/Kaggle example leaves `exhaustive_classes` empty deliberately:
verify coverage and set these to the appropriate mapped names before full training.
For road/lane-only samples, leave detection classes empty. An explicit negative
example uses an empty box list and the exhaustively inspected class list.

```powershell
python -m aurora2w.cli prepare --spec configs/my_sources.json --output data/phase2/manifest.jsonl
python -m aurora2w.cli audit --manifest data/phase2/manifest.jsonl --hashes
```

Review the audit's task counts, per-class positives, exhaustive-image counts,
trusted-IMU count, and source balance. A valid manifest alone does not ensure a
balanced or representative training set. Inspect visual labels before training.

## Smartphone synchronization

Quaternion CSV: `timestamp_s,qw,qx,qy,qz`, normalized **device-to-world** wxyz,
world z up. Frame CSV: `frame_index,timestamp_s,exposure_s`; exposure duration is
optional, otherwise alignment uses capture start. Calibration JSON must supply
`R_device_camera` (camera axes into device axes) and `camera_to_imu_offset_s`
such that `t_imu = t_camera + offset`. Supply measured values, not identity defaults.

```powershell
python -m aurora2w.cli sync --quaternions data/raw_imu/attitude.csv --frames data/raw_imu/frames.csv --calibration data/raw_imu/calibration.json --output data/synced/camera_roll.csv
```

The output is `timestamp_s,roll_rad,confidence` in camera time. If video PTS are
relative but capture times absolute, rebase the frame CSV and output roll timestamps
to the same origin. For video inference, `--frame-times` supplies the matching
`frame_index,timestamp_s` list; otherwise decoder PTS are used. No extrapolation
or fabricated sensor values are used when coverage ends. SLERP does not estimate
missing raw gyro bias or remove motion blur.
