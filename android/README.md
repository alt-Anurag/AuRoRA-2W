# Android app

[Build and test](../docs/DEVELOPMENT.md) · [Model contract](../docs/MODEL_CONTRACT.md)

Kotlin, CameraX and ONNX Runtime provide portrait/landscape layouts, independent live preview, matched frame inspection, road/lane overlays, torch control, motion graphs and relative phone attitude animation.

YOLOP assets are generated separately. A clean build without assets provides camera and motion controls and accepts a compatible ZIP. The pretrained starter supports road and lane masks only. Sensors do not condition its neural network. The S21 FE has physical runtime checks; other phones require their own measurements.
