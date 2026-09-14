# Development

## Environment
Use Python 3.12 and compatible PyTorch/torchvision. The lock file records the Windows CUDA environment; other platforms may need dependency adjustments.
~~~powershell
python -m venv .venv
. .\scripts\env.ps1
python -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu128
python -m aurora2w.cli doctor --device cuda
~~~
Caches and temporary output stay inside the checkout. Toolchains and datasets are not bundled.

## Android
Install JDK 17, Gradle 8.11.1 and SDK platform 35. Set JAVA_HOME, ANDROID_HOME and PATH. Android Studio can also open the android directory. The app targets Android 8 and later; physical verification currently covers the S21 FE on Android 16.

~~~powershell
.\scripts\build-android.ps1
.\scripts\build-android.ps1 -Install
~~~

No model weights are committed. The YOLOP preparation entry point uses the private IDD validation manifest at data/phase2/prepared/research_baseline_v2/manifest.jsonl to check conversion. After preparing that data:
~~~powershell
.\scripts\prepare-yolop.ps1 -Size 320 -PrepareAssets
.\scripts\build-android.ps1
~~~
This fetches pinned upstream source, exports road/lane outputs and creates private test assets without training. Without model assets, a camera and motion build remains possible; compatible model ZIPs can be imported through the app.

## Checks
~~~powershell
python -B -m pytest -p no:cacheprovider tests -q --basetemp=output/tests_local
.\scripts\build-android.ps1
# After YOLOP assets are prepared, with one authorized device:
.\scripts\test-android.ps1
~~~
Some Python tests use synthetic optimizer/export checks. Android instrumentation generates an isolated IDFA fixture and uses prepared YOLOP references. Neither is trained Aurora accuracy.

## Research training
The first full run still awaits the researcher's approval. After data review and approval:
~~~powershell
.\scripts\train.ps1 -Manifest data/phase2/prepared/research_baseline_v2/manifest.jsonl -Model baseline -Run output/research_baseline_v2
python -m aurora2w.cli export --checkpoint output/research_baseline_v2/best.pt --output artifacts/aurora-baseline.zip
~~~
IDFA continuation needs verified roll supervision and a matched visual control. Resume is explicit. See [model metadata](MODEL_CONTRACT.md); scores from one resolution do not establish another export's accuracy.
