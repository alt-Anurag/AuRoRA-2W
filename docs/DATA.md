# Data and attribution

Download from publishers under their terms. Credentials, archives and raw images do not belong in Git.

- [IDD](https://idd.insaan.iiit.ac.in/): Indian road masks and traffic. Registration and research licence acceptance required.
- [BDD100K](https://bdd-data.berkeley.edu/): road and lane labels. The prepared experiment uses the pinned legacy 2018 release; current same-name archives may differ.
- [RDD2022](https://github.com/sekilab/RoadDamageDetector): India D40 pothole boxes.
- [YOLOP](https://github.com/hustvl/YOLOP): the adapter expects commit 8d8f68df318c71f01d6f813c024df646c7d1978f and embeds its MIT notice in exported metadata.

The README road comparison is an IDD diagnostic figure, credited to G. Varma and colleagues, WACV 2019. App screenshots show emulator output. Graphs show saved project measurements. The DrivableNets reference manuscript/video and dataset archives are not redistributed here.

## Preparation
Use the CLI and maintained preparation tools with a private local data directory:
~~~powershell
python -m aurora2w.cli index --help
python -m aurora2w.cli prepare --help
python -m scripts.prepare_idd_masks --help
python -m scripts.prepare_idd_subset --help
python -m scripts.prepare_idd_detection_subset --help
python -m scripts.prepare_bdd_subset --help
python -m scripts.prepare_rdd_holdout --help
python -m scripts.merge_prepared --help
~~~

Follow [the annotation contract](PHASE2_DATA.md). Missing labels must not become negative supervision. The saved partition has 20,854 training and 3,784 validation records; the private manifest is not bundled. Recreating it needs the corresponding releases, source selection and duplicate review. The RDD image holdout does not prove independent rides.

I2WDD and MOTOR can support later motion studies only after their timing, camera geometry, stabilization and annotation coverage are audited. No new licence is assigned to third party material.
