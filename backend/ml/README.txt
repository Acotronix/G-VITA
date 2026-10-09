YOGAMAT ML PRODUCTION PACKAGE
==============================

Pose detector:
    YOLO11m-Pose

Pose classifier:
    Random Forest

Supported poses:
    - Downdog
    - Goddess
    - Plank
    - Tree
    - Warrior2

Keypoints:
    17 COCO/Ultralytics pose keypoints

Features:
    51 normalized landmark features

Current dataset test accuracy:
    96.79%

External image tests completed:
    2 / 2 correctly classified

Important:
    External tests are preliminary and do not constitute
    a statistically sufficient generalization benchmark.

Contents:
    yolo11m-pose.pt
    yogamat_pose_classifier_rf.pkl
    yogamat_model_config.json
    yogamat_production_summary.json
    yogamat_inference.py
    freeze_metadata.json
    requirements.txt
