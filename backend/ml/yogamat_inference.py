
"""
YOGAMAT Production Inference Module
====================================

Generated from the validated YOGAMAT Colab ML pipeline.

Components:
- YOLO11m-Pose
- 17 body keypoints
- 51 normalized features
- Random Forest pose classifier
- Pose-specific form analyzers

Supported poses:
- Downdog
- Goddess
- Plank
- Tree
- Warrior2
"""

import os
import json
import joblib
import numpy as np

from ultralytics import YOLO


# ============================================================
# Paths
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

YOLO_MODEL_PATH = os.path.join(
    BASE_DIR,
    "yolo11m-pose.pt"
)

CLASSIFIER_PATH = os.path.join(
    BASE_DIR,
    "yogamat_pose_classifier_rf.pkl"
)

CONFIG_PATH = os.path.join(
    BASE_DIR,
    "yogamat_model_config.json"
)


# ============================================================
# Configuration
# ============================================================

PROJECT_NAME = "YOGAMAT"

YOLO_MODEL_NAME = "yolo11m-pose.pt"

POSE_CLASSES = [
    "Downdog",
    "Goddess",
    "Plank",
    "Tree",
    "Warrior2"
]

NUM_CLASSES = 5

NUM_KEYPOINTS = 17

FEATURES_PER_KEYPOINT = 3

NUM_FEATURES = 51

KEYPOINT_CONFIDENCE_THRESHOLD = 0.70


# ============================================================
# Load production models
# ============================================================

pose_detector = YOLO(YOLO_MODEL_PATH)

pose_classifier = joblib.load(
    CLASSIFIER_PATH
)


# ============================================================
# Form analyzer registry
# ============================================================

def extract_keypoints(image_path, model):
    """
    Run YOLO11m-Pose on one image.

    Returns:
        keypoints: numpy array with shape (17, 3)
                   [x, y, confidence]

        None if no reliable pose is detected.
    """

    results = model.predict(
        source=image_path,
        verbose=False
    )

    # --------------------------------------------------------
    # Check whether YOLO returned anything
    # --------------------------------------------------------

    if not results:
        return None

    result = results[0]

    if result.keypoints is None:
        return None

    if len(result.keypoints.data) == 0:
        return None

    # --------------------------------------------------------
    # Take the first detected person
    # --------------------------------------------------------

    keypoints = (
        result.keypoints.data[0]
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # Validate shape
    # --------------------------------------------------------

    if keypoints.shape != (17, 3):
        return None

    return keypoints


def normalize_keypoints(keypoints):
    """
    Normalize YOLO keypoints relative to hip center
    and shoulder-to-hip body scale.

    Input:
        (17, 3)

    Output:
        (17, 3)
    """

    points = keypoints[:, :2].copy()
    confidence = keypoints[:, 2].copy()

    # --------------------------------------------------------
    # Hip center
    # --------------------------------------------------------

    left_hip = points[11]
    right_hip = points[12]

    hip_center = (
        left_hip +
        right_hip
    ) / 2.0

    # --------------------------------------------------------
    # Shoulder center
    # --------------------------------------------------------

    left_shoulder = points[5]
    right_shoulder = points[6]

    shoulder_center = (
        left_shoulder +
        right_shoulder
    ) / 2.0

    # --------------------------------------------------------
    # Body scale
    # --------------------------------------------------------

    body_scale = np.linalg.norm(
        shoulder_center -
        hip_center
    )

    if body_scale < 1e-6:
        body_scale = 1.0

    # --------------------------------------------------------
    # Normalize coordinates
    # --------------------------------------------------------

    normalized_points = (
        points - hip_center
    ) / body_scale

    # --------------------------------------------------------
    # Restore confidence
    # --------------------------------------------------------

    normalized = np.column_stack(
        [
            normalized_points,
            confidence
        ]
    )

    return normalized


def flatten_landmarks(normalized_keypoints):
    """
    Convert (17, 3) landmarks into
    a single 51-feature vector.
    """

    return normalized_keypoints.flatten()


def calculate_angle(point_a, point_b, point_c):
    """
    Calculate the angle at point_b formed by:
    point_a -> point_b -> point_c
    """
    ba = point_a - point_b
    bc = point_c - point_b

    cosine_angle = np.dot(ba, bc) / (
        np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8
    )

    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    return np.degrees(np.arccos(cosine_angle))


def analyze_downdog_form(keypoints):

    measurements = calculate_downdog_measurements(keypoints)

    # Elbow scores
    left_elbow_score, left_elbow_feedback = score_elbow_extension(
        measurements["left_elbow"],
        measurements["left_elbow_conf"]
    )

    right_elbow_score, right_elbow_feedback = score_elbow_extension(
        measurements["right_elbow"],
        measurements["right_elbow_conf"]
    )

    elbow_score = (
        left_elbow_score + right_elbow_score
    ) / 2.0

    # Knee extension scores
    left_knee_score, left_knee_feedback = score_knee_extension(
        measurements["left_knee"],
        measurements["left_knee_conf"]
    )

    right_knee_score, right_knee_feedback = score_knee_extension(
        measurements["right_knee"],
        measurements["right_knee_conf"]
    )

    knee_extension_score = (
        left_knee_score + right_knee_score
    ) / 2.0

    # Knee symmetry
    knee_symmetry_score, knee_symmetry_feedback = score_knee_symmetry(
        measurements["left_knee"],
        measurements["right_knee"],
        measurements["left_knee_conf"],
        measurements["right_knee_conf"]
    )

    # Knee quality = 70% extension + 30% symmetry
    knee_score = (
        0.70 * knee_extension_score
        + 0.30 * knee_symmetry_score
    )

    # Overall = 70% elbow + 30% knee
    form_score = (
        0.70 * elbow_score
        + 0.30 * knee_score
    )

    # ------------------------------------------
    # Feedback selection
    # ------------------------------------------

    feedback = "Good DownDog form."

    if elbow_score < 70:
        feedback = "Straighten your elbows more."

    elif knee_extension_score < 70:
        feedback = "Try to straighten your knees more."

    elif knee_symmetry_score < 70:
        feedback = "Try to balance both knees more evenly."

    elif form_score >= 90:
        feedback = "Excellent DownDog form."

    elif form_score >= 75:
        feedback = "Good form. Keep refining your alignment."

    else:
        feedback = "Focus on extending your arms and legs."

    return {
        "form_score": round(form_score, 2),
        "elbow_score": round(elbow_score, 2),
        "knee_score": round(knee_score, 2),
        "knee_extension_score": round(knee_extension_score, 2),
        "knee_symmetry_score": round(knee_symmetry_score, 2),
        "measurements": measurements,
        "feedback": feedback
    }


def analyze_goddess_form(keypoints):

    measurements = calculate_goddess_measurements(keypoints)

    # ------------------------------------------
    # Individual knee scores
    # ------------------------------------------

    left_knee_score, left_feedback = score_goddess_knee_bend(
        measurements["left_knee"],
        measurements["left_knee_conf"]
    )

    right_knee_score, right_feedback = score_goddess_knee_bend(
        measurements["right_knee"],
        measurements["right_knee_conf"]
    )

    knee_extension_score = (
        left_knee_score + right_knee_score
    ) / 2.0

    # ------------------------------------------
    # Knee symmetry
    # ------------------------------------------

    knee_symmetry_score, symmetry_feedback = (
        score_goddess_knee_symmetry(
            measurements["left_knee"],
            measurements["right_knee"],
            measurements["left_knee_conf"],
            measurements["right_knee_conf"]
        )
    )

    # ------------------------------------------
    # Overall knee quality
    # ------------------------------------------

    knee_score = (
        0.70 * knee_extension_score
        + 0.30 * knee_symmetry_score
    )

    # Goddess currently relies primarily
    # on knee geometry.
    form_score = knee_score

    # ------------------------------------------
    # Feedback
    # ------------------------------------------

    if knee_score >= 90:
        feedback = "Excellent Goddess form."

    elif knee_score >= 75:
        feedback = "Good form. Keep refining your alignment."

    elif knee_score >= 60:
        feedback = "Try to deepen your Goddess stance."

    else:
        feedback = "Focus on bending both knees evenly."

    return {
        "form_score": round(form_score, 2),
        "knee_score": round(knee_score, 2),
        "knee_extension_score": round(knee_extension_score, 2),
        "knee_symmetry_score": round(knee_symmetry_score, 2),
        "measurements": measurements,
        "feedback": feedback
    }


def analyze_plank_form(keypoints):

    measurements = calculate_plank_measurements(keypoints)

    # ------------------------------------------
    # Hip alignment
    # ------------------------------------------

    left_hip_score, left_hip_feedback = score_plank_hip_alignment(
        measurements["left_hip"],
        measurements["left_hip_conf"]
    )

    right_hip_score, right_hip_feedback = score_plank_hip_alignment(
        measurements["right_hip"],
        measurements["right_hip_conf"]
    )

    hip_score = (
        left_hip_score + right_hip_score
    ) / 2.0

    # ------------------------------------------
    # Elbow extension
    # ------------------------------------------

    left_elbow_score, left_elbow_feedback = score_plank_elbow_extension(
        measurements["left_elbow"],
        measurements["left_elbow_conf"]
    )

    right_elbow_score, right_elbow_feedback = score_plank_elbow_extension(
        measurements["right_elbow"],
        measurements["right_elbow_conf"]
    )

    elbow_score = (
        left_elbow_score + right_elbow_score
    ) / 2.0

    # ------------------------------------------
    # Overall Plank score
    # ------------------------------------------

    form_score = (
        0.60 * hip_score
        + 0.40 * elbow_score
    )

    # ------------------------------------------
    # Feedback
    # ------------------------------------------

    if hip_score < 60:
        feedback = "Keep your hips aligned with your body."

    elif elbow_score < 60:
        feedback = "Try to straighten your elbows more."

    elif form_score >= 90:
        feedback = "Excellent Plank form."

    elif form_score >= 75:
        feedback = "Good form. Keep refining your alignment."

    else:
        feedback = "Focus on keeping your body straight."

    return {
        "form_score": round(form_score, 2),
        "hip_score": round(hip_score, 2),
        "elbow_score": round(elbow_score, 2),
        "measurements": measurements,
        "feedback": feedback
    }


def analyze_tree_form(keypoints):

    measurements = calculate_tree_measurements(keypoints)

    # ------------------------------------------
    # Determine standing and bent legs
    # ------------------------------------------

    left_knee = measurements["left_knee"]
    right_knee = measurements["right_knee"]

    if left_knee >= right_knee:
        standing_side = "left"
        standing_knee = left_knee
        standing_conf = measurements["left_knee_conf"]

        bent_knee = right_knee
        bent_conf = measurements["right_knee_conf"]

    else:
        standing_side = "right"
        standing_knee = right_knee
        standing_conf = measurements["right_knee_conf"]

        bent_knee = left_knee
        bent_conf = measurements["left_knee_conf"]

    # ------------------------------------------
    # Standing leg
    # ------------------------------------------

    standing_score = score_tree_standing_knee(
        standing_knee,
        standing_conf
    )

    # ------------------------------------------
    # Bent leg
    # ------------------------------------------

    bent_score = score_tree_bent_knee(
        bent_knee,
        bent_conf
    )

    # ------------------------------------------
    # Hip alignment
    # ------------------------------------------

    left_hip_score = score_tree_hip_alignment(
        measurements["left_hip"],
        measurements["left_hip_conf"]
    )

    right_hip_score = score_tree_hip_alignment(
        measurements["right_hip"],
        measurements["right_hip_conf"]
    )

    hip_score = (
        left_hip_score + right_hip_score
    ) / 2.0

    # ------------------------------------------
    # Overall score
    # ------------------------------------------

    knee_score = (
        0.60 * standing_score
        + 0.40 * bent_score
    )

    form_score = (
        0.70 * knee_score
        + 0.30 * hip_score
    )

    # ------------------------------------------
    # Feedback
    # ------------------------------------------

    if standing_score < 60:
        feedback = "Try to straighten your standing leg."

    elif bent_score < 60:
        feedback = "Bend your raised leg more."

    elif hip_score < 60:
        feedback = "Keep your hips more aligned."

    elif form_score >= 90:
        feedback = "Excellent Tree form."

    elif form_score >= 75:
        feedback = "Good form. Keep refining your balance."

    else:
        feedback = "Focus on your standing leg and hip alignment."

    return {
        "form_score": round(form_score, 2),
        "knee_score": round(knee_score, 2),
        "standing_leg_score": round(standing_score, 2),
        "bent_leg_score": round(bent_score, 2),
        "hip_score": round(hip_score, 2),
        "standing_side": standing_side,
        "measurements": measurements,
        "feedback": feedback
    }


def analyze_warrior2_form(keypoints):

    measurements = calculate_warrior2_measurements(keypoints)

    left_knee = measurements["left_knee"]
    right_knee = measurements["right_knee"]

    # ------------------------------------------
    # Determine bent and straight legs
    # ------------------------------------------

    if left_knee < right_knee:
        bent_side = "left"

        bent_knee = left_knee
        bent_conf = measurements["left_knee_conf"]

        straight_knee = right_knee
        straight_conf = measurements["right_knee_conf"]

    else:
        bent_side = "right"

        bent_knee = right_knee
        bent_conf = measurements["right_knee_conf"]

        straight_knee = left_knee
        straight_conf = measurements["left_knee_conf"]

    # ------------------------------------------
    # Knee scores
    # ------------------------------------------

    bent_knee_score = score_warrior2_bent_knee(
        bent_knee,
        bent_conf
    )

    straight_knee_score = score_warrior2_straight_knee(
        straight_knee,
        straight_conf
    )

    # ------------------------------------------
    # Elbow scores
    # ------------------------------------------

    left_elbow_score = score_warrior2_elbow(
        measurements["left_elbow"],
        measurements["left_elbow_conf"]
    )

    right_elbow_score = score_warrior2_elbow(
        measurements["right_elbow"],
        measurements["right_elbow_conf"]
    )

    elbow_score = (
        left_elbow_score + right_elbow_score
    ) / 2.0

    # ------------------------------------------
    # Hip scores
    # ------------------------------------------

    left_hip_score = score_warrior2_hip(
        measurements["left_hip"],
        measurements["left_hip_conf"]
    )

    right_hip_score = score_warrior2_hip(
        measurements["right_hip"],
        measurements["right_hip_conf"]
    )

    hip_score = (
        left_hip_score + right_hip_score
    ) / 2.0

    # ------------------------------------------
    # Overall score
    # ------------------------------------------

    form_score = (
        0.35 * bent_knee_score
        + 0.30 * straight_knee_score
        + 0.25 * elbow_score
        + 0.10 * hip_score
    )

    # ------------------------------------------
    # Feedback
    # ------------------------------------------

    if bent_knee_score < 60:
        feedback = "Bend your front knee more."

    elif straight_knee_score < 60:
        feedback = "Straighten your back leg more."

    elif elbow_score < 60:
        feedback = "Extend your arms more."

    elif hip_score < 60:
        feedback = "Focus on your hip alignment."

    elif form_score >= 90:
        feedback = "Excellent Warrior II form."

    elif form_score >= 75:
        feedback = "Good form. Keep refining your stance."

    else:
        feedback = "Focus on your knee and arm alignment."

    return {
        "form_score": round(form_score, 2),
        "bent_knee_score": round(bent_knee_score, 2),
        "straight_knee_score": round(straight_knee_score, 2),
        "elbow_score": round(elbow_score, 2),
        "hip_score": round(hip_score, 2),
        "bent_side": bent_side,
        "measurements": measurements,
        "feedback": feedback
    }



def analyze_pose_form(pose_name, keypoints):
    """
    Select the correct form analyzer.
    Keep pose classification available if form analysis fails.
    """

    if pose_name not in FORM_ANALYZERS:
        return {
            "success": False,
            "message": f"No form analyzer available for {pose_name}."
        }

    analyzer = FORM_ANALYZERS[pose_name]

    try:
        result = analyzer(keypoints)

        return {
            "success": True,
            "pose": pose_name,
            "result": result
        }

    except Exception as error:
        import logging

        logging.exception(
            "Form analysis failed for pose: %s",
            pose_name
        )

        return {
            "success": False,
            "pose": pose_name,
            "message": (
                "Pose classification may succeed, but form analysis "
                "is temporarily unavailable because a required "
                "analysis function failed."
            ),
            "error": str(error)
        }



def run_yogamat_inference(image_path):

    # ------------------------------------------
    # 1. YOLO11m-Pose detection
    # ------------------------------------------
    keypoints = extract_keypoints(
        image_path,
        pose_detector
    )

    if keypoints is None:
        return {
            "success": False,
            "message": "No person/pose detected."
        }

    # ------------------------------------------
    # 2. Normalize landmarks
    # ------------------------------------------
    normalized = normalize_keypoints(keypoints)

    # ------------------------------------------
    # 3. Convert to 51 features
    # ------------------------------------------
    features = flatten_landmarks(normalized)

    # ------------------------------------------
    # 4. Random Forest pose classification
    # ------------------------------------------
    feature_input = features.reshape(1, -1)

    predicted_pose = pose_classifier.predict(
        feature_input
    )[0]

    pose_probabilities = pose_classifier.predict_proba(
        feature_input
    )[0]

    pose_confidence = float(
        np.max(pose_probabilities) * 100
    )

    # ------------------------------------------
    # 5. Unified form analysis
    # ------------------------------------------
    form_analysis = analyze_pose_form(
        predicted_pose,
        keypoints
    )

    # ------------------------------------------
    # 6. Final result
    # ------------------------------------------
    return {
        "success": True,
        "predicted_pose": predicted_pose,
        "pose_confidence": round(pose_confidence, 2),
        "form_analysis": form_analysis,
        "keypoints": keypoints
    }


# ============================================================
# Form analyzer registry
# ============================================================

FORM_ANALYZERS = {
    "Downdog": analyze_downdog_form,
    "Goddess": analyze_goddess_form,
    "Plank": analyze_plank_form,
    "Tree": analyze_tree_form,
    "Warrior2": analyze_warrior2_form
}


# ============================================================
# Production smoke test
# ============================================================

def get_model_info():

    return {
        "project": PROJECT_NAME,
        "pose_detector": YOLO_MODEL_NAME,
        "pose_classes": POSE_CLASSES,
        "num_classes": NUM_CLASSES,
        "num_keypoints": NUM_KEYPOINTS,
        "num_features": NUM_FEATURES
    }
"""
YOGAMAT Production Inference Module
YOLO11-Pose keypoint extraction + Random Forest pose classification
+ pose-specific form feedback.

Supported classes: Downdog, Goddess, Plank, Tree, Warrior2
COCO keypoint indices:
0 nose, 5/6 shoulders, 7/8 elbows, 9/10 wrists,
11/12 hips, 13/14 knees, 15/16 ankles.
"""

import os
import logging
import json

import joblib
import numpy as np
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_MODEL_NAME = "yolo11m-pose.pt"
YOLO_MODEL_PATH = os.path.join(BASE_DIR, YOLO_MODEL_NAME)
CLASSIFIER_PATH = os.path.join(BASE_DIR, "yogamat_pose_classifier_rf.pkl")
CONFIG_PATH = os.path.join(BASE_DIR, "yogamat_model_config.json")

PROJECT_NAME = "YOGAMAT"
POSE_CLASSES = ["Downdog", "Goddess", "Plank", "Tree", "Warrior2"]
NUM_CLASSES = 5
NUM_KEYPOINTS = 17
FEATURES_PER_KEYPOINT = 3
NUM_FEATURES = 51
KEYPOINT_CONFIDENCE_THRESHOLD = 0.30

# Load models once when the API starts.
pose_detector = YOLO(YOLO_MODEL_PATH)
pose_classifier = joblib.load(CLASSIFIER_PATH)


def extract_keypoints(image_path, model=None):
    """Return first person's COCO keypoints as a (17, 3) array or None."""
    model = model or pose_detector
    results = model.predict(source=image_path, verbose=False)
    if not results:
        return None
    result = results[0]
    if result.keypoints is None or len(result.keypoints.data) == 0:
        return None

    keypoints = result.keypoints.data[0].cpu().numpy()
    if keypoints.shape != (17, 3):
        return None
    if not np.isfinite(keypoints).all():
        return None
    if np.count_nonzero(keypoints[:, 2] >= KEYPOINT_CONFIDENCE_THRESHOLD) < 5:
        return None
    return keypoints.astype(float)


def normalize_keypoints(keypoints):
    """Normalize xy coordinates around the hip center; retain confidence."""
    points = np.asarray(keypoints, dtype=float)[:, :2].copy()
    confidence = np.asarray(keypoints, dtype=float)[:, 2].copy()
    hip_center = (points[11] + points[12]) / 2.0
    shoulder_center = (points[5] + points[6]) / 2.0
    body_scale = float(np.linalg.norm(shoulder_center - hip_center))
    if not np.isfinite(body_scale) or body_scale < 1e-6:
        body_scale = max(float(np.linalg.norm(np.ptp(points, axis=0))), 1.0)
    normalized_points = (points - hip_center) / body_scale
    return np.column_stack((normalized_points, confidence))


def flatten_landmarks(normalized_keypoints):
    """Convert (17, 3) landmarks into the classifier's 51-feature vector."""
    features = np.asarray(normalized_keypoints, dtype=float).reshape(-1)
    if features.size != NUM_FEATURES:
        raise ValueError(f"Expected {NUM_FEATURES} features, got {features.size}.")
    return features


def calculate_angle(point_a, point_b, point_c):
    """Return angle ABC in degrees."""
    ba = np.asarray(point_a, dtype=float) - np.asarray(point_b, dtype=float)
    bc = np.asarray(point_c, dtype=float) - np.asarray(point_b, dtype=float)
    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denominator < 1e-8:
        return 0.0
    cosine = np.clip(np.dot(ba, bc) / denominator, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _point(kp, index):
    return np.asarray(kp[index, :2], dtype=float)


def _conf(kp, index_a, index_b=None, index_c=None):
    indices = [index_a]
    if index_b is not None:
        indices.append(index_b)
    if index_c is not None:
        indices.append(index_c)
    return float(np.mean([kp[i, 2] for i in indices]))


def _angle(kp, a, b, c):
    return calculate_angle(_point(kp, a), _point(kp, b), _point(kp, c))


def _score_range(value, ideal_low, ideal_high, confidence=1.0, softness=35.0):
    """Transparent heuristic score; lower landmark confidence reduces score."""
    value = float(value)
    if ideal_low <= value <= ideal_high:
        geometry_score = 100.0
    elif value < ideal_low:
        geometry_score = max(0.0, 100.0 - (ideal_low - value) * 100.0 / softness)
    else:
        geometry_score = max(0.0, 100.0 - (value - ideal_high) * 100.0 / softness)
    confidence_factor = float(np.clip(confidence, 0.0, 1.0))
    return round(geometry_score * confidence_factor, 2)


def _measurement(kp, a, b, c):
    return {"angle": round(_angle(kp, a, b, c), 2),
            "confidence": round(_conf(kp, a, b, c), 3)}


def calculate_downdog_measurements(kp):
    le = _measurement(kp, 5, 7, 9)
    re = _measurement(kp, 6, 8, 10)
    lk = _measurement(kp, 11, 13, 15)
    rk = _measurement(kp, 12, 14, 16)
    return {
        "left_elbow": le["angle"], "left_elbow_conf": le["confidence"],
        "right_elbow": re["angle"], "right_elbow_conf": re["confidence"],
        "left_knee": lk["angle"], "left_knee_conf": lk["confidence"],
        "right_knee": rk["angle"], "right_knee_conf": rk["confidence"],
    }


def calculate_goddess_measurements(kp):
    lk = _measurement(kp, 11, 13, 15)
    rk = _measurement(kp, 12, 14, 16)
    return {
        "left_knee": lk["angle"], "left_knee_conf": lk["confidence"],
        "right_knee": rk["angle"], "right_knee_conf": rk["confidence"],
    }


def calculate_plank_measurements(kp):
    # Hip alignment is approximated by the shoulder-hip-ankle body-line angle.
    lh = _measurement(kp, 5, 11, 15)
    rh = _measurement(kp, 6, 12, 16)
    le = _measurement(kp, 5, 7, 9)
    re = _measurement(kp, 6, 8, 10)
    return {
        "left_hip": lh["angle"], "left_hip_conf": lh["confidence"],
        "right_hip": rh["angle"], "right_hip_conf": rh["confidence"],
        "left_elbow": le["angle"], "left_elbow_conf": le["confidence"],
        "right_elbow": re["angle"], "right_elbow_conf": re["confidence"],
    }


def calculate_tree_measurements(kp):
    lk = _measurement(kp, 11, 13, 15)
    rk = _measurement(kp, 12, 14, 16)
    # Hip line angle against horizontal is an imperfect 2D proxy for level hips.
    left_hip = _point(kp, 11)
    right_hip = _point(kp, 12)
    hip_tilt = abs(float(np.degrees(np.arctan2(
        right_hip[1] - left_hip[1], right_hip[0] - left_hip[0]
    ))))
    hip_tilt = min(hip_tilt, abs(180.0 - hip_tilt))
    return {
        "left_knee": lk["angle"], "left_knee_conf": lk["confidence"],
        "right_knee": rk["angle"], "right_knee_conf": rk["confidence"],
        "left_hip": hip_tilt, "left_hip_conf": _conf(kp, 11, 12),
        "right_hip": hip_tilt, "right_hip_conf": _conf(kp, 11, 12),
    }


def calculate_warrior2_measurements(kp):
    lk = _measurement(kp, 11, 13, 15)
    rk = _measurement(kp, 12, 14, 16)
    le = _measurement(kp, 5, 7, 9)
    re = _measurement(kp, 6, 8, 10)
    # Hip tilt from horizontal, used as a simple 2D symmetry proxy.
    hip_l, hip_r = _point(kp, 11), _point(kp, 12)
    tilt = abs(float(np.degrees(np.arctan2(
        hip_r[1] - hip_l[1], hip_r[0] - hip_l[0]
    ))))
    tilt = min(tilt, abs(180.0 - tilt))
    return {
        "left_knee": lk["angle"], "left_knee_conf": lk["confidence"],
        "right_knee": rk["angle"], "right_knee_conf": rk["confidence"],
        "left_elbow": le["angle"], "left_elbow_conf": le["confidence"],
        "right_elbow": re["angle"], "right_elbow_conf": re["confidence"],
        "left_hip": tilt, "left_hip_conf": _conf(kp, 11, 12),
        "right_hip": tilt, "right_hip_conf": _conf(kp, 11, 12),
    }


# Scoring helpers used by the pose analyzers.
def score_elbow_extension(angle, confidence):
    return _score_range(angle, 155, 180, confidence), (
        "Straighten your elbows more." if angle < 155 else "Good elbow extension."
    )


def score_knee_extension(angle, confidence):
    return _score_range(angle, 145, 180, confidence), (
        "Try to straighten your knees more." if angle < 145 else "Good knee extension."
    )


def score_knee_symmetry(left, right, left_conf, right_conf):
    score = _score_range(abs(left - right), 0, 20, min(left_conf, right_conf), softness=35)
    return score, ("Try to balance both knees more evenly." if score < 70 else "Good knee symmetry.")


def score_goddess_knee_bend(angle, confidence):
    return _score_range(angle, 75, 125, confidence), (
        "Bend your knees to a comfortable depth." if angle > 125 else "Good knee bend."
    )


def score_goddess_knee_symmetry(left, right, left_conf, right_conf):
    score = _score_range(abs(left - right), 0, 20, min(left_conf, right_conf), softness=35)
    return score, ("Bend both knees more evenly." if score < 70 else "Good knee symmetry.")


def score_plank_hip_alignment(angle, confidence):
    return _score_range(angle, 155, 180, confidence), (
        "Keep your hips aligned with your body." if angle < 155 else "Good body-line alignment."
    )


def score_plank_elbow_extension(angle, confidence):
    return _score_range(angle, 150, 180, confidence), (
        "Try to straighten your elbows more." if angle < 150 else "Good elbow alignment."
    )


def score_tree_standing_knee(angle, confidence):
    return _score_range(angle, 155, 180, confidence), (
        "Try to straighten your standing leg." if angle < 155 else "Good standing-leg alignment."
    )


def score_tree_bent_knee(angle, confidence):
    return _score_range(angle, 35, 125, confidence), (
        "Adjust your raised leg to a comfortable bend." if not 35 <= angle <= 125 else "Good raised-leg bend."
    )


def score_tree_hip_alignment(angle, confidence):
    return _score_range(angle, 0, 12, confidence, softness=30)


def score_warrior2_bent_knee(angle, confidence):
    return _score_range(angle, 70, 125, confidence), (
        "Bend your front knee to a comfortable depth." if angle > 125 else "Good front-knee bend."
    )


def score_warrior2_straight_knee(angle, confidence):
    return _score_range(angle, 150, 180, confidence), (
        "Straighten your back leg more." if angle < 150 else "Good back-leg extension."
    )


def score_warrior2_elbow(angle, confidence):
    return _score_range(angle, 155, 180, confidence), (
        "Extend your arms more." if angle < 155 else "Good arm extension."
    )


def score_warrior2_hip(angle, confidence):
    return _score_range(angle, 0, 18, confidence, softness=35)


def analyze_downdog_form(keypoints):
    m = calculate_downdog_measurements(keypoints)
    le, _ = score_elbow_extension(m["left_elbow"], m["left_elbow_conf"])
    re, _ = score_elbow_extension(m["right_elbow"], m["right_elbow_conf"])
    elbow_score = (le + re) / 2
    lk, _ = score_knee_extension(m["left_knee"], m["left_knee_conf"])
    rk, _ = score_knee_extension(m["right_knee"], m["right_knee_conf"])
    knee_extension = (lk + rk) / 2
    symmetry, _ = score_knee_symmetry(m["left_knee"], m["right_knee"],
                                      m["left_knee_conf"], m["right_knee_conf"])
    knee_score = 0.70 * knee_extension + 0.30 * symmetry
    score = 0.70 * elbow_score + 0.30 * knee_score
    feedback = ("Straighten your elbows more." if elbow_score < 70 else
                "Try to straighten your knees more." if knee_extension < 70 else
                "Try to balance both knees more evenly." if symmetry < 70 else
                "Excellent Downward Dog form." if score >= 90 else
                "Good form. Keep refining your alignment." if score >= 75 else
                "Focus on extending your arms and legs.")
    return {"form_score": round(score, 2), "elbow_score": round(elbow_score, 2),
            "knee_score": round(knee_score, 2), "knee_extension_score": round(knee_extension, 2),
            "knee_symmetry_score": round(symmetry, 2), "measurements": m, "feedback": feedback}


def analyze_goddess_form(keypoints):
    m = calculate_goddess_measurements(keypoints)
    left, _ = score_goddess_knee_bend(m["left_knee"], m["left_knee_conf"])
    right, _ = score_goddess_knee_bend(m["right_knee"], m["right_knee_conf"])
    extension = (left + right) / 2
    symmetry, _ = score_goddess_knee_symmetry(m["left_knee"], m["right_knee"],
                                               m["left_knee_conf"], m["right_knee_conf"])
    score = 0.70 * extension + 0.30 * symmetry
    feedback = ("Excellent Goddess form." if score >= 90 else
                "Good form. Keep refining your alignment." if score >= 75 else
                "Try a comfortable, deeper Goddess stance." if score >= 60 else
                "Focus on bending both knees evenly.")
    return {"form_score": round(score, 2), "knee_score": round(score, 2),
            "knee_extension_score": round(extension, 2), "knee_symmetry_score": round(symmetry, 2),
            "measurements": m, "feedback": feedback}


def analyze_plank_form(keypoints):
    m = calculate_plank_measurements(keypoints)
    lh, _ = score_plank_hip_alignment(m["left_hip"], m["left_hip_conf"])
    rh, _ = score_plank_hip_alignment(m["right_hip"], m["right_hip_conf"])
    hip = (lh + rh) / 2
    le, _ = score_plank_elbow_extension(m["left_elbow"], m["left_elbow_conf"])
    re, _ = score_plank_elbow_extension(m["right_elbow"], m["right_elbow_conf"])
    elbow = (le + re) / 2
    score = 0.60 * hip + 0.40 * elbow
    feedback = ("Keep your hips aligned with your body." if hip < 60 else
                "Try to straighten your elbows more." if elbow < 60 else
                "Excellent Plank form." if score >= 90 else
                "Good form. Keep refining your alignment." if score >= 75 else
                "Focus on keeping your body straight.")
    return {"form_score": round(score, 2), "hip_score": round(hip, 2),
            "elbow_score": round(elbow, 2), "measurements": m, "feedback": feedback}


def analyze_tree_form(keypoints):
    m = calculate_tree_measurements(keypoints)

    if m["left_knee"] >= m["right_knee"]:
        standing_side = "left"
        standing_knee, standing_conf = m["left_knee"], m["left_knee_conf"]
        bent_knee, bent_conf = m["right_knee"], m["right_knee_conf"]
    else:
        standing_side = "right"
        standing_knee, standing_conf = m["right_knee"], m["right_knee_conf"]
        bent_knee, bent_conf = m["left_knee"], m["left_knee_conf"]

    standing, standing_feedback = score_tree_standing_knee(
        standing_knee, standing_conf
    )
    bent, bent_feedback = score_tree_bent_knee(
        bent_knee, bent_conf
    )

    left_hip = score_tree_hip_alignment(
        m["left_hip"], m["left_hip_conf"]
    )
    right_hip = score_tree_hip_alignment(
        m["right_hip"], m["right_hip_conf"]
    )
    hip = (left_hip + right_hip) / 2.0

    knee = 0.60 * standing + 0.40 * bent
    score = 0.70 * knee + 0.30 * hip

    if standing < 60:
        feedback = standing_feedback
    elif bent < 60:
        feedback = bent_feedback
    elif hip < 60:
        feedback = "Keep your hips more aligned."
    elif score >= 90:
        feedback = "Excellent Tree form."
    elif score >= 75:
        feedback = "Good form. Keep refining your balance."
    else:
        feedback = "Focus on your standing leg and hip alignment."

    return {
        "form_score": round(score, 2),
        "knee_score": round(knee, 2),
        "standing_leg_score": round(standing, 2),
        "bent_leg_score": round(bent, 2),
        "hip_score": round(hip, 2),
        "standing_side": standing_side,
        "measurements": m,
        "feedback": feedback
    }


def analyze_warrior2_form(keypoints):
    m = calculate_warrior2_measurements(keypoints)

    if m["left_knee"] < m["right_knee"]:
        bent_side = "left"
        bent, bent_conf = m["left_knee"], m["left_knee_conf"]
        straight, straight_conf = m["right_knee"], m["right_knee_conf"]
    else:
        bent_side = "right"
        bent, bent_conf = m["right_knee"], m["right_knee_conf"]
        straight, straight_conf = m["left_knee"], m["left_knee_conf"]

    bent_score, bent_feedback = score_warrior2_bent_knee(
        bent, bent_conf
    )
    straight_score, straight_feedback = score_warrior2_straight_knee(
        straight, straight_conf
    )

    left_elbow, left_elbow_feedback = score_warrior2_elbow(
        m["left_elbow"], m["left_elbow_conf"]
    )
    right_elbow, right_elbow_feedback = score_warrior2_elbow(
        m["right_elbow"], m["right_elbow_conf"]
    )
    elbow = (left_elbow + right_elbow) / 2.0

    left_hip = score_warrior2_hip(
        m["left_hip"], m["left_hip_conf"]
    )
    right_hip = score_warrior2_hip(
        m["right_hip"], m["right_hip_conf"]
    )
    hip = (left_hip + right_hip) / 2.0

    score = (
        0.35 * bent_score
        + 0.30 * straight_score
        + 0.25 * elbow
        + 0.10 * hip
    )

    if bent_score < 60:
        feedback = bent_feedback
    elif straight_score < 60:
        feedback = straight_feedback
    elif elbow < 60:
        feedback = (
            left_elbow_feedback
            if left_elbow <= right_elbow
            else right_elbow_feedback
        )
    elif hip < 60:
        feedback = "Focus on your hip alignment."
    elif score >= 90:
        feedback = "Excellent Warrior II form."
    elif score >= 75:
        feedback = "Good form. Keep refining your stance."
    else:
        feedback = "Focus on your knee and arm alignment."

    return {
        "form_score": round(score, 2),
        "bent_knee_score": round(bent_score, 2),
        "straight_knee_score": round(straight_score, 2),
        "elbow_score": round(elbow, 2),
        "hip_score": round(hip, 2),
        "bent_side": bent_side,
        "measurements": m,
        "feedback": feedback
    }


FORM_ANALYZERS = {
    "Downdog": analyze_downdog_form,
    "Goddess": analyze_goddess_form,
    "Plank": analyze_plank_form,
    "Tree": analyze_tree_form,
    "Warrior2": analyze_warrior2_form,
}


def analyze_pose_form(pose_name, keypoints):
    """Run the pose-specific form analyzer without hiding its error."""
    analyzer = FORM_ANALYZERS.get(str(pose_name))
    if analyzer is None:
        return {"success": False, "pose": pose_name,
                "message": f"No form analyzer available for {pose_name}."}
    try:
        return {"success": True, "pose": pose_name, "result": analyzer(keypoints)}
    except Exception as error:
        logging.exception("Form analysis failed for pose %s", pose_name)
        return {"success": False, "pose": pose_name,
                "message": "Pose classified, but form analysis failed.",
                "error": str(error)}


def run_yogamat_inference(image_path):
    """Run person pose detection, pose classification, and form analysis."""
    keypoints = extract_keypoints(image_path, pose_detector)
    if keypoints is None:
        return {"success": False, "message": "No reliable person pose detected."}

    normalized = normalize_keypoints(keypoints)
    features = flatten_landmarks(normalized).reshape(1, -1)
    predicted_pose = str(pose_classifier.predict(features)[0])

    if hasattr(pose_classifier, "predict_proba"):
        probabilities = pose_classifier.predict_proba(features)[0]
        confidence = float(np.max(probabilities) * 100.0)
    else:
        confidence = None

    form_analysis = analyze_pose_form(predicted_pose, keypoints)
    return {
        "success": True,
        "predicted_pose": predicted_pose,
        "pose_confidence": round(confidence, 2) if confidence is not None else None,
        "form_analysis": form_analysis,
        "keypoints": keypoints.tolist(),
    }


def get_model_info():
    """Metadata for the API health endpoint."""
    config = {}
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as file:
                config = json.load(file)
        except (OSError, json.JSONDecodeError):
            logging.warning("Could not read model config at %s", CONFIG_PATH)
    return {
        "project": PROJECT_NAME,
        "pose_detector": YOLO_MODEL_NAME,
        "pose_classes": POSE_CLASSES,
        "num_classes": NUM_CLASSES,
        "num_keypoints": NUM_KEYPOINTS,
        "num_features": NUM_FEATURES,
        **({"config": config} if config else {}),
    }
