from dataclasses import dataclass
from typing import List, Tuple
import math
import numpy as np

from .pose_estimation import PoseSequence, Keypoints


@dataclass
class SwingFeatures:
    # Plane-related
    backswing_plane_median_deg: float
    backswing_plane_std_deg: float
    downswing_alignment_mean_abs_deg: float

    # Rotation & stability related
    spine_angle_change_deg: float
    head_movement_norm: float
    hip_forward_shift_norm: float

    # Wrist hinge
    wrist_hinge_deg: float


def _get_point(kp: Keypoints, name: str) -> np.ndarray:
    if name not in kp:
        return np.array([math.nan, math.nan], dtype=float)
    x, y, vis = kp[name]
    return np.array([x, y], dtype=float)


def _angle_deg(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Angle of vector p2 - p1 relative to horizontal axis, in degrees.
    """
    dx, dy = (p2 - p1).tolist()
    return math.degrees(math.atan2(dy, dx))


def _safe_nanmean(values: List[float]) -> float:
    arr = np.array(values, dtype=float)
    return float(np.nanmean(arr)) if not np.all(np.isnan(arr)) else 0.0


def _safe_nanstd(values: List[float]) -> float:
    arr = np.array(values, dtype=float)
    return float(np.nanstd(arr)) if not np.all(np.isnan(arr)) else 0.0


def _body_height(kp: Keypoints) -> float:
    """
    Rough body height estimate: distance from ankles to shoulders.
    """
    hip = _get_point(kp, "left_hip") + _get_point(kp, "right_hip")
    hip /= 2.0
    shoulder = _get_point(kp, "left_shoulder") + _get_point(kp, "right_shoulder")
    shoulder /= 2.0
    ankle = _get_point(kp, "left_ankle") + _get_point(kp, "right_ankle")
    ankle /= 2.0

    return float(np.linalg.norm(shoulder - ankle))


def extract_features(
    sequence: PoseSequence,
    swing_start: int,
    swing_end: int,
) -> SwingFeatures:
    """
    Compute basic biomechanical features for a swing between frame indices.
    For simplicity, we split this interval into:
      - backswing: first 40%
      - downswing: last 40%
    """

    frames = sequence.frames
    n = swing_end - swing_start + 1
    if n <= 0:
        # degenerate swing
        return SwingFeatures(
            backswing_plane_median_deg=0.0,
            backswing_plane_std_deg=0.0,
            downswing_alignment_mean_abs_deg=0.0,
            spine_angle_change_deg=0.0,
            head_movement_norm=0.0,
            hip_forward_shift_norm=0.0,
            wrist_hinge_deg=90.0,
        )

    backswing_end = swing_start + int(0.4 * n)
    downswing_start = swing_start + int(0.6 * n)

    backswing_indices = list(range(swing_start, min(backswing_end, swing_end)))
    downswing_indices = list(range(max(downswing_start, swing_start), swing_end + 1))

    # 1) Swing plane: approximate by vector (shoulder -> wrist)
    back_angles: List[float] = []
    down_angles: List[float] = []

    for i in backswing_indices:
        kp = frames[i]
        shoulder = (_get_point(kp, "left_shoulder") + _get_point(kp, "right_shoulder")) / 2.0
        wrist = (_get_point(kp, "left_wrist") + _get_point(kp, "right_wrist")) / 2.0
        if not math.isnan(shoulder[0]) and not math.isnan(wrist[0]):
            back_angles.append(_angle_deg(shoulder, wrist))

    for i in downswing_indices:
        kp = frames[i]
        shoulder = (_get_point(kp, "left_shoulder") + _get_point(kp, "right_shoulder")) / 2.0
        wrist = (_get_point(kp, "left_wrist") + _get_point(kp, "right_wrist")) / 2.0
        if not math.isnan(shoulder[0]) and not math.isnan(wrist[0]):
            down_angles.append(_angle_deg(shoulder, wrist))

    backswing_median = _safe_nanmean(back_angles)
    backswing_std = _safe_nanstd(back_angles)

    # alignment difference: angle difference between downswing and backswing plane
    diffs: List[float] = []
    for a in down_angles:
        diffs.append(abs(a - backswing_median))
    alignment_mean_abs = _safe_nanmean(diffs)

    # 2) Spine angle change & head movement (address vs impact ~ mid swing)
    address_idx = swing_start
    impact_idx = swing_start + n // 2

    addr_kp = frames[address_idx]
    imp_kp = frames[impact_idx]

    addr_shoulder = (_get_point(addr_kp, "left_shoulder") + _get_point(addr_kp, "right_shoulder")) / 2.0
    addr_hip = (_get_point(addr_kp, "left_hip") + _get_point(addr_kp, "right_hip")) / 2.0

    imp_shoulder = (_get_point(imp_kp, "left_shoulder") + _get_point(imp_kp, "right_shoulder")) / 2.0
    imp_hip = (_get_point(imp_kp, "left_hip") + _get_point(imp_kp, "right_hip")) / 2.0

    spine_addr_angle = _angle_deg(addr_hip, addr_shoulder)
    spine_imp_angle = _angle_deg(imp_hip, imp_shoulder)
    spine_change = abs(spine_imp_angle - spine_addr_angle)

    # head movement (in pixels, normalized by body height)
    head_addr = _get_point(addr_kp, "nose")
    head_imp = _get_point(imp_kp, "nose")
    body_h = _body_height(addr_kp)
    if body_h <= 0:
        head_movement_norm = 0.0
        hip_shift_norm = 0.0
    else:
        head_movement_norm = float(np.linalg.norm(head_imp - head_addr) / body_h)

        # hip forward shift: horizontal movement (x-axis) of hips
        hip_shift = abs(imp_hip[0] - addr_hip[0])
        hip_shift_norm = float(hip_shift / body_h)

    # 3) Wrist hinge: angle between forearm and club proxy at top (mid-backswing)
    if backswing_indices:
        top_idx = backswing_indices[-1]
    else:
        top_idx = swing_start
    top_kp = frames[top_idx]

    # Forearm vector: elbow -> wrist (average of both sides)
    left_elbow = _get_point(top_kp, "left_elbow")
    left_wrist = _get_point(top_kp, "left_wrist")
    right_elbow = _get_point(top_kp, "right_elbow")
    right_wrist = _get_point(top_kp, "right_wrist")

    valid_forearms = []
    if not math.isnan(left_elbow[0]) and not math.isnan(left_wrist[0]):
        valid_forearms.append(_angle_deg(left_elbow, left_wrist))
    if not math.isnan(right_elbow[0]) and not math.isnan(right_wrist[0]):
        valid_forearms.append(_angle_deg(right_elbow, right_wrist))

    forearm_angle = _safe_nanmean(valid_forearms) if valid_forearms else 0.0

    # Club proxy: shoulder -> wrist
    club_angles = []
    if not math.isnan(left_shoulder[0]) and not math.isnan(left_wrist[0]):
        club_angles.append(_angle_deg(left_shoulder, left_wrist))
    if not math.isnan(right_shoulder[0]) and not math.isnan(right_wrist[0]):
        club_angles.append(_angle_deg(right_shoulder, right_wrist))

    club_angle = _safe_nanmean(club_angles) if club_angles else 0.0

    wrist_hinge = abs(club_angle - forearm_angle)

    return SwingFeatures(
        backswing_plane_median_deg=backswing_median,
        backswing_plane_std_deg=backswing_std,
        downswing_alignment_mean_abs_deg=alignment_mean_abs,
        spine_angle_change_deg=spine_change,
        head_movement_norm=head_movement_norm,
        hip_forward_shift_norm=hip_shift_norm,
        wrist_hinge_deg=wrist_hinge,
    )
