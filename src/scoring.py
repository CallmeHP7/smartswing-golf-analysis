from dataclasses import dataclass
from typing import Dict, Any, List
import math

from .config import ScoringConfig
from .features import SwingFeatures


@dataclass
class SwingScore:
    total_score: float
    plane_consistency_score: float
    alignment_score: float
    rotation_stability_score: float
    error_flags: List[str]


def _exp_normalize(value: float, ref: float) -> float:
    """
    Smaller value is better (like std deviation, angle error).
    We map 0 -> 1, ref -> ~0.5, large -> closer to 0.
    """
    if ref <= 0:
        return 1.0
    x = value / ref
    return math.exp(-math.log(2.0) * x)  # exp(-ln2 * x)


def score_swing(features: SwingFeatures, config: ScoringConfig) -> SwingScore:
    # Component scores in [0,1]
    plane_score = _exp_normalize(features.backswing_plane_std_deg,
                                 config.plane_std_ref_deg)

    alignment_score = _exp_normalize(features.downswing_alignment_mean_abs_deg,
                                     config.alignment_ref_deg)

    # For rotation, we blend spine angle change + head movement + hip shift
    spine_part = _exp_normalize(features.spine_angle_change_deg,
                                config.spine_change_ref_deg)
    head_part = _exp_normalize(features.head_movement_norm,
                               config.head_move_ref)
    hip_part = _exp_normalize(features.hip_forward_shift_norm,
                              config.hip_shift_ref)

    rotation_score = (spine_part + head_part + hip_part) / 3.0

    # Weighted sum -> [0,1]
    total_norm = (
        config.w_plane_consistency * plane_score
        + config.w_alignment * alignment_score
        + config.w_rotation_stability * rotation_score
    )
    total_score = 100.0 * total_norm

    # Error flags
    errors: List[str] = []

    if features.downswing_alignment_mean_abs_deg > config.ott_alignment_thresh_deg:
        errors.append("over_the_top_downswing")

    if (features.spine_angle_change_deg > config.early_ext_spine_thresh_deg
            and features.hip_forward_shift_norm > config.early_ext_hip_thresh):
        errors.append("early_extension")

    if (features.wrist_hinge_deg < config.wrist_hinge_min_deg
            or features.wrist_hinge_deg > config.wrist_hinge_max_deg):
        errors.append("improper_wrist_hinge")

    return SwingScore(
        total_score=float(total_score),
        plane_consistency_score=float(plane_score * 100.0),
        alignment_score=float(alignment_score * 100.0),
        rotation_stability_score=float(rotation_score * 100.0),
        error_flags=errors,
    )


def swing_score_to_dict(score: SwingScore) -> Dict[str, Any]:
    return {
        "total_score": round(score.total_score, 1),
        "plane_consistency_score": round(score.plane_consistency_score, 1),
        "alignment_score": round(score.alignment_score, 1),
        "rotation_stability_score": round(score.rotation_stability_score, 1),
        "error_flags": score.error_flags,
    }
