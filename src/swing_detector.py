from typing import Tuple, Optional, List
import numpy as np

from .config import SwingDetectionConfig
from .pose_estimation import PoseSequence, Keypoints


def _get_wrist_position(kp: Keypoints) -> Optional[np.ndarray]:
    """
    Return average of both wrists if available, else one wrist, else None.
    """
    candidates = []
    for name in ["left_wrist", "right_wrist"]:
        if name in kp:
            x, y, vis = kp[name]
            if vis > 0.3:
                candidates.append((x, y))
    if not candidates:
        return None
    arr = np.array(candidates, dtype=float)
    return arr.mean(axis=0)


def compute_wrist_velocity(sequence: PoseSequence) -> List[float]:
    """
    Compute per-frame wrist speed (pixels per frame).
    """
    positions: List[Optional[np.ndarray]] = []
    for kp in sequence.frames:
        positions.append(_get_wrist_position(kp))

    velocities: List[float] = [0.0] * len(positions)
    prev_pos = None

    for i, pos in enumerate(positions):
        if pos is None or prev_pos is None:
            velocities[i] = 0.0
        else:
            velocities[i] = float(np.linalg.norm(pos - prev_pos))
        prev_pos = pos if pos is not None else prev_pos

    return velocities


def detect_main_swing(
    sequence: PoseSequence,
    config: SwingDetectionConfig,
) -> Optional[Tuple[int, int]]:
    """
    Detect the main swing interval [start_idx, end_idx] using wrist velocity.

    Simple heuristic:
    - find segment where velocity is above start_threshold
    - stop when it falls below end_threshold for a while
    - choose the longest such segment as main swing
    """
    v = compute_wrist_velocity(sequence)
    v = np.array(v, dtype=float)

    start_thresh = config.start_velocity_threshold
    end_thresh = config.end_velocity_threshold

    candidate_segments: List[Tuple[int, int]] = []
    in_swing = False
    start_idx = 0

    for i, val in enumerate(v):
        if not in_swing:
            if val > start_thresh:
                in_swing = True
                start_idx = i
        else:
            # already in swing
            if val < end_thresh:
                end_idx = i
                if end_idx - start_idx >= config.min_swing_frames:
                    candidate_segments.append((start_idx, end_idx))
                in_swing = False

    if in_swing:
        # If swing continued till end
        end_idx = len(v) - 1
        if end_idx - start_idx >= config.min_swing_frames:
            candidate_segments.append((start_idx, end_idx))

    if not candidate_segments:
        return None

    # choose longest segment
    best = max(candidate_segments, key=lambda seg: seg[1] - seg[0])
    return best
