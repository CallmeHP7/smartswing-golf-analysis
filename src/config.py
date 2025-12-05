from dataclasses import dataclass

@dataclass
class SwingDetectionConfig:
    # Velocity thresholds for detecting start/end of swing
    start_velocity_threshold: float = 5.0   # pixels per frame (tunable)
    end_velocity_threshold: float = 2.0     # pixels per frame (tunable)
    min_swing_frames: int = 10             # minimum frames to consider as a swing


@dataclass
class ScoringConfig:
    # Weights for component scores
    w_plane_consistency: float = 0.4
    w_alignment: float = 0.35
    w_rotation_stability: float = 0.25

    # Normalization constants (rough, assignment-level)
    plane_std_ref_deg: float = 10.0       # std dev of plane angle for a "good" swing
    alignment_ref_deg: float = 10.0       # mean abs difference for a "good" swing
    spine_change_ref_deg: float = 10.0    # spine angle change for a "good" swing
    head_move_ref: float = 0.05          # head movement as fraction of body height
    hip_shift_ref: float = 0.05          # hip shift as fraction of body height

    # Error thresholds (heuristic)
    ott_alignment_thresh_deg: float = 15.0
    early_ext_spine_thresh_deg: float = 15.0
    early_ext_hip_thresh: float = 0.08
    wrist_hinge_min_deg: float = 60.0
    wrist_hinge_max_deg: float = 120.0
