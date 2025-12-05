import argparse
import json

from .config import SwingDetectionConfig, ScoringConfig
from .pose_estimation import PoseEstimator
from .swing_detector import detect_main_swing
from .features import extract_features
from .scoring import score_swing, swing_score_to_dict


def main():
    parser = argparse.ArgumentParser(description="Analyze a golf swing video.")
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to swing video (mp4, avi, etc.)",
    )
    args = parser.parse_args()

    print(f"[INFO] Running pose estimation on {args.video} ...")
    estimator = PoseEstimator()
    sequence = estimator.extract_sequence(args.video)
    if sequence is None or not sequence.frames:
        print("[ERROR] Could not extract pose sequence.")
        return

    swing_cfg = SwingDetectionConfig()
    swing_interval = detect_main_swing(sequence, swing_cfg)
    if swing_interval is None:
        print("[ERROR] No valid swing detected.")
        return

    start_idx, end_idx = swing_interval
    print(f"[INFO] Detected swing from frame {start_idx} to {end_idx}.")

    print("[INFO] Extracting features ...")
    features = extract_features(sequence, start_idx, end_idx)

    print("[INFO] Computing score ...")
    scoring_cfg = ScoringConfig()
    score = score_swing(features, scoring_cfg)

    result = swing_score_to_dict(score)
    print("[RESULT]")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
