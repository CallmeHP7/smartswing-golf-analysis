from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np


# Simple type alias: keypoints[name] = (x, y, visibility)
Keypoints = Dict[str, Tuple[float, float, float]]


@dataclass
class PoseSequence:
    """Holds pose keypoints for a whole video."""
    frames: List[Keypoints]
    fps: float
    width: int
    height: int


class PoseEstimator:
    """
    Thin wrapper around MediaPipe Pose to extract 2D keypoints from a video.

    We import heavy libs (cv2, mediapipe) inside methods so that importing
    this module does not fail if they are not installed.
    """

    def __init__(self):
        # Nothing heavy here; actual imports happen in extract_sequence
        pass

    def _landmark_to_dict(
        self,
        landmarks,
        width: int,
        height: int,
    ) -> Keypoints:
        """
        Convert MediaPipe pose landmarks to a dict of named keypoints.
        """
        # MediaPipe index mapping (subset we care about)
        index_to_name = {
            0: "nose",
            11: "left_shoulder",
            12: "right_shoulder",
            13: "left_elbow",
            14: "right_elbow",
            15: "left_wrist",
            16: "right_wrist",
            23: "left_hip",
            24: "right_hip",
            25: "left_knee",
            26: "right_knee",
            27: "left_ankle",
            28: "right_ankle",
        }

        keypoints: Keypoints = {}
        for idx, lm in enumerate(landmarks):
            if idx not in index_to_name:
                continue
            name = index_to_name[idx]
            x_px = lm.x * width
            y_px = lm.y * height
            vis = lm.visibility
            keypoints[name] = (float(x_px), float(y_px), float(vis))
        return keypoints

    def extract_sequence(self, video_path: str) -> Optional[PoseSequence]:
        """
        Run pose estimation on a video and return PoseSequence.

        Returns None if video cannot be opened.
        """
        import cv2
        import mediapipe as mp

        mp_pose = mp.solutions.pose

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {video_path}")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames: List[Keypoints] = []

        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose:

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert BGR (OpenCV) to RGB for MediaPipe
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb)

                if result.pose_landmarks:
                    kp = self._landmark_to_dict(
                        result.pose_landmarks.landmark,
                        width,
                        height,
                    )
                    frames.append(kp)
                else:
                    # If pose not found, store empty dict
                    frames.append({})

        cap.release()
        return PoseSequence(frames=frames, fps=fps, width=width, height=height)
