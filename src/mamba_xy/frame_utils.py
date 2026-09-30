"""
Reconstructs the exact cropped RGB frame that was fed to DINOv2 for a
given video_id + frame_idx, by replaying the same deterministic sampling
used at extraction time (see scripts/03_extract_dinov2_embeddings_xy.py).
Verified against results/dinov2_embeddings_xy/config.json's actual
recorded values: num_frames=32, skip_seconds=2.0,
crop_box=(1049,337,1831,915). z reuses x/y's exact same embeddings/frames
(see scripts/03_extract_dinov2_embeddings_z.py), so this one function
covers both targets -- no separate logic needed.

Returns raw uint8 RGB images (not DINOv2-normalized) -- for display only.
"""

import cv2
import numpy as np

NUM_FRAMES = 32
SKIP_SECONDS = 2.0
CROP_BOX = (1049, 337, 1831, 915)


def _crop(frame, crop_box):
    if crop_box is None:
        return frame
    x1, y1, x2, y2 = crop_box
    h, w = frame.shape[:2]
    x1 = max(0, min(int(x1), w))
    x2 = max(0, min(int(x2), w))
    y1 = max(0, min(int(y1), h))
    y2 = max(0, min(int(y2), h))
    return frame[y1:y2, x1:x2]


def get_frame_indices(video_path: str, num_frames: int = NUM_FRAMES, skip_seconds: float = SKIP_SECONDS) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    skip_frames = int(fps * skip_seconds) if fps > 0 else 0
    skip_frames = min(skip_frames, max(total_frames - 1, 0))
    return np.linspace(skip_frames, total_frames - 1, num_frames).astype(int)


def get_cropped_frame(video_path: str, frame_idx: int, crop_box=CROP_BOX,
                       num_frames: int = NUM_FRAMES, skip_seconds: float = SKIP_SECONDS) -> np.ndarray:
    """Returns the cropped RGB frame (uint8, HxWx3) at position frame_idx (0..num_frames-1)."""
    frame_indices = get_frame_indices(video_path, num_frames, skip_seconds)
    target = int(frame_indices[frame_idx])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    success, frame = cap.read()
    cap.release()
    if not success:
        raise RuntimeError(f"Could not read frame {target} from {video_path}")

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = _crop(frame, crop_box)
    return frame
