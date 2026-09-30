# scripts/03_extract_dinov2_embeddings_master.py
#
# Extracts each (video_id, variant) pair EXACTLY ONCE into a master cache,
# instead of once per fold -- avoids re-encoding the same video multiple
# times across the 5-fold AND leave-one-batch-out split schemes (both need
# the same underlying per-video embeddings, just grouped into different
# train/val sets). At 128 frames/video this redundancy would otherwise be
# expensive to repeat.
#
# 106 videos x {clean, mirror} = 212 encodings total, 128 frames each.
# Every video needs mirror too, since every video is a TRAIN video in most
# folds of both split schemes.
#
# Same crop box, same 2-second skip, same uniform-linspace sampling as the
# original xy_early_prediction extraction -- only NUM_FRAMES changed.
#
# Writes: results/dinov2_master_cache/{video_id_safe}__{variant}.pt
#   each file: {"embedding": [128, D] tensor, "video_id": str, "variant": str}

import os
import sys
import json

import cv2
import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

from mamba_xy.core import set_seed

CROP_BOX = (1049, 337, 1831, 915)
NUM_FRAMES = 128
IMAGE_SIZE = 224
SKIP_SECONDS = 2.0
SEED = 42
CACHE_DIR = os.path.join(ROOT, "results", "dinov2_master_cache")


def _crop(frame, crop_box):
    x1, y1, x2, y2 = crop_box
    h, w = frame.shape[:2]
    x1, x2 = max(0, min(int(x1), w)), max(0, min(int(x2), w))
    y1, y2 = max(0, min(int(y1), h)), max(0, min(int(y2), h))
    return frame[y1:y2, x1:x2]


def sample_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    skip = min(int(fps * SKIP_SECONDS) if fps > 0 else 0, max(total - 1, 0))
    idxs = np.linspace(skip, total - 1, NUM_FRAMES).astype(int)

    raw_frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        raw_frames.append(_crop(frame, CROP_BOX))
    cap.release()

    if not raw_frames:
        raise RuntimeError(f"No readable frames: {video_path}")
    while len(raw_frames) < NUM_FRAMES:
        raw_frames.append(raw_frames[-1])
    return raw_frames[:NUM_FRAMES]


def preprocess(raw_frames, mirror):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    out = []
    for frame in raw_frames:
        f = cv2.flip(frame, 1) if mirror else frame
        f = cv2.resize(f, (IMAGE_SIZE, IMAGE_SIZE))
        f = f.astype(np.float32) / 255.0
        f = (f - mean) / std
        out.append(np.transpose(f, (2, 0, 1)))
    return torch.tensor(np.stack(out, axis=0), dtype=torch.float32)


@torch.no_grad()
def encode(backbone, frames, device, chunk_size=4):
    outputs = []
    for start in range(0, frames.shape[0], chunk_size):
        chunk = frames[start:start + chunk_size].to(device)
        outputs.append(backbone(chunk).cpu())
    return torch.cat(outputs, dim=0)


def safe_name(video_id: str) -> str:
    return video_id.replace("/", "_").replace("\\", "_").replace(" ", "_")


def main():
    set_seed(SEED)
    os.makedirs(CACHE_DIR, exist_ok=True)

    xy = pd.read_csv(os.path.join(ROOT, "data", "processed", "video_xy_dataset.csv"))
    z_ids = set(pd.read_csv(os.path.join(ROOT, "data", "processed", "video_z_dataset.csv"))["video_id"])
    videos = xy[xy["video_id"].isin(z_ids)][["video_id", "video_path"]].drop_duplicates("video_id")
    print(f"{len(videos)} videos (106-video set), {NUM_FRAMES} frames each, clean+mirror")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    n_total = len(videos) * 2
    n_done = 0
    for _, row in videos.iterrows():
        video_id, video_path = row["video_id"], os.path.join(ROOT, row["video_path"])
        raw_frames = sample_frames(video_path)
        for variant, mirror in (("clean", False), ("mirror", True)):
            out_path = os.path.join(CACHE_DIR, f"{safe_name(video_id)}__{variant}.pt")
            n_done += 1
            if os.path.exists(out_path):
                print(f"[{n_done}/{n_total}] {video_id} ({variant}) -- already cached, skipping")
                continue
            frames = preprocess(raw_frames, mirror)
            emb = encode(backbone, frames, device)
            torch.save({"embedding": emb, "video_id": video_id, "variant": variant}, out_path)
            print(f"[{n_done}/{n_total}] {video_id} ({variant}) -- done")

    with open(os.path.join(CACHE_DIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"num_frames": NUM_FRAMES, "image_size": IMAGE_SIZE, "skip_seconds": SKIP_SECONDS,
                    "crop_box": list(CROP_BOX), "dinov2_model": "dinov2_vits14", "n_videos": len(videos)}, f, indent=2)
    print(f"\nDone. Master cache: {CACHE_DIR}")


if __name__ == "__main__":
    main()
