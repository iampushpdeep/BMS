#!/usr/bin/env python3
"""Embed enriched posts (text + media, jointly) with Qwen3-VL-Embedding-2B.

These embeddings are the shared representation behind two downstream uses:
  - matching BMS-labeled posts to their nearest unlabeled neighbor in a large
    firehose sample (see build_faiss_index.py / search_neighbors.py).
  - per-label UMAP + HDBSCAN clustering (see cluster_umap_hdbscan.py).

Each post is encoded as a single chat-style multimodal input -- text plus whatever
images it has plus a handful of sampled video frames -- via the model's own
`sentence-transformers` wrapper (which handles the image/video processor config
Qwen3-VL-Embedding-2B expects; loading it through raw `transformers.AutoModel` skips
that setup). Two kinds of rows are dropped before embedding: negated/retracted label
events (`neg: true` -- the label no longer applies) and pure quote-posts
(`app.bsky.embed.record`, i.e. a repost-with-no-media-of-its-own) which don't carry
any content of their own to embed.

Output is written incrementally as .npy shards alongside a parquet sidecar mapping
shard row -> (uri, cid, label), so a run can be resumed or parallelized across shards
without re-embedding everything from scratch.

Usage:
    python3 generate_embeddings.py --input "out/batch_*.jsonl" --output-dir embeddings/
    python3 generate_embeddings.py --input "out/batch_*.jsonl" --output-dir embeddings/ \
        --batch-size 64 --device cuda --max-video-frames 4
"""
import argparse
import glob
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset

MODEL_NAME = "Qwen/Qwen3-VL-Embedding-2B"
EMBED_DIM = 2048


def extract_video_frames(video_path: str, max_frames: int, fps: float, max_side: int, min_side: int = 32) -> list[Image.Image]:
    """Uniformly samples frames at ~fps, resizing to fit max_side while dropping any
    frame too small to be meaningful (min_side) -- a full video is far more than a
    semantic embedding needs. opencv is only imported here, not at module level, so
    processing a dataset with no video posts never requires installing it."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    frame_step = max(int(video_fps / fps), 1)

    frames, count, target_size = [], 0, None
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_step == 0:
            h, w, _ = frame.shape
            if h >= min_side and w >= min_side:
                if target_size is None:
                    scale = max_side / max(h, w) if max(h, w) > max_side else 1.0
                    target_size = (int(w * scale), int(h * scale))
                frame = cv2.resize(frame, target_size)
                rh, rw, _ = frame.shape
                if rh >= min_side and rw >= min_side:
                    frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        count += 1
    cap.release()
    return frames


class PostDataset(Dataset):
    def __init__(self, rows: list[dict], max_side: int, video_max_frames: int, video_fps: float):
        self.rows = rows
        self.max_side = max_side
        self.video_max_frames = video_max_frames
        self.video_fps = video_fps

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        images = []
        for img in row.get("images") or []:
            path = img.get("file")
            if not path or not Path(path).exists():
                continue
            try:
                with Image.open(path) as im:
                    im = im.convert("RGB")
                    w, h = im.size
                    if max(w, h) > self.max_side:
                        scale = self.max_side / max(w, h)
                        im = im.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
                    images.append(im.copy())
            except Exception:
                continue

        video_frames = []
        video_path = row.get("video")
        if video_path and Path(video_path).exists():
            try:
                video_frames = extract_video_frames(
                    video_path, max_frames=self.video_max_frames, fps=self.video_fps, max_side=self.max_side,
                )
            except Exception:
                pass

        return {
            "text": (row.get("text") or "").strip(),
            "images": images or None,
            "video_frames": video_frames or None,
            "uri": row.get("uri"), "cid": row.get("cid"), "label": row.get("label"),
        }


def to_chat_input(item: dict) -> dict:
    content = []
    if item["text"]:
        content.append({"type": "text", "text": item["text"]})
    for img in item["images"] or []:
        content.append({"type": "image", "image": img})
    if item["video_frames"]:
        content.append({"type": "video", "video": item["video_frames"]})
    if not content:
        content.append({"type": "text", "text": ""})
    return {"role": "user", "content": content}


def iter_usable_rows(paths: list[str]):
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("neg"):
                    continue  # a retracted label -- nothing to embed under it anymore
                if row.get("etype") == "app.bsky.embed.record":
                    continue  # pure quote-post, no media/text of its own
                if row.get("skip") and not (row.get("text") or "").strip():
                    continue  # no usable media and no text
                yield row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="glob pattern for enriched batch_*.jsonl files")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers for image/video decode")
    parser.add_argument("--max-side", type=int, default=512, help="resize images/frames so the longer side is at most this many px")
    parser.add_argument("--max-video-frames", type=int, default=4)
    parser.add_argument("--video-fps", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--shard-size", type=int, default=200_000, help="rows per .npy shard")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(glob.glob(args.input))
    if not paths:
        raise SystemExit(f"no files matched: {args.input}")

    print(f"loading {MODEL_NAME} on {args.device}...")
    model = SentenceTransformer(MODEL_NAME, device=args.device, model_kwargs={"torch_dtype": torch.bfloat16})
    model.processor.image_processor.size = {"shortest_edge": 384, "longest_edge": args.max_side}
    model.processor.video_processor.size = {"shortest_edge": 384, "longest_edge": args.max_side}
    model.processor.video_processor.max_frames = args.max_video_frames
    model.processor.video_processor.fps = args.video_fps
    model.eval()

    rows = list(iter_usable_rows(paths))
    print(f"{len(rows):,} posts to embed (after dropping negations / quote-posts / empty rows)")
    dataset = PostDataset(rows, args.max_side, args.max_video_frames, args.video_fps)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        collate_fn=lambda x: x, pin_memory=True, persistent_workers=False,
    )

    shard_vecs, shard_meta, shard_idx, n_total = [], [], 0, 0

    def flush_shard():
        nonlocal shard_vecs, shard_meta, shard_idx
        if not shard_vecs:
            return
        vecs = np.concatenate(shard_vecs, axis=0)
        np.save(args.output_dir / f"embeddings_{shard_idx:05d}.npy", vecs)
        pd.DataFrame(shard_meta).to_parquet(args.output_dir / f"meta_{shard_idx:05d}.parquet", index=False)
        print(f"  shard {shard_idx}: {len(vecs)} vectors")
        shard_idx += 1
        shard_vecs, shard_meta = [], []

    with torch.inference_mode():
        for batch in loader:
            chat_inputs = [to_chat_input(item) for item in batch]
            embeddings = model.encode(chat_inputs, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
            vecs = embeddings.float().cpu().numpy()
            assert vecs.shape[1] == EMBED_DIM, f"unexpected embedding dim {vecs.shape[1]}"

            shard_vecs.append(vecs)
            shard_meta.extend({"uri": it["uri"], "cid": it["cid"], "label": it["label"]} for it in batch)
            n_total += len(batch)
            if sum(len(v) for v in shard_vecs) >= args.shard_size:
                flush_shard()
            if n_total % (args.batch_size * 50) == 0:
                print(f"embedded {n_total:,} / {len(rows):,} posts...")

    flush_shard()
    print(f"done: {n_total:,} posts embedded across {shard_idx} shard(s) in {args.output_dir}")


if __name__ == "__main__":
    main()
