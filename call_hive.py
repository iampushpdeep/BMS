#!/usr/bin/env python3
"""Call Hive AI's visual moderation API for a set of post images and save the raw
per-class scores.

This only calls the API and records what it returns -- one row per image, with a flat
{class_name: score} dict across all of Hive's heads. It makes no judgment about what
those scores mean for any particular label; that mapping is not part of this release.

Usage:
    export HIVE_API_KEY=...
    python3 call_hive.py --input "out/batch_*.jsonl" --output hive_scores.parquet
"""
import argparse
import glob
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests

HIVE_ENDPOINT = "https://api.thehive.ai/api/v2/task/sync"


def call_hive(image_url: str, api_key: str, retries: int = 3) -> dict[str, float]:
    """Returns a flat {class_name: score} dict flattened across all of Hive's heads."""
    for attempt in range(retries):
        resp = requests.post(
            HIVE_ENDPOINT,
            headers={"Authorization": f"Token {api_key}"},
            data={"input": json.dumps({"url": image_url})},
            timeout=30,
        )
        if resp.status_code == 429:
            time.sleep(2 ** attempt)
            continue
        resp.raise_for_status()
        payload = resp.json()
        scores = {}
        for output in payload.get("status", [{}])[0].get("response", {}).get("output", [{}]):
            for cls in output.get("classes", []):
                scores[cls["class"]] = cls["score"]
        return scores
    raise RuntimeError(f"Hive API failed after {retries} retries for {image_url}")


def iter_image_rows(paths: list[str]):
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                images = row.get("images") or []
                if len(images) == 1 and not row.get("video"):
                    yield row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="glob pattern for enriched batch_*.jsonl files")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--api-key", default=os.environ.get("HIVE_API_KEY"))
    parser.add_argument("--image-base-url", default="",
                         help="prefix to turn a local image path into a URL Hive can fetch (e.g. your own CDN/bucket)")
    args = parser.parse_args()
    if not args.api_key:
        raise SystemExit("Hive API key required: --api-key or $HIVE_API_KEY")

    rows = []
    for i, row in enumerate(iter_image_rows(sorted(glob.glob(args.input)))):
        image_path = row["images"][0]["file"]
        image_url = args.image_base_url + image_path
        try:
            scores = call_hive(image_url, args.api_key)
        except Exception as e:
            print(f"  [skip] {row['uri']}: {e}")
            continue

        rows.append({
            "uri": row["uri"], "cid": row["cid"], "bms_label": row.get("label"),
            "hive_scores": json.dumps(scores),
        })
        if (i + 1) % 50 == 0:
            print(f"scored {i + 1} images...")

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"wrote {len(df):,} rows to {args.output}")


if __name__ == "__main__":
    main()
