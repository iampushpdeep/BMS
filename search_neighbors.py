#!/usr/bin/env python3
"""For every BMS-labeled post, find its nearest neighbor in the unlabeled firehose
index -- the "most similar post BMS never flagged."

For each labeled post's embedding, this queries the faiss index (build_faiss_index.py)
for its top-k nearest firehose candidates, then walks that ranked list and keeps the
highest-ranked candidate that (a) carries no BMS label and (b) is a single-image post
(restricting to single-image posts on both sides keeps a Hive AI call further
downstream -- see call_hive.py -- well-defined). k=1000 candidates are pulled per
query specifically so there is still a same-modality, unlabeled candidate left after
that filtering, even for labels where many of the nearest points are themselves
labeled.

Usage:
    python3 search_neighbors.py --index firehose.index --labeled-embeddings labeled_embeddings/ \
        --labeled-posts "out/batch_*.jsonl" --output nearest_neighbors.parquet
"""
import argparse
import glob
import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

TOP_K = 1000


def load_labeled_embeddings(embeddings_dir: Path) -> tuple[np.ndarray, pd.DataFrame]:
    vecs = [np.load(p) for p in sorted(glob.glob(str(embeddings_dir / "embeddings_*.npy")))]
    meta = [pd.read_parquet(p) for p in sorted(glob.glob(str(embeddings_dir / "meta_*.parquet")))]
    return np.concatenate(vecs, axis=0).astype(np.float32), pd.concat(meta, ignore_index=True)


def load_single_image_flags(labeled_posts_glob: str) -> dict[str, bool]:
    """Reads the enriched records to know which labeled URIs are single-image posts."""
    flags = {}
    for path in glob.glob(labeled_posts_glob):
        with open(path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                flags[row["uri"]] = len(row.get("images") or []) == 1 and not row.get("video")
    return flags


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", required=True, type=Path, help="faiss index from build_faiss_index.py")
    parser.add_argument("--labeled-embeddings", required=True, type=Path,
                         help="embeddings_*.npy/meta_*.parquet shards for the labeled posts (generate_embeddings.py run against them)")
    parser.add_argument("--labeled-posts", required=True, help="glob for the same posts' enriched batch_*.jsonl (for the single-image filter)")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()

    print(f"loading index from {args.index}...")
    index = faiss.read_index(str(args.index))
    firehose_meta = pd.read_parquet(args.index.with_suffix(".meta.parquet"))
    firehose_labeled = firehose_meta["label"].notna().to_numpy()

    query_vecs, query_meta = load_labeled_embeddings(args.labeled_embeddings)
    faiss.normalize_L2(query_vecs)
    single_image = load_single_image_flags(args.labeled_posts)
    query_mask = query_meta["uri"].map(single_image).fillna(False).to_numpy()
    query_vecs, query_meta = query_vecs[query_mask], query_meta[query_mask].reset_index(drop=True)
    print(f"{len(query_meta):,} single-image labeled posts to match")

    print(f"searching top-{args.top_k} candidates per query...")
    scores, indices = index.search(query_vecs, args.top_k)

    results = []
    for i, uri in enumerate(query_meta["uri"]):
        match_row, match_score = None, None
        for rank, cand_idx in enumerate(indices[i]):
            if cand_idx < 0:
                break
            if firehose_labeled[cand_idx]:
                continue  # candidate has itself been labeled by BMS -- not a miss
            match_row, match_score = cand_idx, scores[i][rank]
            break
        if match_row is None:
            continue
        results.append({
            "query_uri": uri,
            "query_label": query_meta.loc[i, "label"],
            "match_uri": firehose_meta.loc[match_row, "uri"],
            "match_cid": firehose_meta.loc[match_row, "cid"],
            "cosine_similarity": float(match_score),
        })

    out_df = pd.DataFrame(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.output, index=False)
    print(f"matched {len(out_df):,} / {len(query_meta):,} labeled posts to an unlabeled neighbor")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
