#!/usr/bin/env python3
"""Build a faiss IndexFlatIP over embedded firehose posts.

This is the "haystack" side of the automated-pipeline blindspot analysis: to probe
what BMS's automated pipeline misses, we need -- for every BMS-labeled post -- its
nearest semantic neighbor among posts BMS never flagged. IndexFlatIP does exact
(brute-force) inner-product search: on L2-normalized vectors this is exact cosine
similarity, with no recall trade-off, at the cost of a linear scan per query -- the
right choice at the firehose sample's scale (tens of millions, not billions, of
vectors).

Reads the embedding shards produced by generate_embeddings.py (run against a firehose
sample rather than the label stream) and builds one index covering all of them.

Usage:
    python3 build_faiss_index.py --embeddings-dir firehose_embeddings/ --output firehose.index
"""
import argparse
import glob
from pathlib import Path

import faiss
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embeddings-dir", required=True, type=Path,
                         help="directory of embeddings_*.npy / meta_*.parquet shards from generate_embeddings.py")
    parser.add_argument("--output", required=True, type=Path, help="output index path")
    args = parser.parse_args()

    shard_paths = sorted(glob.glob(str(args.embeddings_dir / "embeddings_*.npy")))
    if not shard_paths:
        raise SystemExit(f"no embeddings_*.npy shards found in {args.embeddings_dir}")

    dim = np.load(shard_paths[0], mmap_mode="r").shape[1]
    print(f"embedding dim: {dim}, {len(shard_paths)} shard(s)")

    index = faiss.IndexFlatIP(dim)

    all_meta = []
    n_total = 0
    for shard_path in shard_paths:
        vecs = np.load(shard_path).astype(np.float32)
        faiss.normalize_L2(vecs)  # inner product on normalized vectors == cosine similarity
        index.add(vecs)

        meta_path = Path(shard_path.replace("embeddings_", "meta_")).with_suffix(".parquet")
        all_meta.append(pd.read_parquet(meta_path))
        n_total += len(vecs)
        print(f"  added {shard_path}: {len(vecs):,} vectors (total {n_total:,})")

    meta = pd.concat(all_meta, ignore_index=True)
    assert len(meta) == index.ntotal, "metadata row count doesn't match index size -- shards out of sync"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(args.output))
    meta_out = args.output.with_suffix(".meta.parquet")
    meta.to_parquet(meta_out, index=False)

    print(f"wrote index ({index.ntotal:,} vectors) to {args.output}")
    print(f"wrote row -> (uri, cid, label) metadata to {meta_out}")
    print("note: faiss index row i corresponds to meta.parquet row i, in insertion order")


if __name__ == "__main__":
    main()
