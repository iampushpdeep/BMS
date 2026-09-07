#!/usr/bin/env python3
"""Per-label UMAP + HDBSCAN clustering, with hyperparameters tuned by Optuna against a
GPU DBCV score -- for surfacing the granular themes within a moderation label.

For each label independently:
  1. High-dimensional post embeddings (generate_embeddings.py) are projected to 5D
     with UMAP (`init="random"`, `min_dist=0.0`, cosine metric). 5 components (rather
     than the usual 2 used just for plotting) gave HDBSCAN a noticeably
     better-separated space to cluster in -- the first two of those five components
     are what any 2D scatterplot of the results is actually showing. UMAP projections
     are cached per `n_neighbors` value (an LRU cache) since many Optuna trials reuse
     the same one, only varying HDBSCAN's parameters.
  2. HDBSCAN clusters that 5D projection. Points it can't confidently assign to any
     dense region come back labeled -1 ("noise") and are excluded from downstream
     interpretation, not forced into the nearest cluster.
  3. UMAP's n_neighbors and HDBSCAN's min_cluster_size / min_samples are jointly
     tuned per label with Optuna's Tree-structured Parzen Estimator sampler (run
     multivariate, so it can learn correlations between the three), scored by the
     Density-Based Clustering Validation (DBCV) index -- an intrinsic measure (no
     ground truth needed) that rewards internally dense, well-separated clusters.
     Search ranges scale with corpus size (a few hundred posts and a few million posts
     don't want the same min_cluster_size range), and min_samples is sampled as a
     *fraction* of min_cluster_size rather than independently, which keeps Optuna from
     wasting trials on nonsensical (min_samples > min_cluster_size) combinations.

Both UMAP and HDBSCAN run on GPU via RAPIDS cuML, and DBCV is computed on GPU too --
`dbcv_gpu()` below re-derives it directly from HDBSCAN's minimum spanning tree rather
than calling out to the CPU `hdbscan` package's reference implementation, since that's
the bottleneck that made a multi-hundred-trial-per-label search tractable at this
dataset's scale.

Labels above --downsample-above posts are uniformly downsampled before tuning and
clustering, purely so a trial finishes in reasonable time -- porn and sexual are the
only labels this actually applies to at this dataset's scale. A tiny amount of
Gaussian noise is added to embeddings before projecting, purely to break exact-duplicate
points that would otherwise create degenerate distances.

The final HDBSCAN fit (at the best found hyperparameters) also records each point's
soft cluster-membership probability (0 for noise points), which downstream cluster
interpretation samples posts by, weighted toward each cluster's most representative
members rather than uniformly.

Requires a CUDA GPU + RAPIDS cuML; see requirements.txt for how to install it.

Usage:
    python3 cluster_umap_hdbscan.py --embeddings-dir embeddings/intolerant/ \
        --label intolerant --output-dir clusters/intolerant/
    python3 cluster_umap_hdbscan.py --embeddings-dir embeddings/porn/ \
        --label porn --output-dir clusters/porn/ --downsample-above 300000
"""
from __future__ import annotations  # defer type-hint evaluation: importing this module
                                     # (e.g. just to reuse load_embeddings elsewhere)
                                     # shouldn't require cuml/cupy to already be installed

import argparse
import glob
import json
from collections import OrderedDict
from pathlib import Path

import cudf
import cupy as cp
import numpy as np
import optuna
import pandas as pd
import torch
from cuml.cluster import HDBSCAN
from cuml.manifold import UMAP

N_COMPONENTS = 5
MIN_DIST = 0.0
METRIC = "cosine"
MAX_UMAP_CACHE = 5  # how many distinct n_neighbors projections to keep resident at once


def setup_rmm(initial: str = "512MiB", maximum: str = "38GiB") -> None:
    """A pooled GPU memory allocator -- without this, the alloc/free churn across
    hundreds of Optuna trials fragments GPU memory badly enough to eventually OOM on
    the larger labels."""
    import rmm
    pool = rmm.mr.PoolMemoryResource(
        rmm.mr.ManagedMemoryResource(), initial_pool_size=initial, maximum_pool_size=maximum,
    )
    rmm.mr.set_current_device_resource(pool)


def load_embeddings(embeddings_dir: Path) -> tuple[np.ndarray, pd.DataFrame]:
    vecs = [np.load(p) for p in sorted(glob.glob(str(embeddings_dir / "embeddings_*.npy")))]
    meta = [pd.read_parquet(p) for p in sorted(glob.glob(str(embeddings_dir / "meta_*.parquet")))]
    return np.concatenate(vecs, axis=0).astype(np.float32), pd.concat(meta, ignore_index=True)


def get_param_ranges(n: int) -> dict:
    """Search ranges scale with corpus size: a few hundred posts and a few million
    posts shouldn't be tuned over the same min_cluster_size range."""
    if n < 20_000:
        return dict(n_neighbors_lo=10, n_neighbors_hi=30, min_cs_lo=50, min_cs_hi=500)
    elif n < 500_000:
        return dict(n_neighbors_lo=15, n_neighbors_hi=45, min_cs_lo=500, min_cs_hi=5000)
    else:
        return dict(n_neighbors_lo=20, n_neighbors_hi=60, min_cs_lo=2000, min_cs_hi=20000)


def dbcv_gpu(minimum_spanning_tree, cluster_labels: np.ndarray) -> float:
    """Density-Based Clustering Validation (Moulavi et al. 2014), computed directly
    from HDBSCAN's minimum spanning tree on GPU. Equivalent in spirit to the `hdbscan`
    package's CPU `validity.validity_index`, just fast enough to call once per trial."""
    device = "cuda"
    labels = torch.tensor(cluster_labels, dtype=torch.long, device=device)

    mst_df = minimum_spanning_tree.to_pandas()
    from_nodes = torch.tensor(mst_df["from"].values, dtype=torch.long, device=device)
    to_nodes = torch.tensor(mst_df["to"].values, dtype=torch.long, device=device)
    distances = torch.tensor(mst_df["distance"].values, dtype=torch.float32, device=device)

    sizes = torch.bincount(labels + 1, minlength=int(torch.max(labels)) + 2)
    cluster_size = sizes[1:].cpu().numpy()
    total = int(sizes.sum().item())
    num_clusters = len(cluster_size)
    if num_clusters < 1:
        return -1.0

    dsc = torch.zeros(num_clusters, dtype=torch.float32, device=device)
    dspc_wrt = torch.full((num_clusters,), float("inf"), dtype=torch.float32, device=device)
    max_distance = torch.max(distances).item()
    min_outlier_sep = float("inf")

    label1, label2 = labels[from_nodes], labels[to_nodes]
    one_noise = (label1 == -1) | (label2 == -1)
    same_cluster = (label1 == label2) & (label1 != -1)
    diff_cluster = (label1 != label2) & (label1 != -1) & (label2 != -1)

    if torch.any(one_noise):
        min_outlier_sep = torch.min(distances[one_noise]).item()
    if torch.any(same_cluster):
        sc_labels, sc_dists = label1[same_cluster], distances[same_cluster]
        for i in range(num_clusters):
            mask = sc_labels == i
            if torch.any(mask):
                dsc[i] = torch.max(sc_dists[mask])
    if torch.any(diff_cluster):
        l1d, l2d, dcd = label1[diff_cluster], label2[diff_cluster], distances[diff_cluster]
        for i in range(num_clusters):
            for mask in (l1d == i, l2d == i):
                if torch.any(mask):
                    dspc_wrt[i] = torch.min(torch.tensor([dspc_wrt[i], torch.min(dcd[mask])]))

    if min_outlier_sep == float("inf"):
        min_outlier_sep = max_distance
    dsc, dspc_wrt = dsc.cpu().numpy(), dspc_wrt.cpu().numpy()
    correction = 2 * (max_distance if num_clusters > 1 else min_outlier_sep)
    dspc_wrt[np.isinf(dspc_wrt)] = correction

    v_index = [(dspc_wrt[i] - dsc[i]) / max(dspc_wrt[i], dsc[i]) for i in range(num_clusters)]
    return float(np.sum([(cluster_size[i] * v_index[i]) / total for i in range(num_clusters)]))


def make_umap_cache(X_gpu: cp.ndarray, seed: int):
    cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def get(n_neighbors: int) -> np.ndarray:
        if n_neighbors in cache:
            cache.move_to_end(n_neighbors)
            return cache[n_neighbors]
        if len(cache) >= MAX_UMAP_CACHE:
            cache.popitem(last=False)
            cp.get_default_memory_pool().free_all_blocks()
        projected = UMAP(
            n_neighbors=n_neighbors, n_components=N_COMPONENTS, min_dist=MIN_DIST,
            metric=METRIC, random_state=seed, init="random",
        ).fit_transform(X_gpu).get()
        cache[n_neighbors] = projected
        return projected

    return get


def fit_hdbscan(projected: np.ndarray, min_cluster_size: int, min_samples: int):
    umap_df = cudf.DataFrame(projected, columns=[f"umap_{i}" for i in range(projected.shape[1])])
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=min_samples, metric="euclidean",
        prediction_data=True, gen_min_span_tree=True,
    )
    labels = clusterer.fit_predict(umap_df).to_numpy()
    return clusterer, labels


def tune(X_gpu: cp.ndarray, n: int, n_trials: int, seed: int) -> dict:
    ranges = get_param_ranges(n)
    get_umap = make_umap_cache(X_gpu, seed)
    seen: dict[tuple, float] = {}

    def objective(trial: optuna.Trial) -> float:
        n_neighbors = trial.suggest_int("n_neighbors", ranges["n_neighbors_lo"], ranges["n_neighbors_hi"])
        min_cluster_size = trial.suggest_int("min_cluster_size", ranges["min_cs_lo"], ranges["min_cs_hi"], log=True)
        min_samples_frac = trial.suggest_float("min_samples_frac", 0.001, 0.20)
        min_samples = max(1, int(min_samples_frac * min_cluster_size))

        key = (n_neighbors, min_cluster_size, min_samples)
        if key in seen:
            return seen[key]

        projected = get_umap(n_neighbors)
        clusterer, labels = fit_hdbscan(projected, min_cluster_size, min_samples)
        score = -1.0 if len(set(labels) - {-1}) < 1 else dbcv_gpu(clusterer.minimum_spanning_tree_, labels)
        seen[key] = score
        del clusterer
        cp.get_default_memory_pool().free_all_blocks()
        return score

    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True, n_startup_trials=20)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params | {"min_samples": max(1, int(study.best_params["min_samples_frac"] * study.best_params["min_cluster_size"])), "dbcv": study.best_value}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embeddings-dir", required=True, type=Path,
                         help="embeddings_*.npy/meta_*.parquet shards for this label, from generate_embeddings.py")
    parser.add_argument("--label", required=True, help="BMS label these embeddings belong to")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--n-trials", type=int, default=None,
                         help="Optuna trials to run (default: adaptive, ~40*log10(n) clamped to [60, 300])")
    parser.add_argument("--downsample-above", type=int, default=300_000,
                         help="uniformly downsample to this many posts if the label has more (only porn/sexual hit this at full scale)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_rmm()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading embeddings for '{args.label}'...")
    vecs, meta = load_embeddings(args.embeddings_dir)
    n_total = len(vecs)
    print(f"{n_total:,} posts")

    rng = np.random.default_rng(args.seed)
    if n_total > args.downsample_above:
        idx = np.sort(rng.choice(n_total, size=args.downsample_above, replace=False))
        vecs, meta = vecs[idx], meta.iloc[idx].reset_index(drop=True)
        print(f"downsampled to {len(vecs):,} posts")

    vecs = vecs + 1e-4 * rng.standard_normal(vecs.shape).astype(np.float32)  # break exact-duplicate degeneracies
    X_gpu = cp.asarray(vecs)

    n_trials = args.n_trials or max(60, min(300, int(40 * np.log10(n_total))))
    print(f"tuning over {n_trials} trials...")
    best = tune(X_gpu, len(vecs), n_trials, args.seed)
    print(f"best params: {best}")

    print("re-fitting UMAP + HDBSCAN with the best params (with membership probabilities)...")
    get_umap = make_umap_cache(X_gpu, args.seed)
    projected = get_umap(best["n_neighbors"])
    clusterer, labels = fit_hdbscan(projected, best["min_cluster_size"], best["min_samples"])
    probs = clusterer.probabilities_.to_numpy()
    probs = np.where(labels == -1, 0.0, probs)  # noise points get probability 0, not cuML's raw value

    n_clusters = len(set(labels) - {-1})
    noise_frac = float((labels == -1).mean())
    print(f"{n_clusters} clusters, {noise_frac:.1%} noise")

    np.save(args.output_dir / "umap_5d.npy", projected.astype(np.float32))
    out_meta = meta.copy()
    out_meta["cluster"] = labels
    out_meta["cluster_probability"] = probs
    out_meta.to_parquet(args.output_dir / "cluster_assignments.parquet", index=False)
    with open(args.output_dir / "best_params.json", "w") as f:
        json.dump({
            "label": args.label, "n_posts": len(vecs), "n_clusters": n_clusters,
            "noise_frac": noise_frac, **best,
        }, f, indent=2)

    print(f"wrote 5D projection, cluster assignments + probabilities, and best params to {args.output_dir}")
    print("(cluster naming/summarization -- sampling representative posts per cluster and "
          "prompting a vision-language model for a short label -- is a separate step, "
          "not included in this release.)")


if __name__ == "__main__":
    main()
