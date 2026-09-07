# BMS characterization pipeline

Collects and characterizes moderation activity from Bluesky's official first-party
**Bluesky Moderation Service (BMS)** — the labeler at `wss://mod.bsky.app/xrpc` that
publishes label events (e.g. `porn`, `rude`, `spam`, `threat`) for posts and accounts
in real time. The pipeline covers three things on top of that data: how quickly labels
get applied (an automation-vs-human-review signal), where the automated pipeline
misses content it should have caught, and what a label actually contains once you
cluster the posts under it semantically.

Each script is an independent stage that reads the previous one's output from disk
rather than being wired together in memory, so any of them can be re-run,
parallelized, or swapped out on its own.

```
collect_labels.py          enrich_posts.py         compute_timeliness.py
┌───────────────────┐  raw ┌──────────────────┐ posts ┌─────────────────────┐
│ subscribe to BMS's │label │ resolve each post │+media │ label delay vs.     │
│ label firehose      │────▶│ download media    │──────▶│ post createdAt      │
└───────────────────┘ event └──────────────────┘        └─────────────────────┘

generate_embeddings.py     build_faiss_index.py    search_neighbors.py     call_hive.py
┌───────────────────┐  ┌────────────────────┐   ┌───────────────────┐   ┌──────────────┐
│ Qwen3-VL-Embedding  │─▶│ faiss.IndexFlatIP   │──▶│ nearest unlabeled │──▶│ Hive AI raw  │
│ -2B, text + media   │  │ over firehose sample│  │ neighbor per post │  │ class scores │
└───────────────────┘  └────────────────────┘   └───────────────────┘   └──────────────┘

cluster_umap_hdbscan.py
┌──────────────────────────────────┐
│ per-label UMAP(5D) + HDBSCAN,     │
│ Optuna-tuned against DBCV         │
└──────────────────────────────────┘
```

`collect_firehose.py` is an alternative, unlabeled counterpart to the first stage: a
general/background sample of Bluesky activity (via `com.atproto.sync.subscribeRepos`)
to compare labeled posts against, used as the firehose sample that
`build_faiss_index.py` / `search_neighbors.py` search over.

## `collect_labels.py`

Connects to the BMS label firehose and writes every label event to disk, verbatim, for
as long as it runs. This is meant to run unattended and indefinitely: any error just
triggers a reconnect with exponential backoff, and the last-seen sequence number is
saved to a cursor file so a restart resumes instead of re-fetching or skipping events.

```bash
python3 collect_labels.py --output labels.jsonl
```

Each output line is one raw label event as defined by the AT Protocol label lexicon
(`com.atproto.label.defs#label`):

| field | meaning |
|---|---|
| `uri` | the subject being labeled — a post's `at://` URI, or a bare DID for an account-level label |
| `cid` | the specific content version being labeled (present for posts) |
| `val` | the label value itself, e.g. `"porn"`, `"rude"` |
| `neg` | `true` if this event *retracts* a previously applied label |
| `cts` | when the label was created (ISO 8601, UTC) |
| `src` | DID of the labeler that issued it (constant: BMS's own DID) |
| `exp` | expiry timestamp, if the label is time-limited (usually `null`) |
| `ver` | label protocol version (usually `null`) |
| `seq` | the firehose sequence number this event arrived on (our resume cursor) |

## `collect_firehose.py` (alternative, random posts source)

Subscribes to the main repo firehose (unfiltered by any labeler) to build a general
background sample of Bluesky posts — the "firehose" that
`build_faiss_index.py` later uses as the haystack to search for near-misses in.

```bash
python3 collect_firehose.py --output posts.jsonl
```

## `enrich_posts.py`

Reads the raw events from `collect_labels.py` and, for every post-level event,
resolves the author's DID to their PDS, fetches the post's full record, and downloads
any attached media. Account-level label events (a bare DID, no post to resolve) are
dropped at this stage — there is nothing to enrich.

```bash
python3 enrich_posts.py --input labels.jsonl --output-dir out/

# only keep specific labels, and use more concurrent downloads
python3 enrich_posts.py --input labels.jsonl --output-dir out/ \
    --labels porn,sexual,nudity --concurrency 50

# process several collected files at once
python3 enrich_posts.py --input "labels_*.jsonl" --output-dir out/
```

Output goes to `out/batch_00000.jsonl`, `out/batch_00001.jsonl`, … (one row per post,
`--batch-size` rows per file), with media saved into sibling `out/images/` and
`out/videos/` directories. This row shape is what every downstream script consumes:

| field | meaning |
|---|---|
| `cid` | the post's content-version id |
| `uri` | the post's `at://` URI |
| `label` | the label value applied to this post (renamed from the raw event's `val`) |
| `neg` | whether this event retracts a previously applied label |
| `cts` | when the **label** was created — not the post's own timestamp |
| `record` | the post's full raw record, as returned by its PDS — includes `record["createdAt"]` (the post's own creation time), `record["text"]`, `record["embed"]`, `record["reply"]`, language tags, etc. |
| `etype` | the post's embed `$type` (`""` if it has no embed) |
| `text` | denormalized copy of `record["text"]`, for convenience |
| `images` | `[{"file": <local path>, "alt": <alt text>}, ...]` — every downloaded image, including link-preview thumbnails (`alt: "external_thumb"`) |
| `video` | local path to the downloaded video, or `null` |
| `skip` | `true` if the post has no usable media (text-only, or every download attempt failed) |
| `failures` | human-readable strings describing anything that went wrong downloading this post's media (empty if all went cleanly) |

**Embed shapes handled:** `app.bsky.embed.images`, `app.bsky.embed.video`,
`app.bsky.embed.external` (its optional link-preview thumbnail), and
`app.bsky.embed.recordWithMedia` (a quote-post that attaches its own image/video/link
card — unwrapped so its media is classified the same way a plain post's would be).
Newer or uncommon embed shapes (e.g. the multi-image `app.bsky.embed.gallery`) are left
unhandled by design and simply come through as `skip: true`.

**Resumable:** progress (which input file and byte offset) is saved to
`out/enrich_cursor.json` after every batch. Re-running the same command picks up where
it left off. Already-downloaded media files are detected and never re-fetched.

## `compute_timeliness.py`

Computes labeling delay (`label cts − post createdAt`) per post and uses it as an
empirical proxy for automation vs. human oversight: a label applied within seconds, at
scale, is likely automated; one taking hours or days indicates a human review step.
Filters out backdated posts and negative delays (both signs of an untrustworthy
`createdAt`, not evidence about the labeler) before computing anything.

```bash
python3 compute_timeliness.py --input "out/batch_*.jsonl" --output timeliness.parquet \
    --percentiles-out delay_percentiles.csv
```

## `generate_embeddings.py`

Embeds each post's text + media jointly with `Qwen3-VL-Embedding-2B`. This is the
shared representation behind both the automated-pipeline blindspot search and the
per-label clustering below — run it once against your labeled posts and once against a
firehose sample.

```bash
python3 generate_embeddings.py --input "out/batch_*.jsonl" --output-dir embeddings/
```

## `build_faiss_index.py`

Builds a `faiss.IndexFlatIP` (exact cosine-similarity search on L2-normalized vectors)
over an embedded firehose sample — the haystack `search_neighbors.py` searches for each
labeled post's nearest unlabeled match in.

```bash
python3 build_faiss_index.py --embeddings-dir firehose_embeddings/ --output firehose.index
```

## `search_neighbors.py`

For every labeled post, finds its highest-ranked unlabeled, single-image match in the
firehose index — "the most semantically similar post BMS never flagged." Restricting
to single-image posts on both sides keeps a downstream Hive AI call (see
`call_hive.py`) well-defined.

```bash
python3 search_neighbors.py --index firehose.index --labeled-embeddings labeled_embeddings/ \
    --labeled-posts "out/batch_*.jsonl" --output nearest_neighbors.parquet
```

## `call_hive.py`

Calls Hive AI's visual moderation API for a set of post images and saves the raw
per-class scores returned. This only covers making the API call and recording its
output — the mapping from those raw scores to a moderation decision (Bluesky's own
Automod rule engine) is not included here.

```bash
export HIVE_API_KEY=...
python3 call_hive.py --input "out/batch_*.jsonl" --output hive_scores.parquet
```

## `cluster_umap_hdbscan.py`

Per label, projects post embeddings to 5D with UMAP and clusters that projection with
HDBSCAN, tuning `n_neighbors` / `min_cluster_size` / `min_samples` jointly with Optuna
against a DBCV validity score. Labels above `--downsample-above` posts are uniformly
downsampled first, purely so a trial finishes in reasonable time. UMAP and HDBSCAN run
on GPU via RAPIDS cuML — see `requirements.txt` for installing it.

```bash
python3 cluster_umap_hdbscan.py --embeddings-dir embeddings/intolerant/ \
    --label intolerant --output-dir clusters/intolerant/
```

Cluster naming/summarization (sampling representative posts per cluster and prompting
a vision-language model for a short label) is a separate step (prompts in paper).

## `data/bms_labeled_post_uris.parquet`

Raw post content (text/images/video) isn't redistributed here. Instead, this file
lists the **identifiers** of every post label collected — `at://` URI, CID, label,
embed type, label/post timestamps, and whether the row is an applied or a negated
(retracted) label — so the exact dataset can be reconstructed by re-fetching each URI
yourself (via `enrich_posts.py`, pointed at these URIs instead of a fresh
`collect_labels.py` run) rather than hosting a copy of other people's posts.

| column | meaning |
|---|---|
| `uri` | the post's `at://` URI |
| `cid` | the post's content-version id at label time |
| `label` | the label value |
| `neg` | `true` if this row is a negation/retraction of a previously applied label |
| `label_batch_source` | which internal collection batch this row came from (provenance only) |
| `post_type` | the post's embed `$type` at label time |
| `label_cts` | when the label (or negation) was created |
| `post_created_at` | the post's own `createdAt` |

~10.67M rows (10.46M applied + 0.21M negated). 

## Requirements

```bash
pip install -r requirements.txt
```