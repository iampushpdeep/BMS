#!/usr/bin/env python3
"""Compute labeling delay ("timeliness") from enriched post records, and use it as an
empirical proxy for automation vs. human oversight.

For every enriched row (see enrich_posts.py), delay is defined as the time between the
post's own creation and the label being applied to it:

    delay = label_cts - record["createdAt"]

A large volume of near-zero delays for a label indicates it is applied automatically;
delays spread across hours or days indicate a human-in-the-loop review step. Two kinds
of bad rows are dropped before that comparison means anything:

  - backdated posts (record["createdAt"] before --cutoff-date): bulk-imported or
    migrated content whose recorded creation time isn't trustworthy, which would
    otherwise show up as an enormous (and meaningless) delay.
  - negative delays (the label's cts is earlier than the post's own createdAt): almost
    always a sign the post's own timestamp is wrong, not that time ran backwards.

Usage:
    python3 compute_timeliness.py --input "out/batch_*.jsonl" --output timeliness.parquet
    python3 compute_timeliness.py --input "out/batch_*.jsonl" --output timeliness.parquet \
        --cutoff-date 2024-06-01 --percentiles-out delay_percentiles.csv
"""
import argparse
import glob
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DEFAULT_CUTOFF = "2024-06-01"  # adjust to whenever your collection actually started


def iter_rows(paths: list[str]):
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def build_dataframe(paths: list[str]) -> pd.DataFrame:
    records = []
    for row in iter_rows(paths):
        record = row.get("record") or {}
        created_at = record.get("createdAt")
        if not created_at:
            continue
        records.append({
            "uri": row.get("uri"),
            "cid": row.get("cid"),
            "label": row.get("label"),
            "neg": bool(row.get("neg", False)),
            "post_type": row.get("etype") or "text-only",
            "label_cts": row.get("cts"),
            "post_created_at": created_at,
        })
    return pd.DataFrame.from_records(records)


def compute_delay(df: pd.DataFrame, cutoff_date: str) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    df["label_cts"] = pd.to_datetime(df["label_cts"], utc=True, errors="coerce")
    df["post_created_at"] = pd.to_datetime(df["post_created_at"], utc=True, errors="coerce")

    total = len(df)
    df = df.dropna(subset=["label_cts", "post_created_at"])
    no_ts = total - len(df)

    cutoff = pd.Timestamp(cutoff_date, tz="utc")
    backdated_mask = df["post_created_at"] < cutoff
    ca_out = int(backdated_mask.sum())
    df = df[~backdated_mask]

    df["timeliness_seconds"] = (df["label_cts"] - df["post_created_at"]).dt.total_seconds()
    neg_mask = df["timeliness_seconds"] < 0
    cts_lt = int(neg_mask.sum())
    df = df[~neg_mask]

    df["timeliness_hours"] = df["timeliness_seconds"] / 3600.0
    df["timeliness_days"] = df["timeliness_seconds"] / 86400.0

    stats = {
        "total": total, "valid": len(df), "no_ts": no_ts,
        "ca_out": ca_out, "cts_lt": cts_lt,
    }
    return df, stats


def percentile_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, g in df[~df["neg"]].groupby("label"):
        s = g["timeliness_seconds"]
        rows.append({
            "label": label, "n": len(s),
            "p5": s.quantile(0.05), "p25": s.quantile(0.25),
            "median": s.quantile(0.50), "p75": s.quantile(0.75),
            "p95": s.quantile(0.95),
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="glob pattern for enriched batch_*.jsonl files")
    parser.add_argument("--output", required=True, type=Path, help="output parquet path")
    parser.add_argument("--cutoff-date", default=DEFAULT_CUTOFF,
                         help=f"drop posts created before this date, ISO format (default: {DEFAULT_CUTOFF})")
    parser.add_argument("--percentiles-out", type=Path, default=None,
                         help="also write a per-label delay percentile summary CSV")
    args = parser.parse_args()

    paths = sorted(glob.glob(args.input))
    if not paths:
        raise SystemExit(f"no files matched: {args.input}")
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] reading {len(paths)} file(s)...")

    df = build_dataframe(paths)
    print(f"loaded {len(df):,} rows")

    df, stats = compute_delay(df, args.cutoff_date)
    print(f"valid: {stats['valid']:,} / {stats['total']:,}  "
          f"(dropped {stats['no_ts']:,} missing-timestamp, {stats['ca_out']:,} backdated, "
          f"{stats['cts_lt']:,} negative-delay)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"wrote {args.output}")

    if args.percentiles_out:
        table = percentile_table(df)
        table.to_csv(args.percentiles_out, index=False)
        print(f"wrote {args.percentiles_out}")
        with pd.option_context("display.float_format", "{:.1f}".format):
            print(table.to_string(index=False))


if __name__ == "__main__":
    main()
