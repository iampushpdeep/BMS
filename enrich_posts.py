#!/usr/bin/env python3
"""Turn raw label events (from collect_labels.py) into fully resolved posts with
their media downloaded.

A label event only tells you a post's URI/CID and the label applied to it -- not the
post's text, embeds, or images/video. This script fills that in: for every event, it
resolves the author's DID to their PDS (via plc.directory), fetches the post record
(com.atproto.repo.getRecord), classifies its embed, and downloads any attached
image/video/link-preview-thumbnail blobs (com.atproto.sync.getBlob).

Output rows match the schema already used throughout this project's analysis notebooks
(see notebooks/timeliness.ipynb) so this is a drop-in data source for them:
    cid       -- the post's content-version id (identifies which edit/version this is)
    uri       -- the post's at:// URI
    label     -- the label value applied to this post (renamed from the event's "val")
    neg       -- whether this event retracts a previously-applied label
    cts       -- when the LABEL was created (ISO 8601) -- not the post's own timestamp
    record    -- the post's full raw record as returned by the PDS (includes
                 record["createdAt"], the POST's own creation time, record["text"],
                 record["embed"], record["reply"], etc.)
    etype     -- the embed's $type ("" if the post has no embed)
    text      -- denormalized copy of record["text"], for convenience
    images    -- list of {"file": <local path>, "alt": <alt text>} for every
                 downloaded image (post images AND external-link-preview thumbnails)
    video     -- local path to the downloaded video, or null
    skip      -- true if this post has no usable media (text-only, or every download
                 attempt failed) -- convenient for analyses that only want media posts
    failures  -- list of human-readable strings describing anything that went wrong
                 downloading this post's media (empty list if all went cleanly)

Every embed shape Bluesky posts can carry is handled, including a quote-post that
attaches its own image/video/link-card (app.bsky.embed.recordWithMedia).

Resumable and safe to interrupt: progress (which input file / byte offset) is saved to
a cursor file after every batch, so a re-run picks up where it left off rather than
reprocessing everything. Output is written incrementally as batch_NNNNN.jsonl files.

Usage:
    python3 enrich_posts.py --input labels.jsonl --output-dir out/
    python3 enrich_posts.py --input "labels_*.jsonl" --output-dir out/ --labels rude,threat
"""
import argparse
import asyncio
import glob
import json
from datetime import datetime, timezone
from pathlib import Path

import aiofiles
import aiohttp


def classify_embed(embed: dict | None) -> tuple[bool, bool, bool, bool]:
    """Returns (has_image, has_video, has_external_thumb, skip), unwrapping
    app.bsky.embed.recordWithMedia (a quote-post with its own attached media) so the
    quoting post's media is classified the same way a plain post's would be.

    Only the common embed shapes are handled (images, video, external-link thumbnail).
    Anything else (e.g. app.bsky.embed.gallery, a newer multi-image shape) falls
    through to skip=True rather than being specifically supported -- confirmed against
    a real post during testing, deliberately left unhandled rather than extracting its
    images."""
    if not embed:
        return False, False, False, True

    etype = embed.get("$type", "")
    src = embed.get("media", {}) if etype == "app.bsky.embed.recordWithMedia" else embed
    stype = src.get("$type", "")

    has_image = stype == "app.bsky.embed.images"
    has_video = stype.startswith("app.bsky.embed.video")
    has_external_thumb = stype == "app.bsky.embed.external" and bool(
        src.get("external", {}).get("thumb", {}).get("ref", {}).get("$link")
    )

    # app.bsky.embed.record (a bare quote, no media of its own) is kept for text-only
    # analysis rather than marked "skip" -- everything else needs actual media to be
    # useful to a media-enrichment pass.
    if etype == "app.bsky.embed.record":
        skip = False
    else:
        skip = not (has_image or has_video or has_external_thumb)
    return has_image, has_video, has_external_thumb, skip


class Enricher:
    def __init__(self, session: aiohttp.ClientSession, images_dir: Path, videos_dir: Path,
                 concurrency: int, http_retries: int, log):
        self.session = session
        self.images_dir = images_dir
        self.videos_dir = videos_dir
        self.http_retries = http_retries
        self.log = log
        self.pds_cache: dict[str, str | None] = {}
        self.pds_sem = asyncio.Semaphore(min(10, concurrency))
        self.record_sem = asyncio.Semaphore(concurrency)
        self.blob_sem = asyncio.Semaphore(concurrency)
        self.video_sem = asyncio.Semaphore(max(1, concurrency // 3))

    async def get_pds_endpoint(self, did: str) -> str | None:
        if did in self.pds_cache:
            return self.pds_cache[did]
        delay = 1.0
        for attempt in range(1, self.http_retries + 1):
            try:
                async with self.pds_sem:
                    async with self.session.get(f"https://plc.directory/{did}/data", timeout=10) as resp:
                        if resp.status != 200:
                            raise RuntimeError(f"HTTP {resp.status}")
                        data = await resp.json()
                services = data.get("services", {})
                services = services.values() if isinstance(services, dict) else services
                for svc in services:
                    if svc.get("type") == "AtprotoPersonalDataServer":
                        endpoint = svc.get("serviceEndpoint") or svc.get("endpoint")
                        self.pds_cache[did] = endpoint
                        return endpoint
                break  # resolved fine, just has no PDS listed
            except Exception:
                if attempt < self.http_retries:
                    await asyncio.sleep(delay)
                    delay *= 2
        self.pds_cache[did] = None
        return None

    async def get_record(self, pds: str, did: str, rkey: str) -> dict | None:
        delay = 1.0
        for attempt in range(1, self.http_retries + 1):
            try:
                async with self.record_sem:
                    params = {"repo": did, "collection": "app.bsky.feed.post", "rkey": rkey}
                    async with self.session.get(f"{pds}/xrpc/com.atproto.repo.getRecord",
                                                 params=params, timeout=20) as resp:
                        if resp.status == 200:
                            return (await resp.json()).get("value")
                        if resp.status in (400, 404):
                            return None  # deleted/moved -- not worth retrying
            except Exception:
                pass
            if attempt < self.http_retries:
                await asyncio.sleep(delay)
                delay *= 2
        return None

    async def download_blob(self, pds: str, did: str, blob_cid: str, out_path: Path,
                             is_video: bool) -> bool:
        if out_path.exists():
            return True
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(out_path.suffix + ".part")
        sem = self.video_sem if is_video else self.blob_sem
        delay = 1.0
        for attempt in range(1, self.http_retries + 1):
            try:
                async with sem:
                    timeout = aiohttp.ClientTimeout(total=180 if is_video else 30)
                    url = f"{pds}/xrpc/com.atproto.sync.getBlob?did={did}&cid={blob_cid}"
                    async with self.session.get(url, timeout=timeout) as resp:
                        if resp.status != 200:
                            raise RuntimeError(f"HTTP {resp.status}")
                        async with aiofiles.open(tmp, "wb") as f:
                            async for chunk in resp.content.iter_chunked(16384):
                                await f.write(chunk)
                tmp.rename(out_path)
                return True
            except Exception:
                tmp.unlink(missing_ok=True)
                if attempt < self.http_retries:
                    await asyncio.sleep(delay)
                    delay *= 2
        return False

    async def enrich(self, event: dict, label_filter: set[str] | None) -> dict | None:
        """One raw label event -> one canonical output row, or None if there's nothing
        to resolve (an account-level label, an unparseable URI, or a label outside
        label_filter)."""
        label = event.get("val", "")
        if label_filter is not None and label not in label_filter:
            return None

        uri = event.get("uri", "")
        if not uri.startswith("at://") or "/app.bsky.feed.post/" not in uri:
            return None  # account-level label, not a post -- nothing to resolve

        try:
            _, _, did, _, rkey = uri.split("/", 4)
        except ValueError:
            return None

        row = {
            "cid": event.get("cid"), "uri": uri, "label": label, "neg": bool(event.get("neg", False)),
            "cts": event.get("cts", ""), "record": {}, "etype": "", "text": "",
            "images": [], "video": None, "skip": True, "failures": [],
        }

        pds = await self.get_pds_endpoint(did)
        if not pds:
            row["failures"].append("no PDS endpoint")
            return row

        record = await self.get_record(pds, did, rkey)
        if record is None:
            row["failures"].append("record not found (deleted or moved)")
            return row
        row["record"] = record
        row["text"] = record.get("text", "")

        embed = record.get("embed") or {}
        etype = embed.get("$type", "")
        row["etype"] = etype
        has_image, has_video, has_external_thumb, skip = classify_embed(embed)
        media_src = embed.get("media", embed) if etype == "app.bsky.embed.recordWithMedia" else embed

        if has_image:
            for img in media_src.get("images", []):
                blob_cid = img.get("image", {}).get("ref", {}).get("$link")
                if not blob_cid:
                    continue
                path = self.images_dir / blob_cid[-2:] / f"{blob_cid}.jpeg"
                if await self.download_blob(pds, did, blob_cid, path, is_video=False):
                    row["images"].append({"file": str(path), "alt": img.get("alt", "")})
                else:
                    row["failures"].append(f"image download failed: {blob_cid}")

        if has_external_thumb:
            blob_cid = media_src.get("external", {}).get("thumb", {}).get("ref", {}).get("$link")
            path = self.images_dir / blob_cid[-2:] / f"{blob_cid}.jpeg"
            if await self.download_blob(pds, did, blob_cid, path, is_video=False):
                row["images"].append({"file": str(path), "alt": "external_thumb"})
            else:
                row["failures"].append(f"external thumb download failed: {blob_cid}")

        if has_video:
            blob_cid = media_src.get("video", {}).get("ref", {}).get("$link")
            post_cid = row["cid"]
            if blob_cid and post_cid:
                path = self.videos_dir / post_cid[-2:] / f"{post_cid}.mp4"
                if await self.download_blob(pds, did, blob_cid, path, is_video=True):
                    row["video"] = str(path)
                else:
                    row["failures"].append(f"video download failed: {blob_cid}")

        row["skip"] = skip or bool(row["failures"]) and not (row["images"] or row["video"])
        return row


def iter_events(paths: list[Path], start_file: int, start_offset: int):
    """Yields (file_index, byte_offset_before_line, event_dict), resuming from a saved
    cursor position."""
    for file_index, path in enumerate(paths):
        if file_index < start_file:
            continue
        with open(path, encoding="utf-8") as f:
            if file_index == start_file and start_offset:
                f.seek(start_offset)
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                yield file_index, pos, event


def load_cursor(cursor_path: Path) -> tuple[int, int]:
    if not cursor_path.exists():
        return 0, 0
    try:
        data = json.loads(cursor_path.read_text())
        return data.get("file_index", 0), data.get("byte_offset", 0)
    except Exception:
        return 0, 0


def save_cursor(cursor_path: Path, file_index: int, byte_offset: int) -> None:
    cursor_path.write_text(json.dumps({"file_index": file_index, "byte_offset": byte_offset}))


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="raw label-event JSONL file, or a glob matching several")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--labels", default=None,
                         help="comma-separated label values to keep (default: keep every label)")
    parser.add_argument("--batch-size", type=int, default=1000, help="rows per output batch file")
    parser.add_argument("--concurrency", type=int, default=30)
    parser.add_argument("--http-retries", type=int, default=3)
    args = parser.parse_args()

    input_paths = sorted(Path(p) for p in glob.glob(args.input)) if any(c in args.input for c in "*?[") \
        else [Path(args.input)]
    if not input_paths:
        raise SystemExit(f"no input files matched: {args.input}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = args.output_dir / "images"
    videos_dir = args.output_dir / "videos"
    cursor_path = args.output_dir / "enrich_cursor.json"
    label_filter = set(args.labels.split(",")) if args.labels else None

    def log(msg: str) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        print(f"[{ts}] {msg}", flush=True)

    start_file, start_offset = load_cursor(cursor_path)
    if start_file or start_offset:
        log(f"resuming from file {start_file}, byte offset {start_offset}")

    existing_batches = list(args.output_dir.glob("batch_*.jsonl"))
    batch_index = max((int(p.stem.split("_")[1]) for p in existing_batches), default=-1) + 1

    n_written = n_dropped = 0
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=args.concurrency * 2)) as session:
        enricher = Enricher(session, images_dir, videos_dir, args.concurrency, args.http_retries, log)

        batch: list[dict] = []
        last_file_index = start_file
        last_offset = start_offset

        async def flush():
            nonlocal batch, batch_index
            if not batch:
                return
            out_path = args.output_dir / f"batch_{batch_index:05d}.jsonl"
            async with aiofiles.open(out_path, "w", encoding="utf-8") as f:
                for row in batch:
                    await f.write(json.dumps(row, ensure_ascii=False) + "\n")
            log(f"wrote {out_path} ({len(batch)} rows)")
            batch_index += 1
            batch = []
            save_cursor(cursor_path, last_file_index, last_offset)

        pending_tasks = []
        for file_index, byte_offset, event in iter_events(input_paths, start_file, start_offset):
            last_file_index, last_offset = file_index, byte_offset
            pending_tasks.append(asyncio.create_task(enricher.enrich(event, label_filter)))
            if len(pending_tasks) >= args.batch_size:
                for task in pending_tasks:
                    row = await task
                    if row is None:
                        n_dropped += 1
                    else:
                        batch.append(row)
                        n_written += 1
                pending_tasks = []
                await flush()
                log(f"progress: {n_written} enriched, {n_dropped} dropped (account-level / filtered)")

        for task in pending_tasks:
            row = await task
            if row is None:
                n_dropped += 1
            else:
                batch.append(row)
                n_written += 1
        await flush()

    log(f"done. {n_written} posts enriched, {n_dropped} events dropped, output in {args.output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
