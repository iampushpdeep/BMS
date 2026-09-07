#!/usr/bin/env python3
"""Subscribe to Bluesky's main repo firehose (com.atproto.sync.subscribeRepos) and
persist every new post to disk, unfiltered by any labeler.

This is the *unlabeled* counterpart to collect_labels.py: subscribeRepos streams every
write across the entire network (posts, likes, follows, profile edits, ...), not just
posts a labeler has acted on -- so this is how you get a general/background sample of
Bluesky activity to compare against, rather than only the subset BMS has already
flagged.

Unlike a label event (a bare reference you must separately resolve), a firehose commit
carries the post's full record inline, CAR/CBOR-encoded -- so this script decodes it
directly and writes out the complete record, no PDS lookup required to get the text.
enrich_posts.py still resolves the author's PDS afterward, but only to download media,
never to re-fetch content already captured here.

Performance note: subscribeRepos' volume is far higher than a label stream's (every
write network-wide, not just moderation actions), so the read loop stays cheap by
filtering on commit.ops (already-decoded operation metadata) for post creates BEFORE
paying the cost of parsing the CAR block store -- the CAR parse only runs on the small
fraction of commits that actually contain one.

Usage:
    python3 collect_firehose.py --output posts.jsonl
    python3 collect_firehose.py --output posts.jsonl --queue-workers 8
"""
import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import libipld
from atproto import CAR, models, parse_subscribe_repos_message
from atproto_firehose import AsyncFirehoseSubscribeReposClient
from atproto_firehose.models import MessageFrame


def sanitize_for_json(obj):
    """A record decoded with strict=False can carry nested CID references (e.g.
    reply.parent.cid) as raw multihash bytes instead of validated CID strings, which
    aren't valid UTF-8 and crash plain JSON serialization. Re-encode any stray bytes as
    proper CID strings with the same library that parsed the CAR blocks in the first
    place; leave everything else untouched."""
    if isinstance(obj, bytes):
        try:
            return libipld.encode_cid(obj)
        except Exception:
            return obj.hex()
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    return obj


def load_cursor(cursor_path: Path) -> int | None:
    if not cursor_path.exists():
        return None
    try:
        return int(cursor_path.read_text().strip())
    except (ValueError, OSError):
        return None


def save_cursor(cursor_path: Path, seq: int) -> None:
    try:
        cursor_path.write_text(str(seq))
    except OSError:
        pass


class FirehoseCollector:
    def __init__(self, output_path: Path, cursor_path: Path, queue_maxsize: int, n_workers: int,
                 reconnect_backoff_start: float = 1.0, reconnect_backoff_max: float = 30.0):
        self.output_path = output_path
        self.cursor_path = cursor_path
        self.reconnect_backoff_start = reconnect_backoff_start
        self.reconnect_backoff_max = reconnect_backoff_max
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=queue_maxsize)
        self.n_workers = n_workers
        self.n_posts = 0
        self.n_dropped = 0
        self.last_seq: int | None = None
        self._out_lock = asyncio.Lock()
        self._client: AsyncFirehoseSubscribeReposClient | None = None

    async def run_forever(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        workers = [asyncio.create_task(self._worker()) for _ in range(self.n_workers)]
        try:
            backoff = self.reconnect_backoff_start
            while True:
                cursor = self.last_seq if self.last_seq is not None else load_cursor(self.cursor_path)
                self._client = AsyncFirehoseSubscribeReposClient(
                    params={"cursor": cursor} if cursor else None
                )
                connected_at = time.monotonic()
                self._log(f"connecting from cursor {cursor}")
                try:
                    await self._client.start(self._on_message)
                    return  # a clean stop (e.g. our own client.stop()) -- don't reconnect
                except Exception as e:
                    if time.monotonic() - connected_at > 60.0:
                        backoff = self.reconnect_backoff_start
                    self._log(f"disconnected ({e!r}) -- reconnecting from cursor={self.last_seq} in {backoff:.1f}s")
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, self.reconnect_backoff_max)
        finally:
            for w in workers:
                w.cancel()
            await asyncio.gather(*workers, return_exceptions=True)

    async def stop(self) -> None:
        if self._client is not None:
            await self._client.stop()  # coroutine on the async client -- must be awaited

    async def _on_message(self, message: MessageFrame) -> None:
        """Never awaits network/disk I/O -- just decodes the commit envelope (cheap)
        and hands off to a queue worker, so a slow worker never stalls the websocket
        read loop (the actual cause of the relay disconnecting you with
        FirehoseError: ConsumerTooSlow if this loop falls behind)."""
        try:
            commit = parse_subscribe_repos_message(message)
        except Exception:
            return
        if not isinstance(commit, models.ComAtprotoSyncSubscribeRepos.Commit):
            return
        self.last_seq = commit.seq
        if not commit.blocks:
            return
        matching_ops = [
            op for op in commit.ops
            if op.action == "create" and op.cid is not None and op.path.startswith("app.bsky.feed.post/")
        ]
        if not matching_ops:
            return
        try:
            self.queue.put_nowait((commit, matching_ops))
        except asyncio.QueueFull:
            self.n_dropped += 1

    async def _worker(self) -> None:
        while True:
            commit, matching_ops = await self.queue.get()
            try:
                self._handle_commit(commit, matching_ops)
            except Exception as e:
                self._log(f"worker failed on a commit: {e!r}")
            finally:
                self.queue.task_done()

    def _handle_commit(self, commit, matching_ops) -> None:
        try:
            car = CAR.from_bytes(commit.blocks)
        except Exception:
            return

        did = commit.repo
        lines = []
        for op in matching_ops:
            raw = car.blocks.get(op.cid)
            if raw is None:
                continue
            try:
                record = models.get_or_create(raw, strict=False)
            except Exception:
                continue
            if not isinstance(record, models.AppBskyFeedPost.Record):
                continue

            rkey = op.path.split("/")[-1]
            record_dict = sanitize_for_json(record.model_dump(mode="python", exclude_none=True, by_alias=True))
            event = {
                "kind": "post",
                "uri": f"at://{did}/app.bsky.feed.post/{rkey}",
                "cid": str(op.cid),
                "did": did,
                "cts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),  # when WE observed it
                "record": record_dict,
                "seq": commit.seq,
            }
            lines.append(json.dumps(event, ensure_ascii=False))

        if lines:
            with open(self.output_path, "a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            self.n_posts += len(lines)
            save_cursor(self.cursor_path, commit.seq)
            if self.n_posts % 500 < len(lines):
                self._log(f"{self.n_posts} posts collected (cursor={commit.seq}, "
                          f"queue={self.queue.qsize()}, dropped={self.n_dropped})")

    def _log(self, msg: str) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        print(f"[{ts}] {msg}", flush=True)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, type=Path, help="JSONL file to append post records to")
    parser.add_argument("--cursor-file", type=Path, default=None,
                         help="where to persist the resume cursor (default: <output>.cursor)")
    parser.add_argument("--queue-maxsize", type=int, default=10000,
                         help="max commits buffered between the read loop and CAR-decode workers "
                              "before new ones are dropped (default: 10000)")
    parser.add_argument("--queue-workers", type=int, default=4,
                         help="number of tasks decoding commits concurrently (default: 4)")
    args = parser.parse_args()

    cursor_path = args.cursor_file or args.output.with_suffix(args.output.suffix + ".cursor")
    collector = FirehoseCollector(args.output, cursor_path, args.queue_maxsize, args.queue_workers)
    try:
        await collector.run_forever()
    except (KeyboardInterrupt, asyncio.CancelledError):
        await collector.stop()
        print(f"\nstopped. {collector.n_posts} posts written to {args.output} "
              f"({collector.n_dropped} commits dropped under backpressure)")


if __name__ == "__main__":
    asyncio.run(main())
