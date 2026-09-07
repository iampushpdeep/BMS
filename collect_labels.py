#!/usr/bin/env python3
"""Subscribe to Bluesky's official Moderation Service (BMS) label stream and persist
every label event to disk, verbatim.

BMS publishes moderation decisions in real time over com.atproto.label.subscribeLabels
(a websocket firehose), not a REST endpoint you can page through -- so "collection" here
means staying connected and writing out whatever arrives, indefinitely. This script does
exactly that and nothing else: it does not resolve posts or download media (see
enrich_posts.py for that), so it stays lightweight enough to run for a long time
unattended.

Output is JSONL, one label event per line, with the fields the AT Protocol label lexicon
(com.atproto.label.defs#label) defines:
    uri   -- the subject being labeled (a post's at:// URI, or a bare DID for an
             account-level label)
    cid   -- the specific content version being labeled, if the subject is a post
    val   -- the label value itself (e.g. "spam", "rude", "porn")
    neg   -- true if this event RETRACTS a previously-applied label (val/uri/cid
             identify which one); most events are additions, not retractions
    cts   -- when the label was created (ISO 8601, UTC)
    src   -- DID of the labeler that issued it (constant here: the BMS DID)
    exp   -- expiry timestamp, if the label is time-limited (usually null)
    ver   -- label protocol version (usually null; present for forward compatibility)
    seq   -- the firehose sequence number this event arrived on, saved as our resume
             cursor (not part of the label lexicon itself)

Resumable: the sequence number of the last event written is saved to a cursor file, so
a restart picks up from where it left off (subscribeLabels replays from any past
cursor) instead of silently skipping or re-fetching everything. Never dies on a
transient error -- any exception just triggers a reconnect with exponential backoff.

Usage:
    python3 collect_labels.py --output labels.jsonl
    python3 collect_labels.py --output labels.jsonl --labeler wss://mod.bsky.app/xrpc
"""
import argparse
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from atproto import (
    FirehoseSubscribeLabelsClient,
    firehose_models,
    models,
    parse_subscribe_labels_message,
)

DEFAULT_LABELER = "wss://mod.bsky.app/xrpc"  # Bluesky's own first-party Moderation Service


def load_cursor(cursor_path: Path) -> int | None:
    """None means "start at the live tip" -- subscribeLabels treats a missing cursor
    this way, but treats cursor=0 as "replay the entire history," which is almost never
    what you want for a fresh run."""
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


class LabelCollector:
    def __init__(self, labeler_uri: str, output_path: Path, cursor_path: Path,
                 reconnect_backoff_start: float = 1.0, reconnect_backoff_max: float = 30.0):
        self.labeler_uri = labeler_uri
        self.output_path = output_path
        self.cursor_path = cursor_path
        self.reconnect_backoff_start = reconnect_backoff_start
        self.reconnect_backoff_max = reconnect_backoff_max
        self._stop = threading.Event()
        self._client: FirehoseSubscribeLabelsClient | None = None
        self.n_events = 0
        self.n_reconnects = 0

    def stop(self) -> None:
        self._stop.set()
        if self._client is not None:
            try:
                self._client.stop()
            except Exception:
                pass

    def run_forever(self) -> None:
        backoff = self.reconnect_backoff_start
        with open(self.output_path, "a", encoding="utf-8") as out_f:
            while not self._stop.is_set():
                try:
                    self._run_once(out_f)
                    backoff = self.reconnect_backoff_start
                except Exception as e:
                    self.n_reconnects += 1
                    self._log(f"error: {e!r} -- reconnecting in {backoff:.1f}s")
                if self._stop.is_set():
                    return
                time.sleep(backoff)
                backoff = min(backoff * 2, self.reconnect_backoff_max)

    def _run_once(self, out_f) -> None:
        cursor = load_cursor(self.cursor_path)
        params = models.ComAtprotoLabelSubscribeLabels.Params(cursor=cursor)
        self._client = FirehoseSubscribeLabelsClient(params=params, base_uri=self.labeler_uri)
        labels_cls = models.ComAtprotoLabelSubscribeLabels.Labels
        info_cls = models.ComAtprotoLabelSubscribeLabels.Info

        def on_message(msg: firehose_models.MessageFrame) -> None:
            if self._stop.is_set():
                try:
                    self._client.stop()
                except Exception:
                    pass
                return
            try:
                parsed = parse_subscribe_labels_message(msg)
            except Exception:
                return  # a single malformed frame is not worth killing the stream over
            if isinstance(parsed, info_cls) or not isinstance(parsed, labels_cls):
                return

            for label in parsed.labels:
                event = {
                    "uri": label.uri,
                    "cid": label.cid,
                    "val": label.val,
                    "neg": bool(getattr(label, "neg", False)),
                    "cts": label.cts,
                    "src": label.src,
                    "exp": getattr(label, "exp", None),
                    "ver": getattr(label, "ver", None),
                    "seq": parsed.seq,
                }
                out_f.write(json.dumps(event, ensure_ascii=False) + "\n")
                out_f.flush()
                self.n_events += 1

            if parsed.seq:
                save_cursor(self.cursor_path, parsed.seq)
            if self.n_events % 500 == 0:
                self._log(f"{self.n_events} events collected (cursor={parsed.seq})")

        self._log(f"connecting to {self.labeler_uri} from cursor {cursor}")
        self._client.start(on_message)

    def _log(self, msg: str) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        print(f"[{ts}] {msg}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, type=Path, help="JSONL file to append label events to")
    parser.add_argument("--labeler", default=DEFAULT_LABELER, help=f"labeler websocket base URI (default: {DEFAULT_LABELER})")
    parser.add_argument("--cursor-file", type=Path, default=None,
                         help="where to persist the resume cursor (default: <output>.cursor)")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cursor_path = args.cursor_file or args.output.with_suffix(args.output.suffix + ".cursor")

    collector = LabelCollector(args.labeler, args.output, cursor_path)
    try:
        collector.run_forever()
    except KeyboardInterrupt:
        collector.stop()
        print(f"\nstopped. {collector.n_events} events written to {args.output}")


if __name__ == "__main__":
    main()
