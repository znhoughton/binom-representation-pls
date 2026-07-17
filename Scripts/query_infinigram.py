"""
Query infini-gram for both orderings of binomial pairs.

Usage:
    python query_infinigram.py [--input PATH] [--output PATH]
                               [--index INDEX] [--workers N] [--delay F]

Defaults:
    --input    ../Data/corpus_binomials.csv
    --output   ../Results/corpus_binomials_infinigram_<index_short>.csv
    --index    v4_dolma-v1_7_llama
    --workers  5      (keep low — API blocks on heavy load)
    --delay    2.0    (seconds between requests per worker)

Resume: re-running skips already-completed pairs (appends to existing output).
Pairs with freq=-1 (errors) are retried on re-run.

Output columns:
    word1, word2, freq_w1w2, freq_w2w1, freq_total

Common runs:
    # Corpus pairs, Dolma (priority — needed for frequency regression)
    python query_infinigram.py

    # Corpus pairs, Pile (for Pythia / GPT-2)
    python query_infinigram.py --index v4_piletrain_llama

    # Full novel set, Dolma (background, takes ~75h)
    python query_infinigram.py --input ../Data/wikipedia_novel_binomials.csv \\
        --output ../Results/novel_binomials_infinigram_dolma.csv

    # Full novel set, Pile
    python query_infinigram.py --input ../Data/wikipedia_novel_binomials.csv \\
        --index v4_piletrain_llama \\
        --output ../Results/novel_binomials_infinigram_pile.csv
"""

import sys
import argparse
import csv
import json
import time
import urllib.request
import urllib.error
import concurrent.futures
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

API_URL          = "https://api.infini-gram.io/"
DEFAULT_INDEX    = "v4_dolma-v1_7_llama"
RATE_LIMIT_WAIT  = 60.0   # seconds to wait when server returns 403/429/503
BACKOFF_BASE     = 2.0    # seconds; doubles on each transient retry
MAX_RETRIES      = 8
BATCH_SIZE       = 100    # rows flushed per batch


# ── API ────────────────────────────────────────────────────────────────────────

def query_count(phrase: str, index: str, request_delay: float) -> int:
    """Return exact n-gram count, or -1 on persistent failure.

    403/429/503 are treated as rate-limit signals: wait RATE_LIMIT_WAIT seconds
    and retry indefinitely (without consuming a transient retry slot).
    """
    payload = json.dumps({"index": index, "query_type": "count", "query": phrase}).encode()
    transient = 0
    while transient < MAX_RETRIES:
        time.sleep(request_delay)
        try:
            req = urllib.request.Request(
                API_URL, data=payload,
                headers={"Content-Type": "application/json",
                         "User-Agent": "Mozilla/5.0 (research; binom-infinigram)"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
                if "error" in data:
                    msg = str(data["error"]).lower()
                    if any(kw in msg for kw in ("rate", "limit", "lock", "quota")):
                        print(f"  [rate-limit body] '{phrase}' — waiting {RATE_LIMIT_WAIT:.0f}s",
                              flush=True)
                        time.sleep(RATE_LIMIT_WAIT)
                        continue
                    raise ValueError(f"API error: {data['error']}")
                return int(data.get("count", 0))

        except urllib.error.HTTPError as e:
            if e.code in (403, 429, 503):
                print(f"  [rate-limit HTTP {e.code}] '{phrase}' — waiting {RATE_LIMIT_WAIT:.0f}s",
                      flush=True)
                time.sleep(RATE_LIMIT_WAIT)
                continue   # don't consume a transient slot
            wait = BACKOFF_BASE ** transient
            print(f"  [retry {transient+1}/{MAX_RETRIES}] '{phrase}': HTTP {e.code} — {wait:.0f}s",
                  flush=True)
            time.sleep(wait)
            transient += 1

        except (urllib.error.URLError, TimeoutError, ValueError, KeyError) as e:
            wait = BACKOFF_BASE ** transient
            print(f"  [retry {transient+1}/{MAX_RETRIES}] '{phrase}': {e} — {wait:.0f}s",
                  flush=True)
            time.sleep(wait)
            transient += 1

    print(f"  [ERROR] giving up on '{phrase}' after {MAX_RETRIES} retries.", flush=True)
    return -1


def query_pair(word1: str, word2: str, index: str, request_delay: float) -> dict:
    f12 = query_count(f"{word1} and {word2}", index, request_delay)
    f21 = query_count(f"{word2} and {word1}", index, request_delay)
    return {
        "word1": word1,
        "word2": word2,
        "freq_w1w2": f12,
        "freq_w2w1": f21,
        "freq_total": (f12 + f21) if (f12 >= 0 and f21 >= 0) else -1,
    }


# ── I/O ────────────────────────────────────────────────────────────────────────

def load_pairs(path: Path) -> list[tuple[str, str]]:
    pairs = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            w1 = row["word1"].strip().lower()
            w2 = row["word2"].strip().lower()
            if w1 and w2:
                pairs.append((w1, w2))
    return pairs


def load_done(path: Path) -> set[tuple[str, str]]:
    done = set()
    if not path.exists():
        return done
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["freq_w1w2"]) >= 0 and int(row["freq_w2w1"]) >= 0:
                done.add((row["word1"], row["word2"]))
    return done


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",   default=None)
    p.add_argument("--output",  default=None)
    p.add_argument("--index",   default=DEFAULT_INDEX)
    p.add_argument("--workers", type=int,   default=5)
    p.add_argument("--delay",   type=float, default=2.0)
    args = p.parse_args()

    base = Path(__file__).parent.parent
    input_path = Path(args.input) if args.input else base / "Data" / "corpus_binomials.csv"

    if args.output:
        output_path = Path(args.output)
    else:
        short = args.index.replace("v4_", "").replace("_llama", "").replace("-v1_7", "")
        stem  = input_path.stem
        output_path = base / "Results" / f"{stem}_infinigram_{short}.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_pairs = load_pairs(input_path)
    done      = load_done(output_path)
    todo      = [(w1, w2) for w1, w2 in all_pairs if (w1, w2) not in done]

    n_total = len(all_pairs)
    n_done  = len(done)

    # Estimate runtime
    pairs_per_sec = args.workers / (args.delay * 2)   # 2 queries per pair, sequential within worker
    eta_h = len(todo) / pairs_per_sec / 3600 if todo else 0

    print(f"Index:      {args.index}")
    print(f"Input:      {input_path}  ({n_total:,} pairs)")
    print(f"Output:     {output_path}")
    print(f"Workers:    {args.workers}  |  Delay: {args.delay}s/req")
    print(f"Done:       {n_done:,}  |  Remaining: {len(todo):,}  |  ETA: {eta_h:.1f}h", flush=True)

    if not todo:
        print("Nothing to do.")
        return

    write_header = not output_path.exists() or output_path.stat().st_size == 0
    out_f  = open(output_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_f, fieldnames=["word1","word2","freq_w1w2","freq_w2w1","freq_total"])
    if write_header:
        writer.writeheader()
        out_f.flush()

    t_start     = time.time()
    n_completed = 0
    n_errors    = 0
    buffer      = []

    def flush_buffer():
        writer.writerows(buffer)
        out_f.flush()
        buffer.clear()

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(query_pair, w1, w2, args.index, args.delay): (w1, w2)
            for w1, w2 in todo
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            buffer.append(result)
            n_completed += 1
            if result["freq_total"] < 0:
                n_errors += 1

            if len(buffer) >= BATCH_SIZE:
                flush_buffer()

            if n_completed % 200 == 0 or n_completed == len(todo):
                elapsed   = time.time() - t_start
                rate      = n_completed / elapsed if elapsed > 0 else 0
                remaining = (len(todo) - n_completed) / rate / 3600 if rate > 0 else 0
                print(
                    f"  {n_done + n_completed:,}/{n_total:,}  "
                    f"({rate:.2f} pairs/s  ETA {remaining:.1f}h  errors={n_errors})",
                    flush=True
                )

    flush_buffer()
    out_f.close()

    elapsed = time.time() - t_start
    print(f"\nDone. {n_completed:,} pairs in {elapsed/3600:.2f}h. Errors: {n_errors:,}.")
    if n_errors:
        print("Re-run to retry failed pairs (freq=-1 rows are skipped on resume).")


if __name__ == "__main__":
    main()
