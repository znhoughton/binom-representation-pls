"""
run_parallel_extraction.py
--------------------------
Launch N parallel shard processes for corpus or Wikipedia extraction,
wait for all to finish, then merge shards into the final CSV.

Usage:
  python run_parallel_extraction.py --mode corpus --num-shards 5
  python run_parallel_extraction.py --mode novel  --num-shards 5
  python run_parallel_extraction.py --mode corpus --num-shards 5 --merge-only
  python run_parallel_extraction.py --mode corpus --num-shards 5 --stagger 30
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

PYTHON   = r"C:\Users\zacha\anaconda3\envs\PRenv\python.exe"
SCRIPTS  = Path(__file__).resolve().parent

CORPUS_SCRIPT = SCRIPTS / "extract_corpus_binomials.py"
NOVEL_SCRIPT  = SCRIPTS / "extract_wikipedia_binomials.py"
MERGE_SCRIPT  = SCRIPTS / "merge_extraction_shards.py"


def launch_shards(script, mode, num_shards, stagger):
    print(f"\nLaunching {num_shards} parallel {mode} extraction shards"
          + (f" (staggered {stagger}s apart)..." if stagger else "..."))
    procs = []
    for i in range(num_shards):
        cmd = [PYTHON, str(script),
               "--num-shards", str(num_shards),
               "--shard-index", str(i)]
        log = SCRIPTS / f"{mode}_shard{i:02d}of{num_shards:02d}.log"
        print(f"  Shard {i:02d}: {log.name}")
        env = {**__import__("os").environ,
               "OMP_NUM_THREADS": "1",
               "MKL_NUM_THREADS": "1",
               "OPENBLAS_NUM_THREADS": "1"}
        with open(log, "w", encoding="utf-8") as f:
            p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        procs.append((i, p, log))
        if stagger and i < num_shards - 1:
            time.sleep(stagger)

    print(f"\nAll {num_shards} shards running. Waiting for completion...")
    failed = []
    for i, p, log in procs:
        p.wait()
        if p.returncode != 0:
            failed.append(i)
            print(f"  [FAIL] Shard {i:02d} (exit {p.returncode}) — see {log.name}")
        else:
            print(f"  [OK]   Shard {i:02d}")

    if failed:
        print(f"\n{len(failed)} shard(s) failed: {failed}")
        print("Fix errors and re-run with --merge-only once all shards are done.")
        sys.exit(1)

    print("\nAll shards completed successfully.")


def merge(mode, num_shards):
    print(f"\nMerging {num_shards} {mode} shards...")
    result = subprocess.run(
        [PYTHON, str(MERGE_SCRIPT), "--mode", mode, "--num-shards", str(num_shards)],
        capture_output=False
    )
    if result.returncode != 0:
        print("Merge failed.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",       choices=["corpus", "novel"], required=True)
    parser.add_argument("--num-shards", type=int, default=5)
    parser.add_argument("--stagger",    type=int, default=0,
                        help="Seconds to wait between launching each shard (default: 0)")
    parser.add_argument("--merge-only", action="store_true",
                        help="Skip extraction; only merge existing shard files")
    args = parser.parse_args()

    script = CORPUS_SCRIPT if args.mode == "corpus" else NOVEL_SCRIPT

    if not args.merge_only:
        launch_shards(script, args.mode, args.num_shards, args.stagger)

    merge(args.mode, args.num_shards)
    print(f"\nDone. Final file: Data/{args.mode}_binomials.csv"
          if args.mode == "corpus"
          else "\nDone. Final file: Data/wikipedia_novel_binomials.csv")


if __name__ == "__main__":
    main()
