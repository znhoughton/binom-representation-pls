"""
Training-dynamics checkpoint pipeline for Pythia models.

Checkpoints are selected by TOKEN COUNT rather than training fraction so that
results are directly comparable to the BabyLM checkpoint analysis (Sections 6–8
of pilot_results_2.md).  The key design question is:

    "When Pythia has seen the same number of tokens as BabyLM (≈2B), does it
     already show the abstraction pattern?  And what happens as it continues?"

Pythia training details (all sizes, The Pile, 1 epoch):
    Total tokens    : 299,892,736,000 ≈ 300B
    Tokens per step : 2,097,152  (1024 sequences × 2048 tokens)
    Total steps     : 143,000
    Checkpoints     : step1, step2, step4, step8, step16, step32, step64,
                      step128, step256, step512,
                      then step1000, step2000, … step143000 (every 1 000 steps)

BabyLM token counts for reference (2B total, all three sizes):
    step48  → ~24M      step144 → ~72M      step384 → ~193M
    step912 → ~458M     step2280 → ~1.14B   final   → 2.0B (= 100% training)

Nearest Pythia steps to each BabyLM token count:
    ~24M  → step16  (33.6M)    ~72M  → step32  (67.1M)
    ~193M → step64  (134M)     ~458M → step256 (537M)
    ~1.1B → step512 (1.07B)    ~2.0B → step1000 (2.10B)  ← BabyLM boundary
    Extended beyond 2B: step2000, step5000, step14000, step33000, step143000

Pythia tag format: "step1000" (no hyphen) — uses --revision, not --checkpoint.

Usage:
    python Scripts/run_pythia_checkpoints.py [--models 160m 1b]
                                             [--gpu 0]
                                             [--range babylm|extended|all]
                                             [--skip-controls]
                                             [--skip-corpus-freq]
"""

import argparse
import lzma
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

PYTHON  = sys.executable
BASE    = Path(__file__).resolve().parent.parent
SCRIPTS = BASE / "Scripts"

# Tokens per Pythia training step (1024 seqs × 2048 tokens)
TOKENS_PER_STEP = 2_097_152

# ── Checkpoint schedule ────────────────────────────────────────────────────────
# "babylm": nearest Pythia steps to BabyLM's 6 log-spaced checkpoints
# "extended": beyond the BabyLM 2B boundary, up to full training
# "all": both lists combined

BABYLM_COMPARABLE_STEPS = [16, 32, 64, 256, 512, 1000]
# Token counts:  33.6M  67.1M  134M  537M  1.07B  2.10B

EXTENDED_STEPS = [2000, 5000, 14000, 33000, 143000]
# Token counts:  4.2B   10.5B  29.4B  69.2B  300B (final)

ALL_STEPS = BABYLM_COMPARABLE_STEPS + EXTENDED_STEPS

# ── Model registry ─────────────────────────────────────────────────────────────
MODELS = [
    {
        "flag":      "160m",
        "id":        "EleutherAI/pythia-160m",
        "n_layers":  12,
        "mlp_batch": 262144,
        "batch_size": 1024,
    },
    {
        "flag":      "410m",
        "id":        "EleutherAI/pythia-410m",
        "n_layers":  24,
        "mlp_batch": 262144,
        "batch_size": 512,
    },
    {
        "flag":      "1b",
        "id":        "EleutherAI/pythia-1b",
        "n_layers":  16,
        "mlp_batch": 262144,
        "batch_size": 512,
    },
    {
        "flag":      "2.8b",
        "id":        "EleutherAI/pythia-2.8b",
        "n_layers":  32,
        "mlp_batch": 131072,
        "batch_size": 256,
    },
]

CONDITIONS = [
    {"name": "default",     "context": "binomial",    "extract": "word",
     "corpus_dir": "embeddings",             "novel_dir": "novel_embeddings"},
    {"name": "attn_zeroed", "context": "attn_zeroed", "extract": "word",
     "corpus_dir": "embeddings_attn_zeroed", "novel_dir": "novel_embeddings_attn_zeroed"},
]


def tokens(step: int) -> str:
    t = step * TOKENS_PER_STEP
    if t >= 1e9:
        return f"{t/1e9:.2f}B"
    return f"{t/1e6:.0f}M"


def revision(step: int) -> str:
    """Pythia HF tag format: no hyphen."""
    return f"step{step}"


def slug_for(model_flag: str, step: int) -> str:
    return f"EleutherAI_pythia-{model_flag}_step{step}"


def banner(msg):
    sep = "=" * 64
    print(f"\n{sep}\n  {msg}\n{sep}", flush=True)


def _fmt_h(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    if h > 0:
        return f"{h}h {m:02d}m"
    return f"{m}m"


def _bar(done: int, total: int, width: int = 24) -> str:
    filled = round(width * done / total) if total else 0
    return f"[{'█' * filled}{'░' * (width - filled)}] {done}/{total}"


def run(cmd, label="", abort_on_fail=True) -> bool:
    banner(label or " ".join(str(c) for c in cmd[:6]))
    t0 = time.perf_counter()
    rc = subprocess.call([str(c) for c in cmd])
    elapsed = time.perf_counter() - t0
    status = "OK" if rc == 0 else f"FAILED (exit {rc})"
    print(f"  {status} in {elapsed:.0f}s ({elapsed/60:.1f}m)", flush=True)
    if rc != 0 and abort_on_fail:
        sys.exit(rc)
    return rc == 0


def delete_dir(path, label=""):
    path = Path(path)
    if path.exists():
        size_gb = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9
        banner(f"DELETE  {label or path.name}  ({size_gb:.1f} GB)")
        shutil.rmtree(path)
        print("  Deleted.", flush=True)


def _pred_compressed(slug: str) -> bool:
    r = BASE / "Results" / slug
    return (r / "by_layer_corpus_pred.csv.gz").exists() or \
           (r / "by_layer_corpus_pred.csv.xz").exists()


def compress_corpus_pred(slug: str):
    src = BASE / "Results" / slug / "by_layer_corpus_pred.csv"
    dst = BASE / "Results" / slug / "by_layer_corpus_pred.csv.xz"
    if not src.exists():
        return
    if _pred_compressed(slug):
        src.unlink(missing_ok=True)
        return
    banner(f"COMPRESS  {slug}/by_layer_corpus_pred.csv")
    src_mb = src.stat().st_size / 1e6
    with open(src, "rb") as f_in, lzma.open(dst, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    dst_mb = dst.stat().st_size / 1e6
    print(f"  {src_mb:.0f} MB → {dst_mb:.0f} MB ({dst_mb/src_mb*100:.0f}%)", flush=True)
    src.unlink()


def step_complete(slug: str) -> bool:
    return _pred_compressed(slug)


def calibrate_batch_size(model_id: str, max_bs: int, gpu: int,
                         revision: str = None) -> int:
    cmd = [PYTHON, SCRIPTS / "extract_embeddings.py",
           "--model",      model_id,
           "--data",       "corpus",
           "--batch-size", str(max_bs),
           "--gpu",        str(gpu),
           "--calibrate"]
    if revision is not None:
        cmd += ["--revision", revision]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", flush=True)
    for line in result.stdout.splitlines():
        if line.startswith("SAFE_BATCH_SIZE="):
            bs = int(line.split("=")[1])
            print(f"  → Calibrated batch_size={bs} for {model_id}", flush=True)
            return bs
    fallback = min(512, max_bs)
    print(f"  WARNING: calibration failed; using fallback bs={fallback}", flush=True)
    return fallback


def run_step(model: dict, step: int, effective_bs: int, gpu: int, emb_dir: Path,
             skip_controls: bool, skip_corpus_freq: bool, force: bool = False,
             checkpoint_index: int = None, n_checkpoints: int = None) -> bool:
    """Returns True if the step ran, False if already complete and skipped."""
    slug = slug_for(model["flag"], step)
    if not force and step_complete(slug):
        banner(f"PYTHIA-{model['flag'].upper()}  step={step}  ALREADY COMPLETE — skipping")
        return False

    rev = revision(step)
    tok = tokens(step)
    banner(f"PYTHIA-{model['flag'].upper()}  step={step}  ({tok} tokens)")

    for cond in CONDITIONS:
        ok = run(
            [PYTHON, SCRIPTS / "run_by_layer_pipeline.py",
             "--conditions",     cond["name"],
             "--gpu",            str(gpu),
             "--skip-mlp",
             "--embeddings-dir", str(emb_dir),
             "--model-id",       model["id"],
             "--num-layers",     str(model["n_layers"]),
             "--revision",       rev,
             "--slug",           slug,
             "--extract-batch-size", str(effective_bs)]
            + (["--force"] if force else []),
            label=f"EXTRACT  pythia-{model['flag']} step={step} / {cond['name']}",
            abort_on_fail=False,
        )
        if not ok:
            print(f"  Extraction failed for {slug} / {cond['name']} — skipping step.", flush=True)
            return False

    force_flag = ["--force"] if force else []
    all_conds  = [c["name"] for c in CONDITIONS]

    _n_layers = model["n_layers"] + 1
    _n_conds  = len(all_conds)
    _n_modes  = 2
    _n_splits = 2
    _main_n   = _n_conds * _n_layers * _n_modes * _n_splits
    _freq_n   = _n_conds * _n_layers * _n_modes if not skip_corpus_freq else 0
    _ctrl_n   = _n_conds * _n_layers * _n_modes * _n_splits if not skip_controls else 0
    _global_n = _main_n + _freq_n + _ctrl_n
    _freq_off = _main_n
    _ctrl_off = _main_n + _freq_n

    ckpt_flags = (["--checkpoint-index", str(checkpoint_index),
                   "--n-checkpoints",    str(n_checkpoints)]
                  if checkpoint_index is not None else [])

    def mlp_cmd(*extra, global_offset=0):
        return [PYTHON, SCRIPTS / "by_layer_mlp.py",
                "--model-slug", slug,
                "--num-layers", str(model["n_layers"]),
                "--conditions", *all_conds,
                "--modes",      "mean_pooled", "words_only",
                "--splits",     "pair_novel",  "word_novel",
                "--gpu",        str(gpu),
                "--batch",      str(model["mlp_batch"]),
                "--embeddings-dir", str(emb_dir),
                "--global-n",       str(_global_n),
                "--global-offset",  str(global_offset),
                *ckpt_flags, *force_flag, *extra]

    run(mlp_cmd(), label=f"MLP CV  pythia-{model['flag']} step={step}")

    if not skip_corpus_freq:
        run([PYTHON, SCRIPTS / "by_layer_mlp.py",
             "--model-slug",    slug,
             "--num-layers",    str(model["n_layers"]),
             "--conditions",    *all_conds,
             "--modes",         "mean_pooled", "words_only",
             "--corpus-freq",
             "--gpu",           str(gpu),
             "--embeddings-dir", str(emb_dir),
             "--global-n",       str(_global_n),
             "--global-offset",  str(_freq_off),
             *force_flag],
            label=f"CORPUS-FREQ  pythia-{model['flag']} step={step}")

    if not skip_controls:
        run(mlp_cmd("--control", global_offset=_ctrl_off),
            label=f"CONTROLS  pythia-{model['flag']} step={step}")

    for cond in CONDITIONS:
        delete_dir(emb_dir / cond["novel_dir"]  / slug)
        delete_dir(emb_dir / cond["corpus_dir"] / slug)

    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=None,
                   choices=["160m", "410m", "1b", "2.8b"],
                   help="Pythia model sizes to run (default: all)")
    p.add_argument("--range", default="all", dest="ckpt_range",
                   choices=["babylm", "extended", "all"],
                   help="babylm: only the 6 steps matching BabyLM token counts (≤2.1B); "
                        "extended: only the 5 steps beyond 2B tokens; "
                        "all: all 11 checkpoints (default)")
    p.add_argument("--gpu",            type=int,  default=0)
    p.add_argument("--embeddings-dir", default=None, dest="embeddings_dir")
    p.add_argument("--skip-controls",    action="store_true", dest="skip_controls")
    p.add_argument("--skip-corpus-freq", action="store_true", dest="skip_corpus_freq")
    p.add_argument("--force",            action="store_true")
    args = p.parse_args()

    emb_dir = Path(args.embeddings_dir) if args.embeddings_dir else BASE / "Data"

    if args.ckpt_range == "babylm":
        steps = BABYLM_COMPARABLE_STEPS
    elif args.ckpt_range == "extended":
        steps = EXTENDED_STEPS
    else:
        steps = ALL_STEPS

    model_list = [m for m in MODELS
                  if args.models is None or m["flag"] in args.models]

    print("Pythia checkpoint plan:", flush=True)
    print(f"  Steps: {steps}", flush=True)
    print(f"  Token counts: {[tokens(s) for s in steps]}", flush=True)
    print(f"  Note: step1000 ≈ 2.1B tokens = BabyLM full training boundary", flush=True)
    print(f"  Models: {[m['flag'] for m in model_list]}", flush=True)
    print(flush=True)

    t_pipeline = time.perf_counter()
    all_step_times: list = []
    compress_thread: threading.Thread = None
    total_all = len(model_list) * len(steps)
    done_all  = 0

    for model in model_list:
        step_times: list = []
        t0_model = time.perf_counter()

        for i, step in enumerate(steps):
            if compress_thread is not None:
                compress_thread.join()
                compress_thread = None

            print(f"\n  Checkpoint {i+1}/{len(steps)}  "
                  f"(step={step}, ~{tokens(step)} tokens)", flush=True)

            effective_bs = model["batch_size"]

            t0_ckpt = time.perf_counter()
            ran = run_step(model, step, effective_bs, args.gpu, emb_dir,
                           args.skip_controls, args.skip_corpus_freq, args.force,
                           checkpoint_index=i + 1, n_checkpoints=len(steps))
            ckpt_elapsed = time.perf_counter() - t0_ckpt
            done_all += 1

            if ran:
                slug = slug_for(model["flag"], step)
                compress_thread = threading.Thread(
                    target=compress_corpus_pred, args=(slug,), daemon=True)
                compress_thread.start()
                print(f"  [compressing {slug}/by_layer_corpus_pred.csv in background]",
                      flush=True)

                step_times.append(ckpt_elapsed)
                all_step_times.append(ckpt_elapsed)
                done_model = i + 1
                avg_model  = sum(step_times) / len(step_times)
                avg_all    = sum(all_step_times) / len(all_step_times)
                model_left = len(steps) - done_model
                phase_left = total_all - done_all

                model_eta = (f"  est. {_fmt_h(avg_model * model_left)} remaining"
                             if model_left else "  done")
                phase_eta = (f"  est. {_fmt_h(avg_all   * phase_left)} remaining"
                             if phase_left else "  done")

                print(f"\n  ── Progress ─────────────────────────────────────────────────", flush=True)
                print(f"     pythia-{model['flag']:5s}  {_bar(done_model, len(steps))}{model_eta}", flush=True)
                print(f"     overall      {_bar(done_all, total_all)}{phase_eta}", flush=True)
                print(f"\n  ── Timing ───────────────────────────────────────────────────", flush=True)
                print(f"     This checkpoint: {_fmt_h(ckpt_elapsed)}", flush=True)
                print(f"     Avg / checkpoint (pythia-{model['flag']}): "
                      f"{_fmt_h(avg_model)}  (n={len(step_times)})", flush=True)

        if compress_thread is not None:
            compress_thread.join()
            compress_thread = None

        banner(f"PYTHIA-{model['flag'].upper()} ALL STEPS DONE  "
               f"({_fmt_h(time.perf_counter() - t0_model)})")

    banner("PYTHIA CHECKPOINT PIPELINE COMPLETE")
    print(f"  Total: {_fmt_h(time.perf_counter() - t_pipeline)}", flush=True)


if __name__ == "__main__":
    main()
