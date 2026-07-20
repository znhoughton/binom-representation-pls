"""
Training-dynamics pipeline: run the full by-layer analysis at log-spaced
training checkpoints of all three OPT-BabyLM 20-epoch models.

Checkpoints are stored as HuggingFace tags named step-N.
Each model was trained for 20 epochs with a different batch size:
    125M  — every 24 steps  → 3,984 total steps  (batch ~512)
    350M  — every 48 steps  → 7,968 total steps  (batch ~256)
    1.3B  — every ~97 steps → 15,908 total steps (batch ~128)

Log-spaced steps are chosen as a fixed fraction of each model's total training,
rounded to the nearest available tag.

Usage:
    python Scripts/run_checkpoint_pipeline.py [--models 125m 350m 1.3b]
                                              [--gpu 0]
                                              [--skip-controls]
                                              [--skip-corpus-freq]
"""

import argparse
import csv
import lzma
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "0")

PYTHON  = sys.executable
BASE    = Path(__file__).resolve().parent.parent
SCRIPTS = BASE / "Scripts"

# Fractions of total training at which to sample (log-spaced, ~8 points)
# Covers: ~0.6%, 1.5%, 3.8%, 9.4%, 23%, 57%, 100%
LOG_FRACTIONS = [0.006, 0.015, 0.038, 0.094, 0.23, 0.57]

MODELS = [
    {
        "flag":       "125m",
        "id":         "znhoughton/opt-babylm-125m-20eps-seed964",
        "n_layers":   12,
        "mlp_batch":  262144,
        "total_steps": 3984,
        "step_interval": 24,
    },
    {
        "flag":       "350m",
        "id":         "znhoughton/opt-babylm-350m-20eps-seed964",
        "n_layers":   24,
        "mlp_batch":  262144,
        "total_steps": 7968,
        "step_interval": 48,
    },
    {
        "flag":       "1.3b",
        "id":         "znhoughton/opt-babylm-1.3b-20eps-seed964",
        "n_layers":   24,
        "mlp_batch":  262144,
        "total_steps": 15908,
        "step_interval": 97,
    },
]

CONDITIONS = [
    {"name": "default",     "context": "binomial",    "extract": "word",
     "corpus_dir": "embeddings",             "novel_dir": "novel_embeddings"},
    {"name": "attn_zeroed", "context": "attn_zeroed", "extract": "word",
     "corpus_dir": "embeddings_attn_zeroed", "novel_dir": "novel_embeddings_attn_zeroed"},
]


def log_spaced_steps(total: int, interval: int) -> list[int]:
    """Return log-spaced steps rounded to the nearest available tag."""
    steps = []
    for frac in LOG_FRACTIONS:
        raw = frac * total
        nearest = round(raw / interval) * interval
        nearest = max(interval, min(nearest, total))
        if nearest not in steps:
            steps.append(nearest)
    return sorted(set(steps))


def slug_for(model_flag: str, step: int) -> str:
    return f"znhoughton_opt-babylm-{model_flag}-20eps-seed964_step{step}"


def banner(msg):
    sep = "=" * 64
    print(f"\n{sep}\n  {msg}\n{sep}", flush=True)


def _fmt_h(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    if h > 0:
        return f"{h}h {m:02d}m"
    return f"{m}m"


def run(cmd, label="", abort_on_fail=True):
    banner(label or " ".join(str(c) for c in cmd[:6]))
    t0 = time.perf_counter()
    rc = subprocess.call([str(c) for c in cmd])
    elapsed = time.perf_counter() - t0
    status = "OK" if rc == 0 else f"FAILED (exit {rc})"
    print(f"  {status} in {elapsed:.0f}s ({elapsed/60:.1f}m)", flush=True)
    if rc != 0 and abort_on_fail:
        sys.exit(rc)


def run_parallel(cmds_labels: list, abort_on_fail=True):
    """Launch all (cmd, label) pairs simultaneously and wait for all to finish."""
    procs = []
    for cmd, label in cmds_labels:
        banner(label)
        procs.append(subprocess.Popen([str(c) for c in cmd]))
    t0 = time.perf_counter()
    failed = False
    for p in procs:
        if p.wait() != 0:
            failed = True
    elapsed = time.perf_counter() - t0
    print(f"  {'OK' if not failed else 'FAILED'} in {elapsed:.0f}s ({elapsed/60:.1f}m)", flush=True)
    if failed and abort_on_fail:
        sys.exit(1)


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
    """Compress by_layer_corpus_pred.csv → .gz and remove the original."""
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
    print(f"  {src_mb:.0f} MB → {dst_mb:.0f} MB", flush=True)
    src.unlink()


def step_complete(slug: str) -> bool:
    """True when corpus-freq predictions are compressed (last thing written per step)."""
    return _pred_compressed(slug)


def run_step(model: dict, step: int, gpu: int, emb_dir: Path,
             skip_controls: bool, skip_corpus_freq: bool, force: bool = False,
             checkpoint_index: int = None, n_checkpoints: int = None) -> bool:
    """Returns True if the step ran, False if it was already complete and skipped."""
    slug = slug_for(model["flag"], step)
    if not force and step_complete(slug):
        banner(f"MODEL {model['flag'].upper()}  step={step}  ALREADY COMPLETE — skipping")
        return False
    banner(f"MODEL {model['flag'].upper()}  step={step}  ({step/model['total_steps']*100:.1f}% of training)")

    # ── Extract (one condition at a time — GPU-heavy) ──────────────────────────
    for cond in CONDITIONS:
        run(
            [PYTHON, SCRIPTS / "run_by_layer_pipeline.py",
             "--models",    model["flag"],
             "--conditions", cond["name"],
             "--gpu",       str(gpu),
             "--skip-mlp",
             "--embeddings-dir", str(emb_dir),
             "--model-id",  model["id"],
             "--checkpoint", str(step),
             "--slug",      slug]
            + (["--force"] if force else []),
            label=f"EXTRACT  {model['flag']} step={step} / {cond['name']}",
        )

    force_flag = ["--force"] if force else []

    ckpt_flags = ([f"--checkpoint-index", str(checkpoint_index),
                   "--n-checkpoints",    str(n_checkpoints)]
                  if checkpoint_index is not None else [])

    all_conds = [c["name"] for c in CONDITIONS]

    def mlp_cmd(*extra):
        return [PYTHON, SCRIPTS / "by_layer_mlp.py",
                "--model-slug", slug,
                "--num-layers", str(model["n_layers"]),
                "--conditions", *all_conds,
                "--modes", "mean_pooled", "individual", "words_only",
                "--splits", "pair_novel", "word_novel",
                "--gpu",    str(gpu),
                "--batch",  str(model["mlp_batch"]),
                "--embeddings-dir", str(emb_dir),
                *ckpt_flags, *force_flag, *extra]

    # ── MLP CV — both conditions in one process (avoids GPU OOM from parallel) ──
    run(mlp_cmd(),
        label=f"MLP CV  {model['flag']} step={step}")

    # ── Corpus-freq — both conditions in one process ───────────────────────────
    if not skip_corpus_freq:
        run([PYTHON, SCRIPTS / "by_layer_mlp.py",
             "--model-slug", slug,
             "--num-layers", str(model["n_layers"]),
             "--conditions", *all_conds,
             "--modes", "mean_pooled", "individual", "words_only",
             "--corpus-freq",
             "--gpu",    str(gpu),
             "--embeddings-dir", str(emb_dir),
             *force_flag],
            label=f"CORPUS-FREQ  {model['flag']} step={step}")

    # ── Controls — both conditions in one process ──────────────────────────────
    if not skip_controls:
        run(mlp_cmd("--control"),
            label=f"CONTROLS  {model['flag']} step={step}")

    # ── Free disk ──────────────────────────────────────────────────────────────
    for cond in CONDITIONS:
        delete_dir(emb_dir / cond["novel_dir"]  / slug)
        delete_dir(emb_dir / cond["corpus_dir"] / slug)

    return True  # caller fires compress_corpus_pred in background


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=None,
                   choices=["125m", "350m", "1.3b"])
    p.add_argument("--gpu",   type=int, default=0)
    p.add_argument("--skip-controls",    action="store_true", dest="skip_controls")
    p.add_argument("--skip-corpus-freq", action="store_true", dest="skip_corpus_freq")
    p.add_argument("--embeddings-dir",   default=None,        dest="embeddings_dir")
    p.add_argument("--force",            action="store_true",
                   help="Re-run all steps even if output already exists")
    args = p.parse_args()

    emb_dir    = Path(args.embeddings_dir) if args.embeddings_dir else BASE / "Data"
    model_list = [m for m in MODELS
                  if args.models is None or m["flag"] in args.models]

    # Print the selected steps for each model upfront
    print("Training-dynamics checkpoint plan:", flush=True)
    for m in model_list:
        steps = log_spaced_steps(m["total_steps"], m["step_interval"])
        fracs = [f"{s/m['total_steps']*100:.1f}%" for s in steps]
        print(f"  {m['flag']:5s}: {steps}  ({', '.join(fracs)})", flush=True)
    print(flush=True)

    t_pipeline = time.perf_counter()
    all_step_times: list = []
    compress_thread: threading.Thread = None

    for m_idx, model in enumerate(model_list):
        steps = log_spaced_steps(model["total_steps"], model["step_interval"])
        step_times: list = []
        t0_model = time.perf_counter()

        for i, step in enumerate(steps):
            # Finish any background compression before step_complete checks _pred_compressed
            if compress_thread is not None:
                compress_thread.join()
                compress_thread = None

            remaining_after = len(steps) - i - 1
            print(f"\n  Checkpoint {i+1}/{len(steps)}  (step={step}, {remaining_after} remaining)", flush=True)
            t0_ckpt = time.perf_counter()
            ran = run_step(model, step, args.gpu, emb_dir,
                           args.skip_controls, args.skip_corpus_freq, args.force,
                           checkpoint_index=i + 1, n_checkpoints=len(steps))
            ckpt_elapsed = time.perf_counter() - t0_ckpt

            if ran:
                slug = slug_for(model["flag"], step)
                compress_thread = threading.Thread(
                    target=compress_corpus_pred, args=(slug,), daemon=True)
                compress_thread.start()
                print(f"  [compressing {slug}/by_layer_corpus_pred.csv in background]", flush=True)

                step_times.append(ckpt_elapsed)
                all_step_times.append(ckpt_elapsed)
                avg_model = sum(step_times) / len(step_times)
                avg_all   = sum(all_step_times) / len(all_step_times)
                remaining_total = remaining_after + sum(
                    len(log_spaced_steps(m["total_steps"], m["step_interval"]))
                    for m in model_list[m_idx + 1:]
                )
                print(f"\n  ── Timing ──────────────────────────────────────────────────────", flush=True)
                print(f"     This checkpoint:              {_fmt_h(ckpt_elapsed)}", flush=True)
                print(f"     Avg / checkpoint ({model['flag']}):    {_fmt_h(avg_model)}  (n={len(step_times)})", flush=True)
                if remaining_after > 0:
                    print(f"     Est. remaining in {model['flag']}:    {_fmt_h(avg_model * remaining_after)}", flush=True)
                if remaining_total > 0:
                    print(f"     Est. remaining in phase 1:    {_fmt_h(avg_all * remaining_total)}", flush=True)

        if compress_thread is not None:
            compress_thread.join()
            compress_thread = None

        model_elapsed = time.perf_counter() - t0_model
        banner(f"MODEL {model['flag'].upper()} ALL STEPS DONE  ({_fmt_h(model_elapsed)})")

    banner("CHECKPOINT PIPELINE COMPLETE")
    print(f"  Total: {_fmt_h(time.perf_counter() - t_pipeline)}", flush=True)


if __name__ == "__main__":
    main()
