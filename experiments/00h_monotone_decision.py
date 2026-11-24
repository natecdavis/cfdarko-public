"""
Step 00h: Monotone Constraint A/B Decision

Diffs holdout metrics from two paired tuning sweeps (unconstrained vs.
all-on monotone constraints) and prints which (pos × model) heads should
keep the constraint. Decision rule mirrors the original threshold:
keep iff Δ Spearman ≥ +KEEP_THRESHOLD (default 0.02).

Inputs (override via CLI):
    --sweep-a-perpos    /tmp/sweep_A/perpos_training_metrics.json
    --sweep-b-perpos    /tmp/sweep_B/perpos_training_metrics.json
    --sweep-a-global    /tmp/sweep_A/global_training_metrics.json
    --sweep-b-global    /tmp/sweep_B/global_training_metrics.json
    --keep-threshold    0.02  (Δ Spearman cutoff)

Output:
    Sorted table with sp_A, sp_B, Δ, n_train, decision per head.
    A literal MONOTONE_ENABLED set ready to paste into config.py.
    Caution flags for tiny-n models (n < TINY_N) and large drops (Δ ≤ −0.05).

Usage:
    python experiments/00h_monotone_decision.py
    python experiments/00h_monotone_decision.py --keep-threshold 0.03
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_KEEP_THRESHOLD = 0.02
TINY_N = 250
LARGE_DROP = -0.05

GLOBAL_HEADS = ("global_role", "global_impact_apy", "global_impact_wav")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-a-perpos",
                        default="/tmp/sweep_A/perpos_training_metrics.json")
    parser.add_argument("--sweep-b-perpos",
                        default="/tmp/sweep_B/perpos_training_metrics.json")
    parser.add_argument("--sweep-a-global",
                        default="/tmp/sweep_A/global_training_metrics.json")
    parser.add_argument("--sweep-b-global",
                        default="/tmp/sweep_B/global_training_metrics.json")
    parser.add_argument("--keep-threshold", type=float,
                        default=DEFAULT_KEEP_THRESHOLD)
    return parser.parse_args()


def _load(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        logger.error("Missing metrics file: %s", p)
        sys.exit(1)
    return json.loads(p.read_text())


def _sp(d: dict, key: str):
    v = d.get(key, {}).get("holdout_spearman")
    return float(v) if isinstance(v, (int, float)) else None


def _n(d: dict, key: str):
    v = d.get(key, {}).get("n_full_train")
    return int(v) if isinstance(v, (int, float)) else None


def main():
    args = parse_args()
    keep = float(args.keep_threshold)

    a_pp = _load(args.sweep_a_perpos)
    b_pp = _load(args.sweep_b_perpos)
    a_gl = _load(args.sweep_a_global)
    b_gl = _load(args.sweep_b_global)

    keys = sorted(set(a_pp) | set(b_pp))
    rows = []
    for k in keys:
        sp_a = _sp(a_pp, k)
        sp_b = _sp(b_pp, k)
        n = _n(a_pp, k) or _n(b_pp, k) or 0
        delta = (sp_b - sp_a) if (sp_a is not None and sp_b is not None) else None
        keep_flag = (delta is not None) and (delta >= keep)
        rows.append((k, sp_a, sp_b, delta, n, keep_flag))

    rows.sort(key=lambda r: (-(r[3] or -9.0), r[0]))

    print()
    print("=== Per-Position Heads ===")
    print(f"{'pos/model':<22} {'sp_A':>8} {'sp_B':>8} {'Δ':>8} {'n_train':>9}  decision  notes")
    for k, sp_a, sp_b, d, n, keep_flag in rows:
        sp_a_s = f"{sp_a:>8.3f}" if sp_a is not None else f"{'?':>8}"
        sp_b_s = f"{sp_b:>8.3f}" if sp_b is not None else f"{'?':>8}"
        d_s = f"{d:>+8.3f}" if d is not None else f"{'?':>8}"
        decision = "KEEP" if keep_flag else "drop"
        notes = []
        if d is not None and d <= LARGE_DROP:
            notes.append("LARGE DROP")
        if n and n < TINY_N:
            notes.append(f"tiny n={n}")
        print(f"{k:<22} {sp_a_s} {sp_b_s} {d_s} {n:>9}  {decision:<8}  {' '.join(notes)}")

    print()
    print("=== Global Heads ===")
    global_deltas = []
    for h in GLOBAL_HEADS:
        sp_a = _sp(a_gl, h)
        sp_b = _sp(b_gl, h)
        n = _n(a_gl, h) or _n(b_gl, h) or 0
        delta = (sp_b - sp_a) if (sp_a is not None and sp_b is not None) else None
        sp_a_s = f"{sp_a:>8.3f}" if sp_a is not None else f"{'?':>8}"
        sp_b_s = f"{sp_b:>8.3f}" if sp_b is not None else f"{'?':>8}"
        d_s = f"{delta:>+8.3f}" if delta is not None else f"{'?':>8}"
        notes = []
        if delta is not None and delta <= LARGE_DROP:
            notes.append("LARGE DROP")
        if n and n < TINY_N:
            notes.append(f"tiny n={n}")
        print(f"{h:<22} {sp_a_s} {sp_b_s} {d_s} {n:>9}            {' '.join(notes)}")
        if delta is not None:
            global_deltas.append(delta)

    keep_global = (
        len(global_deltas) > 0 and (sum(global_deltas) / len(global_deltas)) >= keep
    )
    if global_deltas:
        mean_d = sum(global_deltas) / len(global_deltas)
        print(f"{'global mean Δ':<22} {'':>8} {'':>8} {mean_d:>+8.3f} "
              f"            decision: {'KEEP global' if keep_global else 'drop global'}")

    print()
    print("=== Suggested MONOTONE_ENABLED ===")
    enabled = sorted(
        (tuple(k.split("_", 1)) for k, _a, _b, _d, _n, kf in rows if kf),
        key=lambda x: (x[0], x[1]),
    )
    print("MONOTONE_ENABLED: set[tuple[str, str]] = {")
    for pos, model in enabled:
        print(f'    ("{pos}", "{model}"),')
    if keep_global:
        print('    ("__global__", "__global__"),')
    print("}")
    print()
    print(f"(Decision threshold: Δ Spearman ≥ {keep:+.3f})")


if __name__ == "__main__":
    main()
