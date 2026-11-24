"""
cfDARKO Pipeline Runner

Usage:
    python run_pipeline.py                    # full run (default — current behavior)
    python run_pipeline.py --test             # small test run
    python run_pipeline.py --from 4           # start from step 4
    python run_pipeline.py --step 2           # run only step 2

Tuning modes (mutually exclusive):
    python run_pipeline.py --no-tune          # fast iteration: skip every tunable
                                              # surface; reuse cached params
    python run_pipeline.py --retune           # full deep refit: chain in 04b/04d
                                              # calibration, re-run step 6, --tune
                                              # step 8/8b
    python run_pipeline.py --retune-only 4    # retune only the listed steps;
                                              # all other steps run in --no-tune
                                              # mode (steps comma-separated)

`--retune --test` is rejected (tuning on a 1k subset is meaningless).
"""

import argparse
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

STEPS = [
    # ── Data ingestion & player identity ──────────────────────────────────
    (1,   "pipeline/01_data_ingestion.py",       "Data Ingestion (CFBD API)"),
    # ── Play-by-play ingestion (slots 1.4, 1.5, 1.6) ──────────────────────
    (1.4, "pipeline/01d_play_ingestion.py",      "Play-by-Play Ingestion"),
    (1.5, "pipeline/01e_venue_orientation.py",   "Stadium Orientation (OSM)"),
    (1.6, "pipeline/01f_player_plays.py",        "Player-Plays Join"),
    (2,   "pipeline/02_bridge_players.py",       "Bridge Player Identification"),
    (3,   "pipeline/03_player_attributes.py",    "Player Attributes (height/weight/age)"),
    # ── Game-grain contextual models ──────────────────────────────────────
    (4,   "pipeline/04_contextual_models.py",    "Contextual Model Fitting"),
    # ── Per-play contextual models (slot 4.5) ─────────────────────────────
    (4.5, "pipeline/04p_play_contextual_models.py",
                                                 "Per-Play Contextual Models (LightGBM)"),
    # ── Residual computation ──────────────────────────────────────────────
    (5,   "pipeline/05_residual_computation.py", "Residual Computation"),
    # ── Per-play residual aggregation (slot 5.3) ──────────────────────────
    (5.3, "pipeline/05p_play_residual_aggregation.py",
                                                 "Per-Play Residual Aggregation"),
    # ── Skill estimation ──────────────────────────────────────────────────
    (6,   "pipeline/06_skill_estimation.py",     "Skill Estimation (Kalman + EWMA)"),
    # ── NFL labels & draft models ─────────────────────────────────────────
    (7,   "pipeline/07_nfl_labels.py",           "NFL Outcome Labels"),
    (7.5, "pipeline/07b_rookie_value.py",        "Rookie-Deal Value Label + Slot Curve"),
    (8,   "pipeline/08_draft_models.py",         "Draft Models (Impact)"),
    (8.5, "pipeline/08b_global_draft_model.py",  "Global Cross-Position Draft Model"),
    (8.7, "pipeline/08c_calibrate_draft_scores.py",
                                                 "Post-Hoc Calibration (impact_wav + global heads)"),
    (8.8, "pipeline/08d_offboard_model.py",      "Off-Board Model (merged board engine)"),
    # ── Comps & output ────────────────────────────────────────────────────
    (9,   "pipeline/09_comps.py",                "Draft Prospect Comps"),
    (10,  "pipeline/10_output.py",               "Final Output & Rankings"),
]

CALIBRATION_4B = (4.2, "pipeline/04b_calibrate_betas_role.py",   "Decay-Beta Calibration (04b)")
CALIBRATION_4D = (4.4, "pipeline/04d_calibrate_kalman_draft.py", "Kalman Calibration (04d)")
# 00h is intentionally not chained — it's an A/B comparison tool that requires
# pre-generated unconstrained vs monotone sweeps in /tmp/sweep_A and
# /tmp/sweep_B. Run it manually when you want to audit MONOTONE_ENABLED.

DEFAULT_TUNE_TRIALS = 30
DEFAULT_TUNE_SEEDS = 1


@dataclass
class StepInvocation:
    step_num: float
    script: str
    name: str
    extra_args: list[str] = field(default_factory=list)


def parse_args():
    parser = argparse.ArgumentParser(description="cfDARKO pipeline runner")
    parser.add_argument("--test", action="store_true",
                        help="Pass --test flag to each step (small subset)")
    parser.add_argument("--from", dest="from_step", type=float, default=1,
                        help="Start pipeline from this step number (default 1)")
    parser.add_argument("--step", type=float, default=None,
                        help="Run only this step number")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--no-tune", action="store_true",
                      help="Fast iteration mode: skip every tunable surface; "
                           "reuse cached params and refit on current data.")
    mode.add_argument("--retune", action="store_true",
                      help="Full deep refit: chain 04b/04d calibration + "
                           "--tune for step 8/8b. (00h monotone audit is a "
                           "separate manual tool — see banner at end of run.)")
    parser.add_argument("--retune-only", type=str, default=None,
                        help="Comma-separated step numbers to retune (e.g. '4,8'). "
                             "All other steps run in --no-tune mode.")
    parser.add_argument("--tune-trials", type=int, default=DEFAULT_TUNE_TRIALS,
                        help="Optuna trials per (pos, model) when retuning step 8/8b")
    parser.add_argument("--tune-seeds", type=int, default=DEFAULT_TUNE_SEEDS,
                        help="Number of independent Optuna seeds per head; winner "
                             "is the seed with the lowest CV. Costs N× wall time "
                             "for the tuning steps. Default 1 (legacy behavior).")
    return parser.parse_args()


# Steps with a retune surface (see _step_extra_args): 4 (alpha grid),
# 6 (decay-beta opt), 8 / 8.5 (Optuna + MI + Boruta).
RETUNABLE_STEPS: frozenset[float] = frozenset({4, 6, 8, 8.5})


def _parse_retune_only(spec: str | None) -> set[float]:
    if not spec:
        return set()
    out: set[float] = set()
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            step = float(tok)
        except ValueError:
            raise SystemExit(f"--retune-only: cannot parse '{tok}' as a step number")
        if step not in RETUNABLE_STEPS:
            valid = ", ".join(f"{s:g}" for s in sorted(RETUNABLE_STEPS))
            raise SystemExit(
                f"--retune-only: step '{tok}' has no retune behavior. "
                f"Valid steps: {valid}")
        out.add(step)
    return out


def _resolve_mode(args) -> tuple[str, set[float]]:
    """Returns (mode, retune_only_steps).

    mode is 'default' | 'no_tune' | 'retune'.
    retune_only steps override the per-step extra_args toward a retune even when
    the surrounding mode is 'no_tune' — i.e. surgical retune within fast mode.
    """
    retune_only = _parse_retune_only(args.retune_only)
    if args.retune:
        if args.test:
            raise SystemExit("--retune --test is rejected: tuning on a test subset is meaningless")
        return "retune", retune_only
    if args.no_tune or retune_only:
        return "no_tune", retune_only
    return "default", retune_only


def _step_extra_args(step_num: float, mode: str, retune_only: set[float],
                      tune_trials: int, tune_seeds: int) -> list[str]:
    """Map (step, mode) -> extra CLI args for that step's invocation.

    `retune_only` overrides toward retune for the listed steps regardless of mode.
    """
    retune_this = (mode == "retune") or (step_num in retune_only)
    no_tune_this = (mode == "no_tune") and (step_num not in retune_only)

    extra: list[str] = []
    if step_num == 4:
        if no_tune_this:
            extra.append("--use-cached-alpha")
        # default and retune: no flag (full grid is the existing default)
    elif step_num == 6:
        if no_tune_this:
            extra.append("--skip-beta-opt")
        elif step_num in retune_only and mode != "retune":
            # Surgical retune (--retune-only 6): force the inline beta
            # optimization, otherwise step 6 silently reuses the cached
            # betas and "retune step 6" is a no-op.
            extra.append("--force-beta-opt")
        # Note: FULL --retune deliberately does NOT pass --force-beta-opt.
        # The first step-6 run consumes whatever beta cache is on disk; 04b
        # then writes a fresh role-calibrated decay_betas_role.json after
        # step 7, and the re-run picks that up via the cache priority chain.
        # Forcing inline beta opt would *override* 04b's calibration since
        # step 6's loading priority puts --force-beta-opt above the role cache.
    elif step_num == 8:
        if retune_this:
            # MI filter computes the per-(pos × model) state-space subset
            # before tuning; tune + final fit then read the cache via
            # _filter_state_space_to_mi. Boruta runs as an audit pass (does
            # not gate fit; informs config.BORUTA_DROP_FEATURES manually).
            extra += ["--tune", "--mi-filter", "--boruta",
                      "--tune-trials", str(tune_trials),
                      "--tune-seeds", str(tune_seeds)]
    elif step_num == 8.5:
        if retune_this:
            extra += ["--tune", "--mi-filter", "--boruta",
                      "--tune-trials", str(tune_trials),
                      "--tune-seeds", str(tune_seeds)]
    return extra


def _build_invocations(args, mode: str, retune_only: set[float]
                        ) -> list[StepInvocation]:
    """Build the ordered invocation list. In retune mode, splice 04b/04d after
    step 7, then re-run step 6 (single run) so it consumes the fresh
    calibrations, and append the 00h advisory at the end.

    Gating flags read from config at build-time:
      - ENABLE_PLAY_LAYER: gates the play-grain chain 1.4 / 1.5 / 1.6 (ingest,
        stadium orientation, player-plays join) + 4.5 / 5.3 (per-play
        contextual models + residual aggregation).
    (The season-grain PFF state-space chain — 04f/04g calibration + step 6.5
    Kalman/EWMA — was removed 2026-07-21 after three failed A/Bs; recover
    via git history.)
    """
    # Read flags at build-time so the runner reflects the current config.
    try:
        from config import ENABLE_PLAY_LAYER  # noqa: WPS433
    except Exception:
        ENABLE_PLAY_LAYER = True
    play_layer_step_nums = {1.4, 1.5, 1.6, 4.5, 5.3}

    invocations: list[StepInvocation] = []
    base_steps = STEPS
    skip_nums: set[float] = set()
    if not ENABLE_PLAY_LAYER:
        skip_nums |= play_layer_step_nums
    if skip_nums:
        base_steps = [s for s in base_steps if s[0] not in skip_nums]

    # Filter by --step / --from
    if args.step is not None:
        base_steps = [s for s in base_steps if s[0] == args.step]
    elif args.from_step > 1:
        base_steps = [s for s in base_steps if s[0] >= args.from_step]

    test_args = ["--test"] if args.test else []

    for step_num, script, name in base_steps:
        extra = _step_extra_args(step_num, mode, retune_only,
                                  args.tune_trials, args.tune_seeds)
        # In retune mode the pre-calibration step-6 run is pure waste: step 7
        # sources its player universe from player_attributes (not the skill
        # snapshot), so nothing between 5.x and the calibration chain consumes
        # step-6 output. Skip it here; the post-7 splice runs it exactly
        # once with the fresh calibrations.
        if mode == "retune" and step_num == 6:
            logger.info("Retune mode: deferring step %s until after 04b/04d "
                        "calibration (single run)", step_num)
        else:
            invocations.append(StepInvocation(step_num, script, name, test_args + extra))

        # In retune mode, splice the calibration chain right after step 7 runs.
        # Order: 7 → 04b → 04d → 6 (single run, consumes new role-calibrated
        # JSON via the cache priority chain) → continue.
        if mode == "retune" and step_num == 7:
            invocations.append(StepInvocation(
                CALIBRATION_4B[0], CALIBRATION_4B[1], CALIBRATION_4B[2], [],
            ))
            invocations.append(StepInvocation(
                CALIBRATION_4D[0], CALIBRATION_4D[1], CALIBRATION_4D[2], [],
            ))
            invocations.append(StepInvocation(
                6, "pipeline/06_skill_estimation.py",
                "Skill Estimation (re-run w/ calibrated betas/Kalman)",
                test_args,
            ))

    return invocations


def run_step(script: str, extra_args: list[str]) -> int:
    cmd = [sys.executable, script] + extra_args
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_args()
    mode, retune_only = _resolve_mode(args)
    invocations = _build_invocations(args, mode, retune_only)

    if not invocations:
        logger.error("No matching steps found")
        sys.exit(1)

    logger.info("Mode: %s%s", mode,
                f"  (retune-only steps: {sorted(retune_only)})" if retune_only else "")

    for inv in invocations:
        logger.info("=" * 60)
        logger.info("STEP %s: %s", inv.step_num, inv.name)
        logger.info("=" * 60)
        rc = run_step(inv.script, inv.extra_args)
        if rc != 0:
            logger.error("Step %s failed (exit code %d). Stopping.", inv.step_num, rc)
            sys.exit(rc)
        logger.info("Step %s complete.\n", inv.step_num)

    if mode == "retune":
        logger.warning("=" * 60)
        logger.warning("RETUNE COMPLETE — to audit MONOTONE_ENABLED, run two")
        logger.warning("paired sweeps (unconstrained vs monotone-on) and then")
        logger.warning("    python experiments/00h_monotone_decision.py")
        logger.warning("00h is not auto-chained because it needs pre-generated sweeps.")
        logger.warning("-" * 60)
        logger.warning("Feature surface changed? Re-run the per-position-vs-global")
        logger.warning("gates before trusting GLOBAL_SCORE_OVERRIDES:")
        logger.warning("    python pipeline/08b_global_draft_model.py --blend-ab")
        logger.warning("then re-evaluate offline on calibrated scores (see memory /")
        logger.warning("blend_ab_fold_preds.parquet + 8c global calibrators).")
        logger.warning("=" * 60)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
