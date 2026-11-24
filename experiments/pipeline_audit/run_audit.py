"""
Top-level cfDARKO pipeline audit driver.

Aggregates Phase 1-5 audits into a single markdown report at
experiments/pipeline_audit/report.md, plus the per-phase CSVs each module
emits. Read-only — never mutates data/.

Usage:
  python experiments/pipeline_audit/run_audit.py                  # Phases 1-4
  python experiments/pipeline_audit/run_audit.py --phase 1        # one phase
  python experiments/pipeline_audit/run_audit.py --phase 1 2 3    # subset
  python experiments/pipeline_audit/run_audit.py --deep           # +Phase 5
  python experiments/pipeline_audit/run_audit.py --all            # all phases incl. 5
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.pipeline_audit import (   # noqa: E402
    audit_feature_flow,
    audit_lineage,
    audit_manifests,
    audit_models,
    audit_output,
)

logger = logging.getLogger(__name__)

OUT_DIR = Path(__file__).parent
REPORT_PATH = OUT_DIR / "report.md"


@dataclass
class PhaseResult:
    phase: int
    name: str
    runtime_sec: float
    section_md: str
    error_count: int
    warning_count: int


# Each phase: (number, label, run-fn returning (section_md, n_errors, n_warnings))

def _run_phase1() -> tuple[str, int, int]:
    manifests, provenance, missing = audit_manifests.collect()
    audit_manifests.write_csvs(manifests, provenance, missing)
    section = audit_manifests.build_section(manifests, provenance, missing)
    n_err = sum(1 for m in manifests if m.note not in {"ok", "pre-v2"})
    n_err += sum(1 for p in provenance if p.tuning_mode == "cached-stale")
    n_err += len(missing)
    n_warn = sum(1 for m in manifests if m.note == "pre-v2")
    return section, n_err, n_warn


def _run_phase2() -> tuple[str, int, int]:
    rows = audit_lineage.collect()
    audit_lineage.write_csv(rows)
    section = audit_lineage.build_section(rows)
    n_err = sum(1 for r in rows if (not r.passed) and r.severity == "ERROR")
    n_warn = sum(1 for r in rows if (not r.passed) and r.severity == "WARNING")
    return section, n_err, n_warn


def _run_phase3() -> tuple[str, int, int]:
    rows = audit_feature_flow.collect()
    audit_feature_flow.write_csv(rows)
    section = audit_feature_flow.build_section(rows)
    n_err = sum(1 for r in rows if r.severity == "ERROR")
    n_warn = sum(1 for r in rows if r.severity == "WARNING")
    return section, n_err, n_warn


def _run_phase4() -> tuple[str, int, int]:
    artifacts, issues = audit_output.collect()
    audit_output.write_csv(artifacts)
    section = audit_output.build_section(artifacts, issues)
    n_err = sum(1 for i in issues if i.severity == "ERROR")
    n_warn = sum(1 for i in issues if i.severity == "WARNING")
    return section, n_err, n_warn


def _run_phase5() -> tuple[str, int, int]:
    rows = audit_models.collect()
    audit_models.write_csv(rows)
    section = audit_models.build_section(rows)
    n_err = sum(1 for r in rows if r.severity == "ERROR")
    n_warn = sum(1 for r in rows if r.severity == "WARNING")
    return section, n_err, n_warn


PHASE_RUNNERS = {
    1: ("Manifests + Provenance", _run_phase1),
    2: ("Lineage",                 _run_phase2),
    3: ("Feature Flow",            _run_phase3),
    4: ("Output Sanity",           _run_phase4),
    5: ("Model Behavior",          _run_phase5),
}


def _run_pipeline_validate() -> tuple[str, int, int]:
    """Wrap pipeline/validate.py --all output into a top section. Captures
    the regression baseline: existing 10 step validators must keep passing."""
    cmd = [sys.executable, str(ROOT / "pipeline" / "validate.py"), "--all"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    body = proc.stderr or proc.stdout
    # Count surfaced FAIL lines for the report header.
    n_err = sum(1 for line in body.splitlines() if "FAIL" in line)
    n_warn = sum(1 for line in body.splitlines() if "WARN" in line)
    section = (
        "## Phase 0 — `pipeline/validate.py --all`\n\n"
        f"Exit code: `{proc.returncode}`\n\n"
        "```text\n" + body.strip() + "\n```\n"
    )
    return section, n_err, n_warn


def main() -> int:
    parser = argparse.ArgumentParser(description="Run cfDARKO pipeline audit")
    parser.add_argument("--phase", nargs="+", type=int,
                        choices=range(1, 6), metavar="N",
                        help="Run specific phase(s) (default: 1-4)")
    parser.add_argument("--deep", action="store_true",
                        help="Include Phase 5 (model behavior)")
    parser.add_argument("--all", action="store_true",
                        help="Run all phases (1-5)")
    parser.add_argument("--skip-validate", action="store_true",
                        help="Skip Phase 0 (pipeline/validate.py --all)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    if args.all:
        phases = [1, 2, 3, 4, 5]
    elif args.phase:
        phases = sorted(set(args.phase))
    else:
        phases = [1, 2, 3, 4] + ([5] if args.deep else [])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sections: list[str] = []
    summary_rows: list[PhaseResult] = []

    # Phase 0 — wrap pipeline/validate.py.
    if not args.skip_validate:
        logger.info("Running Phase 0: pipeline/validate.py --all")
        t0 = time.time()
        sec, ne, nw = _run_pipeline_validate()
        sections.append(sec)
        summary_rows.append(PhaseResult(0, "validate.py --all",
                                          time.time() - t0, sec, ne, nw))

    for ph in phases:
        label, runner = PHASE_RUNNERS[ph]
        logger.info("Running Phase %d: %s", ph, label)
        t0 = time.time()
        try:
            sec, ne, nw = runner()
        except Exception as exc:
            logger.exception("Phase %d crashed: %s", ph, exc)
            sec = f"## Phase {ph} — {label}\n\n**CRASHED:** {exc}"
            ne, nw = 1, 0
        sections.append(sec)
        summary_rows.append(PhaseResult(ph, label, time.time() - t0,
                                          sec, ne, nw))

    # Compose report.md
    header = [
        "# cfDARKO Pipeline Audit Report",
        "",
        f"_Generated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}_",
        "",
        "## Summary",
        "",
        "| phase | label | runtime (s) | errors | warnings |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in summary_rows:
        header.append(
            f"| {r.phase} | {r.name} | {r.runtime_sec:.1f} | "
            f"{r.error_count} | {r.warning_count} |"
        )
    header.append("")

    REPORT_PATH.write_text("\n".join(header + sections) + "\n")
    logger.info("Audit report → %s", REPORT_PATH)

    total_err = sum(r.error_count for r in summary_rows)
    return 1 if total_err > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
