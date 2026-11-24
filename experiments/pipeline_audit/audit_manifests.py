"""
Phase 1: walk every v2 manifest sidecar + provenance file under
data/processed/ and report metadata, fingerprint keys, and tuning-mode
distribution.

Inspect-only — does NOT recompute consumer-side fingerprints. The pipeline
steps' fingerprint builders are defined inline in scripts (not importable
modules), so a faithful re-derive would duplicate logic and risk drifting.
The right way to refresh a stale manifest is `python run_pipeline.py
--retune-only <step>`. This audit just surfaces what's on disk.

Outputs:
  - manifests.csv  (one row per *_manifest.json)
  - provenance.csv (one row per *_provenance.json or summary with _tuning_mode)
  - section_text   (returned by `build_section()` for run_audit.py)
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config import PROCESSED_DIR  # noqa: E402

logger = logging.getLogger(__name__)

OUT_DIR = Path(__file__).parent
MANIFEST_CSV = OUT_DIR / "manifests.csv"
PROVENANCE_CSV = OUT_DIR / "provenance.csv"

EXPECTED_MANIFEST_VERSION = 2

# Files that the design specifies should carry a `_tuning_mode` provenance
# stamp. Each entry is (path, kind) where kind ∈ {"sidecar", "inline"}:
#  - "sidecar" → expect <stem>_provenance.json next to the primary file
#  - "inline"  → expect _tuning_mode key inside the primary file itself
# Pulled from src/utils/cache_manifest.py module docstring + the project's
# two-mode pipeline memory.
EXPECTED_PROVENANCE: list[tuple[str, str]] = [
    ("contextual_models/model_summary.json",         "sidecar"),
    ("draft_models/training_metrics.json",           "sidecar"),
    ("global_draft_models/training_metrics.json",    "sidecar"),
    ("kalman_hyperparams_draft.json",                "inline"),
]


@dataclass
class ManifestRow:
    path: str
    version: int
    mode_used: Optional[str]
    step_name: Optional[str]
    written_at: Optional[str]
    fingerprint_keys: str   # comma-joined sorted list
    note: str               # "ok" | "pre-v2" | "unparseable" | etc.


@dataclass
class ProvenanceRow:
    path: str
    tuning_mode: Optional[str]   # "fresh" | "cached" | "cached-stale" | None
    step: Optional[str]
    stamped_at: Optional[str]


@dataclass
class MissingProvenanceRow:
    path: str            # expected file or sidecar
    kind: str            # "sidecar" | "inline"
    note: str            # "primary file missing" | "no _tuning_mode key" | etc.


def _iter_manifests(root: Path) -> Iterator[Path]:
    """Yield every *_manifest.json beneath root."""
    yield from sorted(root.rglob("*_manifest.json"))


def _iter_provenance_files(root: Path) -> Iterator[Path]:
    """Yield every *_provenance.json + every JSON containing _tuning_mode at
    the top level. Excludes manifest sidecars (those carry _mode_used, not
    _tuning_mode)."""
    seen: set[Path] = set()
    for p in sorted(root.rglob("*_provenance.json")):
        seen.add(p)
        yield p
    # Walk other JSONs that might carry _tuning_mode (training_metrics.json,
    # tuned_params*.json, etc.) but skip manifest sidecars to avoid double-count.
    for p in sorted(root.rglob("*.json")):
        if p in seen or p.name.endswith("_manifest.json"):
            continue
        try:
            payload = json.loads(p.read_text())
        except Exception:
            continue
        if isinstance(payload, dict) and "_tuning_mode" in payload:
            yield p


def _safe_rel(path: Path) -> str:
    """Best-effort relative path for display. Falls back to the absolute
    path if `path` isn't under the project root (e.g., temp dirs in tests)."""
    try:
        return str(path.relative_to(PROCESSED_DIR.parent.parent))
    except ValueError:
        return str(path)


def _parse_manifest(path: Path) -> ManifestRow:
    rel = _safe_rel(path)
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        return ManifestRow(rel, 0, None, None, None, "", f"unparseable: {exc}")
    if not isinstance(payload, dict):
        return ManifestRow(rel, 0, None, None, None, "",
                           f"not a dict (got {type(payload).__name__})")

    version = int(payload.get("_manifest_version", 1))
    mode_used = payload.get("_mode_used")
    step_name = payload.get("_step_name")
    written = payload.get("_written_at_utc")

    fp_keys = sorted(
        k for k in payload.keys() if not k.startswith("_")
    )
    note = "ok" if version == EXPECTED_MANIFEST_VERSION else f"pre-v{EXPECTED_MANIFEST_VERSION}"
    return ManifestRow(rel, version, mode_used, step_name, written,
                       ",".join(fp_keys), note)


def _parse_provenance(path: Path) -> ProvenanceRow:
    rel = _safe_rel(path)
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return ProvenanceRow(rel, None, None, None)
    if not isinstance(payload, dict):
        return ProvenanceRow(rel, None, None, None)
    return ProvenanceRow(
        rel,
        payload.get("_tuning_mode"),
        payload.get("_tuning_step"),
        payload.get("_tuning_stamped_at_utc"),
    )


def _check_expected_provenance() -> list[MissingProvenanceRow]:
    """Return MissingProvenanceRow entries for any EXPECTED_PROVENANCE
    target that doesn't actually carry _tuning_mode."""
    missing: list[MissingProvenanceRow] = []
    for relpath, kind in EXPECTED_PROVENANCE:
        primary = PROCESSED_DIR / relpath
        if kind == "inline":
            if not primary.exists():
                missing.append(MissingProvenanceRow(
                    _safe_rel(primary),
                    kind, "primary file missing"))
                continue
            try:
                payload = json.loads(primary.read_text())
            except Exception as exc:
                missing.append(MissingProvenanceRow(
                    _safe_rel(primary),
                    kind, f"unparseable: {exc}"))
                continue
            if not (isinstance(payload, dict) and "_tuning_mode" in payload):
                missing.append(MissingProvenanceRow(
                    _safe_rel(primary),
                    kind, "no _tuning_mode key"))
        elif kind == "sidecar":
            sidecar = primary.with_name(primary.stem + "_provenance.json")
            if not sidecar.exists():
                missing.append(MissingProvenanceRow(
                    _safe_rel(sidecar),
                    kind, "sidecar missing"))
                continue
            try:
                payload = json.loads(sidecar.read_text())
            except Exception as exc:
                missing.append(MissingProvenanceRow(
                    _safe_rel(sidecar),
                    kind, f"unparseable: {exc}"))
                continue
            if not (isinstance(payload, dict) and "_tuning_mode" in payload):
                missing.append(MissingProvenanceRow(
                    _safe_rel(sidecar),
                    kind, "no _tuning_mode key"))
    return missing


# ── Public API ───────────────────────────────────────────────────────────────

def collect() -> tuple[list[ManifestRow], list[ProvenanceRow], list[MissingProvenanceRow]]:
    manifests = [_parse_manifest(p) for p in _iter_manifests(PROCESSED_DIR)]
    provenance = [_parse_provenance(p) for p in _iter_provenance_files(PROCESSED_DIR)]
    missing = _check_expected_provenance()
    return manifests, provenance, missing


def write_csvs(manifests: list[ManifestRow],
                provenance: list[ProvenanceRow],
                missing: list[MissingProvenanceRow]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with MANIFEST_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "version", "mode_used", "step_name",
                    "written_at_utc", "fingerprint_keys", "note"])
        for m in manifests:
            w.writerow([m.path, m.version, m.mode_used or "",
                        m.step_name or "", m.written_at or "",
                        m.fingerprint_keys, m.note])
    with PROVENANCE_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "tuning_mode", "step", "stamped_at_utc"])
        for r in provenance:
            w.writerow([r.path, r.tuning_mode or "",
                        r.step or "", r.stamped_at or ""])
    missing_csv = OUT_DIR / "provenance_missing.csv"
    with missing_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "kind", "note"])
        for r in missing:
            w.writerow([r.path, r.kind, r.note])


def build_section(manifests: list[ManifestRow],
                   provenance: list[ProvenanceRow],
                   missing: list[MissingProvenanceRow]) -> str:
    """Return a markdown section summarizing manifest + provenance state."""
    lines: list[str] = ["## Phase 1 — Manifests + Provenance", ""]

    # Punch list at top: anything not "ok", marked cached-stale, or expected
    # provenance that's missing.
    issues: list[str] = []
    for m in manifests:
        if m.note != "ok":
            issues.append(f"- `{m.path}` — {m.note}")
    for p in provenance:
        if p.tuning_mode == "cached-stale":
            issues.append(f"- `{p.path}` — `_tuning_mode=cached-stale`")
    for r in missing:
        issues.append(f"- `{r.path}` — expected {r.kind} provenance: {r.note}")
    if issues:
        lines += ["**Issues found:**", "", *issues, ""]
    else:
        lines += ["No issues — all manifests v2 and provenance complete.", ""]

    # Manifest table
    lines += ["### Manifests", "",
              "| path | version | mode | step | written_at | fingerprint_keys |",
              "| --- | --- | --- | --- | --- | --- |"]
    for m in manifests:
        lines.append(
            f"| `{m.path}` | {m.version} | "
            f"{m.mode_used or '—'} | {m.step_name or '—'} | "
            f"{m.written_at or '—'} | "
            f"`{m.fingerprint_keys[:80]}{'…' if len(m.fingerprint_keys) > 80 else ''}` |"
        )

    # Provenance table
    lines += ["", "### Provenance sidecars (`_tuning_mode`)", "",
              "| path | mode | step | stamped_at |",
              "| --- | --- | --- | --- |"]
    for r in provenance:
        lines.append(
            f"| `{r.path}` | {r.tuning_mode or '—'} | "
            f"{r.step or '—'} | {r.stamped_at or '—'} |"
        )

    # Summary stats
    mode_counts: dict[str, int] = {}
    for r in provenance:
        key = r.tuning_mode or "—"
        mode_counts[key] = mode_counts.get(key, 0) + 1
    lines += ["", "### Mode distribution",
              "",
              "| mode | count |",
              "| --- | --- |",
              *[f"| `{k}` | {v} |" for k, v in sorted(mode_counts.items())]]

    # Expected-but-missing provenance
    if missing:
        lines += ["", "### Expected provenance — missing", "",
                  "| path | kind | note |",
                  "| --- | --- | --- |",
                  *[f"| `{r.path}` | {r.kind} | {r.note} |" for r in missing]]
    return "\n".join(lines)


def has_errors(manifests: list[ManifestRow],
                provenance: list[ProvenanceRow],
                missing: list[MissingProvenanceRow]) -> bool:
    if any(m.note not in {"ok", "pre-v2"} for m in manifests):
        return True
    if any(p.tuning_mode == "cached-stale" for p in provenance):
        return True
    if missing:
        return True
    return False


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    manifests, provenance, missing = collect()
    write_csvs(manifests, provenance, missing)
    print(build_section(manifests, provenance, missing))
    return 1 if has_errors(manifests, provenance, missing) else 0


if __name__ == "__main__":
    sys.exit(main())
