"""
Manifest sidecars for cached tuning artifacts.

Each tuning step writes a sidecar JSON next to its primary cache file
(`tuned_params.json` → `tuned_params_manifest.json`). The sidecar records the
inputs that, if changed, would invalidate the cached params: feature columns,
config slices, hyperparameter grids, upstream-dependency shapes/hashes.

On a `--no-tune` run the consumer calls `check_manifest()` against the current
state; if the saved fingerprint mismatches, the caller surfaces a STALE
warning naming exactly which keys differ. cfdarko's `--no-tune` is *warn-only*
by design — we never silently auto-retune from a stale cache, since that
would defeat the speed of fast-iteration mode.

## Producer / consumer asymmetry

`_diff_fingerprints` only flags keys present in the *current* fingerprint but
missing or different in the *saved* one. Keys present in saved but missing
in current are ignored.

This means producers can write a *superset* of fingerprint keys while
consumers check only the *subset* they actually depend on. e.g. step 4b
writes `{stats, cal_season_cutoff, bounds, nfl_labels_file, ...}` but step 6
loading those betas only checks `{stats}` — step 6 doesn't care if 04b's
internal `param_bounds` changed.

Always design the consumer's fingerprint dict to be the minimal set of keys
that, if changed, would mean the cached artifact no longer reflects the
inputs the consumer cares about.

## Provenance vocabulary

`tag_summary(payload, mode_used=...)` stamps three values:
  - "fresh"        — step actually fit / tuned this run
  - "cached"       — step loaded a prior artifact and the manifest matched
  - "cached-stale" — step loaded a prior artifact and the manifest mismatched

Stamping summary JSONs with `_tuning_mode` prevents downstream readers from
silently comparing a cached `cv_mae` against a fresh-tune baseline.

## Manifest version

Manifests carry `_manifest_version` (currently 2). v1 manifests (written
before the fingerprint-dict refactor) are treated as "missing" — the next
tune writes a v2 manifest. One-shot transition; cache hits go silent
afterward.
"""

from __future__ import annotations

import enum
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

logger = logging.getLogger(__name__)

MANIFEST_VERSION = 2

ModeUsed = Literal["fresh", "cached", "cached-stale"]


# ── Primitive hashers ────────────────────────────────────────────────────────

def _hash_iter(values: Iterable[Any]) -> str:
    payload = json.dumps(sorted(str(v) for v in values), sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()


def hash_features(feature_cols: Iterable[str]) -> str:
    return _hash_iter(feature_cols)


def hash_config(config_slice: Any) -> str:
    payload = json.dumps(config_slice, sort_keys=True, default=_json_default)
    return hashlib.sha1(payload.encode()).hexdigest()


# ── CheckResult ──────────────────────────────────────────────────────────────

@dataclass(eq=False)
class CheckResult:
    """Result of a manifest check.

    `status`:
      - "ok"      — manifest present and all checked keys match
      - "stale"   — manifest present but at least one checked key differs
      - "missing" — no manifest, unreadable manifest, or pre-v2 manifest

    `diffs` is a sorted list of human-readable difference lines (empty
    unless status == "stale"). Pass into `log_stale_warning(diffs=...)`
    so the user sees what actually changed.

    Backward-compat: `result == "stale"` works (compares against status).
    """
    status: Literal["ok", "stale", "missing"]
    diffs: list[str] = field(default_factory=list)

    @property
    def is_ok(self) -> bool:        return self.status == "ok"
    @property
    def is_stale(self) -> bool:     return self.status == "stale"
    @property
    def is_missing(self) -> bool:   return self.status == "missing"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.status == other
        if isinstance(other, CheckResult):
            return self.status == other.status and self.diffs == other.diffs
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.status, tuple(self.diffs)))


# ── Fingerprint building blocks ──────────────────────────────────────────────

def schema_fingerprint(
    df,
    *,
    season_col: Optional[str] = None,
    extra_cols_to_summarize: Optional[list[str]] = None,
) -> dict:
    """Lightweight 'has the data shape changed' fingerprint for a DataFrame.

    Captures column list, row count, and (optionally) season range plus
    null-count summary for important columns. Does NOT hash the values —
    that would be slow and brittle. Tuning depends on the *shape* of the
    training data, not its byte-identical contents.
    """
    out: dict = {
        "n_rows":  int(len(df)),
        "columns": sorted(df.columns.tolist()),
    }
    if season_col and season_col in df.columns:
        non_null = df[season_col].dropna()
        if len(non_null):
            out["season_min"] = int(non_null.min())
            out["season_max"] = int(non_null.max())
    if extra_cols_to_summarize:
        out["null_counts"] = {
            c: int(df[c].isna().sum())
            for c in extra_cols_to_summarize if c in df.columns
        }
    return out


def json_content_fingerprint(path: Path | str) -> Optional[str]:
    """Hash the contents of a small JSON config file (e.g.
    `decay_betas_role.json`). Returns None if the file doesn't exist.
    Use for dependency artifacts that are themselves outputs of earlier
    tuning steps — when they change, downstream tuning is stale.
    """
    p = Path(path)
    if not p.exists():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def parquet_metadata_fingerprint(path: Path | str) -> Optional[dict]:
    """Lightweight 'has the parquet shape changed' fingerprint that reads only
    the file footer (n_rows + column names), not the data. Stable across
    no-op rebuilds where the file mtime changes but contents don't.
    Returns None if the file doesn't exist or pyarrow is unavailable.
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        import pyarrow.parquet as pq
    except ImportError:
        logger.warning("pyarrow not available — falling back to size+mtime")
        return file_size_mtime_fingerprint(p)
    md = pq.read_metadata(str(p))
    schema = pq.read_schema(str(p))
    return {
        "n_rows":  int(md.num_rows),
        "columns": sorted(field.name for field in schema),
    }


def file_size_mtime_fingerprint(path: Path | str) -> Optional[dict]:
    """Best-effort fingerprint for large files where hashing contents would
    be expensive AND the file isn't a parquet (use parquet_metadata for that).
    Note: mtime changes on every rewrite even if contents are identical —
    prefer parquet_metadata_fingerprint or json_content_fingerprint when
    you can.
    """
    p = Path(path)
    if not p.exists():
        return None
    stat = p.stat()
    return {
        "size_bytes": stat.st_size,
        "mtime_utc":  datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
                              .isoformat(timespec="seconds"),
    }


# ── Path convention ──────────────────────────────────────────────────────────

def manifest_path_for(params_path: Path | str) -> Path:
    p = Path(params_path)
    return p.with_name(p.stem + "_manifest.json")


# ── Write / Check ────────────────────────────────────────────────────────────

def _compose_legacy_fingerprint(
    feature_cols: Optional[Iterable[str]],
    config_slice: Optional[Any],
) -> dict:
    """Build a fingerprint dict from the legacy (feature_cols, config_slice)
    kwargs so old call sites stay working without migration."""
    fp: dict = {}
    if feature_cols is not None:
        cols = list(feature_cols)
        fp["feature_cols_hash"] = hash_features(cols)
        fp["n_features"] = len(cols)
    if config_slice is not None:
        fp["config_hash"] = hash_config(config_slice)
    return fp


def write_manifest(
    params_path: Path | str,
    *,
    fingerprint: Optional[dict] = None,
    mode_used: Optional[ModeUsed] = None,
    step_name: Optional[str] = None,
    extra: Optional[dict] = None,
    # legacy kwargs:
    feature_cols: Optional[Iterable[str]] = None,
    config_slice: Optional[Any] = None,
) -> Path:
    """Write a manifest sidecar next to params_path.

    Preferred: pass `fingerprint=<dict>` directly (composed by the caller
    using the building blocks above). Legacy `feature_cols`/`config_slice`
    kwargs are still honored — internally composed into a fingerprint dict.

    `mode_used` and `step_name` are recorded in the manifest's metadata
    block (keys prefixed `_`) so future debugging can tell what wrote it.

    `extra` is merged into the top level (after fingerprint), useful for
    descriptive fields like `n_models` that aren't part of staleness checks.
    """
    if fingerprint is None:
        fingerprint = _compose_legacy_fingerprint(feature_cols, config_slice)

    payload: dict[str, Any] = {
        "_manifest_version": MANIFEST_VERSION,
        "_written_at_utc":   datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if mode_used is not None:
        payload["_mode_used"] = mode_used
    if step_name is not None:
        payload["_step_name"] = step_name
    payload.update(fingerprint)
    if extra:
        payload.update(extra)

    out = manifest_path_for(params_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=_json_default))
    return out


def check_manifest(
    params_path: Path | str,
    *,
    fingerprint: Optional[dict] = None,
    feature_cols: Optional[Iterable[str]] = None,
    config_slice: Optional[Any] = None,
) -> CheckResult:
    """Compare the current fingerprint against the saved manifest.

    See module docstring for the producer/consumer asymmetry: only keys
    present in `current` (the fingerprint passed here) are checked.
    Producer-only keys are silently ignored.
    """
    if fingerprint is None:
        fingerprint = _compose_legacy_fingerprint(feature_cols, config_slice)

    mpath = manifest_path_for(params_path)
    if not mpath.exists():
        return CheckResult("missing")
    try:
        saved = json.loads(mpath.read_text())
    except Exception:
        return CheckResult("missing")

    saved_version = saved.get("_manifest_version", 1)
    if saved_version < MANIFEST_VERSION:
        logger.info(
            "[%s] manifest v%s (pre-fingerprint) — treating as missing; "
            "next tune will write v%d",
            mpath.name, saved_version, MANIFEST_VERSION,
        )
        return CheckResult("missing")

    diffs = _diff_fingerprints(saved, fingerprint)
    if diffs:
        return CheckResult("stale", diffs)
    return CheckResult("ok")


def log_stale_warning(
    params_path: Path | str,
    retune_command: str,
    *,
    diffs: Optional[list[str]] = None,
) -> None:
    """Emit a prominent banner when a cached params file is detected as stale.

    When `diffs` is provided (from `CheckResult.diffs`), enumerate the
    specific keys that differ so the user can judge whether to retune.
    """
    msg_lines = [
        "",
        "=" * 78,
        "STALE CACHE DETECTED",
        f"  Cached params:  {params_path}",
        f"  Refresh with:   {retune_command}",
    ]
    if diffs:
        msg_lines.append("  Differences:")
        for d in diffs:
            msg_lines.append(f"    • {d}")
    msg_lines.append("  Continuing with stale cache — results may be sub-optimal.")
    msg_lines.append("=" * 78)
    msg_lines.append("")
    for line in msg_lines:
        logger.warning(line)


# ── Provenance stamping ──────────────────────────────────────────────────────

def tag_summary(
    payload: dict,
    *,
    mode_used: ModeUsed,
    step_name: str,
) -> dict:
    """Stamp a provenance tag onto a summary dict.

    Always include this on JSON summaries that report CV scores or
    hyperparameters — it prevents silent comparison of cached metrics
    against a freshly-tuned baseline.

    Returns a new dict with the metadata keys (`_tuning_*`) prepended,
    so the original payload's keys come last and visually dominate.
    """
    return {
        "_tuning_mode":          mode_used,
        "_tuning_step":          step_name,
        "_tuning_stamped_at_utc": datetime.now(timezone.utc)
                                          .isoformat(timespec="seconds"),
        **payload,
    }


# ── Internal: fingerprint diffing ────────────────────────────────────────────

def _diff_fingerprints(saved: dict, current: dict, *, path: str = "") -> list[str]:
    """Sorted list of human-readable differences between two fingerprints.

    Asymmetric: only flags keys present in `current` but missing or
    different in `saved`. Producer-only keys (in `saved` but not `current`)
    are ignored — see module docstring for the rationale.

    Recurses one extra level when both sides at a key are dicts (so step 8's
    `by_key.{pos}/{model}.feature_cols_hash` shows up as a leaf diff).
    Caps at depth 2 to keep output readable.
    """
    diffs: list[str] = []
    saved_c = {k: v for k, v in saved.items() if not k.startswith("_")}
    for key in sorted(current.keys()):
        if key.startswith("_"):
            continue
        c_val = current[key]
        full_key = f"{path}.{key}" if path else key
        if key not in saved_c:
            diffs.append(f"{full_key}: <missing in cached> vs {_short(c_val)}")
            continue
        s_val = saved_c[key]
        if s_val == c_val:
            continue
        if (isinstance(s_val, dict) and isinstance(c_val, dict)
                and path.count(".") < 1):
            diffs.extend(_diff_fingerprints(s_val, c_val, path=full_key))
        else:
            diffs.append(f"{full_key}: {_short(s_val)} vs {_short(c_val)}")
    return diffs


def _short(value: Any, max_len: int = 80) -> str:
    s = json.dumps(value, default=_json_default)
    return s if len(s) <= max_len else s[:max_len - 3] + "..."


def _json_default(value: Any) -> Any:
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON-serializable: {type(value).__name__}")
