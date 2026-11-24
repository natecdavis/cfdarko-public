"""
Unit tests for the Phase 1 manifest audit. Uses temp-dir manifest sidecars
to exercise the full v2 / v1 / unparseable / cached-stale matrix without
touching the real data/processed/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.pipeline_audit import audit_manifests as AM


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def test_parse_v2_manifest_ok(tmp_path):
    p = tmp_path / "foo_manifest.json"
    _write(p, {
        "_manifest_version": 2,
        "_mode_used": "fresh",
        "_step_name": "04_contextual_models",
        "_written_at_utc": "2026-05-04T12:00:00+00:00",
        "stats": ["pass_yards", "rec_yards"],
    })
    row = AM._parse_manifest(p)
    assert row.version == 2
    assert row.mode_used == "fresh"
    assert row.step_name == "04_contextual_models"
    assert row.note == "ok"
    assert "stats" in row.fingerprint_keys


def test_parse_v1_manifest_marked_pre_v2(tmp_path):
    p = tmp_path / "old_manifest.json"
    _write(p, {"config_hash": "abcd", "written_at": "2026-01-01"})
    row = AM._parse_manifest(p)
    assert row.version == 1
    assert row.note == "pre-v2"


def test_parse_unparseable_manifest(tmp_path):
    p = tmp_path / "bad_manifest.json"
    p.write_text("{not valid json")
    row = AM._parse_manifest(p)
    assert row.note.startswith("unparseable:")


def test_parse_provenance_cached_stale(tmp_path):
    p = tmp_path / "x_provenance.json"
    _write(p, {
        "_tuning_mode": "cached-stale",
        "_tuning_step": "08_draft_models",
        "_tuning_stamped_at_utc": "2026-05-04T12:00:00+00:00",
    })
    row = AM._parse_provenance(p)
    assert row.tuning_mode == "cached-stale"
    assert row.step == "08_draft_models"


def test_has_errors_flags_pre_v2_only_as_warning(tmp_path):
    """has_errors() should treat pre-v2 as a warning, not an error."""
    only_pre_v2 = [AM.ManifestRow("x", 1, None, None, None, "", "pre-v2")]
    assert AM.has_errors(only_pre_v2, [], []) is False


def test_has_errors_flags_unparseable(tmp_path):
    bad = [AM.ManifestRow("x", 0, None, None, None, "", "unparseable: x")]
    assert AM.has_errors(bad, [], []) is True


def test_has_errors_flags_cached_stale(tmp_path):
    prov = [AM.ProvenanceRow("x", "cached-stale", "step8", "")]
    assert AM.has_errors([], prov, []) is True


def test_has_errors_flags_missing_provenance(tmp_path):
    miss = [AM.MissingProvenanceRow("x", "sidecar", "sidecar missing")]
    assert AM.has_errors([], [], miss) is True


def test_iter_provenance_finds_inline_tuning_mode(tmp_path):
    """A non-suffix JSON carrying _tuning_mode at top level should be found."""
    p = tmp_path / "training_metrics.json"
    _write(p, {"_tuning_mode": "fresh", "QB_role": {"cv_mae": 0.05}})
    found = list(AM._iter_provenance_files(tmp_path))
    assert p in found


def test_iter_provenance_skips_manifest_sidecars(tmp_path):
    """*_manifest.json files must not double-report as provenance."""
    p = tmp_path / "thing_manifest.json"
    _write(p, {
        "_manifest_version": 2,
        "_mode_used": "fresh",
        "_step_name": "x",
    })
    found = list(AM._iter_provenance_files(tmp_path))
    assert p not in found
