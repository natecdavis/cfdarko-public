"""T2 — Manifest fingerprint round-trip invariants.

Locks down the staleness detection that gates --no-tune cache reuse:

  * write_manifest → check_manifest with identical fingerprint → "ok"
  * write_manifest → check_manifest with mutated fingerprint   → "stale" + diffs
  * check_manifest with no sidecar present                     → "missing"
  * write a v1 (pre-fingerprint) manifest manually             → "missing"

Closes the gap exposed by C2 (where missing manifests previously logged at
INFO instead of WARNING). Producer/consumer asymmetry is also exercised:
keys saved but not in the current fingerprint should be silently ignored.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.cache_manifest import (
    write_manifest, check_manifest, manifest_path_for, MANIFEST_VERSION,
    hash_features, hash_config,
)


def test_manifest_roundtrip_ok(tmp_path):
    params = tmp_path / "tuned_params.json"
    params.write_text("{}")
    fp = {"features_hash": hash_features(["a", "b"]),
          "config_hash":   hash_config({"alpha_grid": [0.1, 1.0]})}
    write_manifest(params, fingerprint=fp, mode_used="fresh", step_name="t2")
    result = check_manifest(params, fingerprint=fp)
    assert result.is_ok
    assert result.status == "ok"
    assert not result.diffs


def test_manifest_stale_when_input_mutates(tmp_path, caplog):
    params = tmp_path / "tuned_params.json"
    params.write_text("{}")
    fp_old = {"features_hash": hash_features(["a", "b"])}
    write_manifest(params, fingerprint=fp_old, mode_used="fresh", step_name="t2")

    fp_new = {"features_hash": hash_features(["a", "b", "c"])}    # added feature
    result = check_manifest(params, fingerprint=fp_new)
    assert result.is_stale
    assert result.diffs, "stale result must surface diffs naming the changed key"
    assert any("features_hash" in d for d in result.diffs), \
        f"expected features_hash diff, got {result.diffs}"


def test_manifest_missing_returns_missing(tmp_path):
    params = tmp_path / "no_sidecar.json"
    params.write_text("{}")
    result = check_manifest(params, fingerprint={"x": "y"})
    assert result.is_missing
    assert not manifest_path_for(params).exists()


def test_manifest_v1_treated_as_missing(tmp_path):
    """Pre-fingerprint v1 manifests must trigger a fresh tune, not silently
    accept a cache that was written under different semantics."""
    params = tmp_path / "legacy.json"
    params.write_text("{}")
    legacy = {"_manifest_version": 1, "feature_cols_hash": "deadbeef"}
    manifest_path_for(params).write_text(json.dumps(legacy))
    result = check_manifest(params, fingerprint={"feature_cols_hash": "deadbeef"})
    assert result.is_missing


def test_manifest_consumer_subset_ignores_extra_saved_keys(tmp_path):
    """Producer wrote {features_hash, config_hash, bounds_hash}; consumer only
    cares about features_hash. Match on features_hash alone must be ok."""
    params = tmp_path / "params.json"
    params.write_text("{}")
    saved = {"features_hash": "X", "config_hash": "Y", "bounds_hash": "Z"}
    write_manifest(params, fingerprint=saved, mode_used="fresh", step_name="t2")
    consumer = {"features_hash": "X"}
    assert check_manifest(params, fingerprint=consumer).is_ok


def test_manifest_records_mode_and_step(tmp_path):
    params = tmp_path / "p.json"
    params.write_text("{}")
    write_manifest(params, fingerprint={"k": "v"},
                   mode_used="fresh", step_name="step-x")
    saved = json.loads(manifest_path_for(params).read_text())
    assert saved["_manifest_version"] == MANIFEST_VERSION
    assert saved["_mode_used"] == "fresh"
    assert saved["_step_name"] == "step-x"
