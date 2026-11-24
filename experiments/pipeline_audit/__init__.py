"""
cfDARKO pipeline audit harness.

Read-only audit of the full pipeline + data outputs at every step. Each
phase is a runnable module; run_audit.py orchestrates and emits report.md.
Never mutates data/ — fault injection is done in scratch dirs.
"""
