"""
Part-file parquet checkpointing for long-running scrapers.

The old scheme appended every checkpoint chunk by re-reading the ENTIRE
growing parquet and rewriting it — O(N^2) I/O over a long scrape. The new
scheme writes each chunk as its own small part file under

    <name>.parquet.parts/part_<seq>.parquet

and, at successful completion, concatenates all parts (plus any pre-existing
main parquet) into the main file once, then removes the parts directory.

Resume logic must build its done-set from `read_with_parts` (main + parts) so
that an interrupted run — main parquet plus leftover part files — resumes
without re-fetching or duplicating already-saved chunks.
"""

import logging
import shutil
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def parts_dir(path: Path) -> Path:
    """Directory holding checkpoint part files for a main parquet path."""
    path = Path(path)
    return path.parent / (path.name + ".parts")


def write_part(path: Path, df: pd.DataFrame) -> Optional[Path]:
    """
    Write df as the next sequential part file for `path`.

    Returns the part file written, or None when df is empty (nothing written).
    """
    if df is None or df.empty:
        return None
    d = parts_dir(path)
    d.mkdir(parents=True, exist_ok=True)
    existing = sorted(d.glob("part_*.parquet"))
    seq = int(existing[-1].stem.split("_")[1]) + 1 if existing else 0
    out = d / f"part_{seq:05d}.parquet"
    df.to_parquet(out, index=False)
    return out


def read_with_parts(path: Path, columns: Optional[list[str]] = None) -> pd.DataFrame:
    """
    Read the main parquet (if present) plus all part files, concatenated.

    Returns an empty DataFrame when neither exists.
    """
    path = Path(path)
    frames: list[pd.DataFrame] = []
    if path.exists():
        frames.append(pd.read_parquet(path, columns=columns))
    d = parts_dir(path)
    if d.exists():
        for p in sorted(d.glob("part_*.parquet")):
            frames.append(pd.read_parquet(p, columns=columns))
    if not frames:
        return pd.DataFrame()
    if len(frames) == 1:
        return frames[0]
    return pd.concat(frames, ignore_index=True)


def consolidate(path: Path) -> None:
    """
    Fold all part files (plus any pre-existing main parquet) into the main
    parquet, then remove the parts directory. No-op when no parts exist.
    """
    path = Path(path)
    d = parts_dir(path)
    if not d.exists():
        return
    parts = sorted(d.glob("part_*.parquet"))
    if parts:
        frames = [pd.read_parquet(path)] if path.exists() else []
        frames += [pd.read_parquet(p) for p in parts]
        combined = pd.concat(frames, ignore_index=True)
        tmp = path.parent / (path.name + ".tmp")
        combined.to_parquet(tmp, index=False)
        tmp.replace(path)
        logger.info("Consolidated %d part files (%d rows) into %s",
                    len(parts), len(combined), path)
    shutil.rmtree(d)
