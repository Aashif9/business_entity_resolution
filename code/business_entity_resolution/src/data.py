"""Loading raw TSVs and caching normalized versions as parquet."""
from __future__ import annotations

import time
from pathlib import Path

import polars as pl

from .config import ARTIFACT_DIR, WORK_DIR
import json

from . import normalize as N
from .normalize import Transliterator, learn_native_states, learn_translit, normalize_frame

TRANSLIT_PATH = ARTIFACT_DIR / "translit_map.json"
STATES_PATH = ARTIFACT_DIR / "native_states.json"


def read_tsv(path: Path) -> pl.DataFrame:
    """Read a challenge TSV (no quoting, everything as string, empty -> '')."""
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False).fill_null("")


def source_paths(split_dir: Path, split: str):
    return [split_dir / f"{split}_source{i}.tsv" for i in (1, 2, 3)]


def read_ground_truth(train_dir: Path) -> pl.DataFrame:
    return read_tsv(train_dir / "train_ground_truth.tsv")


def gt_pairs(gt: pl.DataFrame) -> pl.DataFrame:
    """Explode ground truth into (s1_id, t_id) rows."""
    return (gt.select(pl.col("source1_entity_id").alias("s1_id"),
                      pl.col("matched_entity_ids").str.split(",").alias("t_id"))
            .explode("t_id").filter(pl.col("t_id").is_not_null() & (pl.col("t_id") != "")))


def get_transliterator(train_dir: Path, log=print) -> Transliterator:
    if TRANSLIT_PATH.exists():
        return Transliterator.load(TRANSLIT_PATH)
    log("[translit] learning native->latin token map from train ground truth")
    s1 = read_tsv(train_dir / "train_source1.tsv").filter(pl.col("country") == "India")
    tg = pl.concat([read_tsv(train_dir / "train_source2.tsv"), read_tsv(train_dir / "train_source3.tsv")])
    tg = tg.filter(pl.col("country") == "India")
    tr = learn_translit(s1, tg, read_ground_truth(train_dir))
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    tr.save(TRANSLIT_PATH)
    log(f"[translit] {len(tr.mapping)} tokens learned")
    return tr


def get_native_states(train_dir: Path, log=print) -> dict:
    """Native-script state names -> English, learned from true pairs' last address components."""
    if STATES_PATH.exists():
        m = json.loads(STATES_PATH.read_text(encoding="utf-8"))
    else:
        last = lambda c: pl.col(c).str.split(",").list.last().str.strip_chars()
        s1 = read_tsv(train_dir / "train_source1.tsv").filter(pl.col("country") == "India")
        tg = pl.concat([read_tsv(train_dir / "train_source2.tsv"), read_tsv(train_dir / "train_source3.tsv")])
        tg = tg.filter(pl.col("country") == "India").select("entity_id", last("business_address").alias("tl"))
        pairs = (gt_pairs(read_ground_truth(train_dir)).join(tg.rename({"entity_id": "t_id"}), on="t_id")
                 .join(s1.select(pl.col("entity_id").alias("s1_id"), last("business_address").alias("sl")), on="s1_id"))
        m = learn_native_states(pairs.select("tl", "sl").iter_rows())
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        STATES_PATH.write_text(json.dumps(m, ensure_ascii=False, indent=0), encoding="utf-8")
        log(f"[states] learned {len(m)} native-script state names")
    N.NATIVE_STATES.clear()
    N.NATIVE_STATES.update(m)
    return m


def load_normalized(split_dir: Path, split: str, tr: Transliterator, log=print, tag: str = ""):
    """Return (s1, targets) normalized frames, cached as parquet in WORK_DIR."""
    get_native_states(split_dir.parent / "train", log=log)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for i, path in enumerate(source_paths(split_dir, split), start=1):
        cache = WORK_DIR / f"{split}{tag}_s{i}_norm.parquet"
        if cache.exists() and cache.stat().st_mtime > path.stat().st_mtime:
            out.append(pl.read_parquet(cache))
            continue
        t = time.time()
        df = normalize_frame(read_tsv(path), tr)
        df.write_parquet(cache)
        log(f"[normalize] {path.name}: {df.height} rows in {time.time() - t:.0f}s")
        out.append(df)
    s1, s2, s3 = out
    return s1, pl.concat([s2, s3])
