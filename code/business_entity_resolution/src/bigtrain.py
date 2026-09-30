"""
Larger-data training (v9 experiment).

Same 150k training + 30k held-out S1 as v8 (identical held-out entities), plus
N_EXTRA extra training S1 blocked against the full pool.  In the extra set all
positives and hard negatives are kept and easy negatives are sampled with
inverse-probability weights, so stage 1 sees ~3x more businesses within 16 GB.
Stage 2 is trained exactly as before on out-of-fold stage-1 probabilities of
the base 150k (the fold models also see the extra data).

    python -m src.bigtrain            # writes models to ../../work/artifacts_v9
"""
from __future__ import annotations

import gc
import json
import shutil
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from .blocking import BlockingConfig, apply_cap
from .config import ARTIFACT_DIR, RANDOM_SEED, TRAIN_DIR, WORK_DIR
from .data import get_transliterator, gt_pairs, load_normalized, read_ground_truth
from .features import set_noise_tokens, set_vocab
from .model import LGB_PARAMS, macro_f05, tune_policy
from .pipeline import add_name_freq, block_all, feature_cols, featurize
from .stage2 import EXTRA_COLS, STAGE2_COLS, stage2_frame

import sys
NT, NV, N_EXTRA = 150_000, 30_000, 300_000
TARGET_EXTRA_PAIRS = 5_000_000                       # per shard
N_SHARDS = int(sys.argv[1]) if len(sys.argv) > 1 else 1   # 1 = v9, 2 = v10 (600k extra)
N_FOLDS = int(sys.argv[2]) if len(sys.argv) > 2 else (3 if N_SHARDS == 1 else 2)
OUT = WORK_DIR / ("artifacts_v9" if N_SHARDS == 1 else f"artifacts_v{8 + N_SHARDS}")
BASE_FEAT = WORK_DIR / "train_feat_150000_30000_d19.parquet"
SHARD_CAND = lambda k: WORK_DIR / ("train_cand_extra300k.parquet" if k == 0 else f"train_cand_extra300k_s{k}.parquet")
SHARD_FEAT = lambda k: WORK_DIR / ("train_feat_extra300k.parquet" if k == 0 else f"train_feat_extra300k_s{k}.parquet")

T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.0f}s] {m}", flush=True)


def to_numpy_chunked(df: pl.DataFrame, cols, out: np.ndarray, start: int, step: int = 500_000) -> int:
    """Copy df[cols] into out[start:] chunk by chunk (no full-size temporary)."""
    for i in range(0, df.height, step):
        c = df.slice(i, step).select(cols).to_numpy().astype(np.float32)
        out[start + i:start + i + c.shape[0]] = c
    return start + df.height


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tr = get_transliterator(TRAIN_DIR, log=log)
    s1, tg = load_normalized(TRAIN_DIR, "train", tr, log=log)
    keep = s1["entity_id"].hash(seed=RANDOM_SEED) % 10_000 >= 1900          # test-like: drop 19% (as v8)
    s1 = s1.filter(keep)
    s1, tg = add_name_freq(s1, tg)
    ids = s1["entity_id"].sample(NT + NV, seed=RANDOM_SEED, shuffle=True)   # identical to v8
    train_ids, val_ids = ids[:NT], ids[NT:]
    rest = s1.select("entity_id").join(pl.DataFrame({"entity_id": ids}), on="entity_id", how="anti")["entity_id"]
    shard_ids = [rest.sample(N_EXTRA, seed=RANDOM_SEED + 1)]            # shard 0 = the v9 extra set
    for k in range(1, N_SHARDS):
        rest = rest.filter(~rest.is_in(pl.concat(shard_ids).implode()))
        shard_ids.append(rest.sample(N_EXTRA, seed=RANDOM_SEED + 1 + k))
    truth_all = gt_pairs(read_ground_truth(TRAIN_DIR))
    set_noise_tokens(json.loads((ARTIFACT_DIR / "noise_tokens.json").read_text()))
    set_vocab(s1["name_core"])

    # ---- extra S1: block, label, subsample easy negatives, featurize -------
    for k, extra_ids in enumerate(shard_ids):
      EXTRA_CAND, EXTRA_FEAT = SHARD_CAND(k), SHARD_FEAT(k)
      if not EXTRA_FEAT.exists():
          if EXTRA_CAND.exists():
              cand = pl.read_parquet(EXTRA_CAND)
          else:
              cand = block_all(s1, tg, BlockingConfig(), query_ids=extra_ids)
              cand.write_parquet(EXTRA_CAND)
          cand = apply_cap(cand.drop("s1_ncand", strict=False), BlockingConfig())
          cand = cand.with_columns(pl.len().over("s1_idx").cast(pl.Int32).alias("s1_ncand"))
          cand = cand.with_columns(pl.Series("s1_id", s1["entity_id"].to_numpy()[cand["s1_idx"].to_numpy()]),
                                   pl.Series("t_id", tg["entity_id"].to_numpy()[cand["t_idx"].to_numpy()]))
          cand = (cand.join(truth_all.with_columns(pl.lit(1, pl.Int8).alias("y")), on=["s1_id", "t_id"], how="left")
                  .with_columns(pl.col("y").fill_null(0)))
          r = lambda c: pl.col(c).fill_null(32000)
          hard = (r("rk_rev") <= 2) | (r("rk_comb") <= 3) | (r("rk_name") <= 2) | (r("rk_addr") <= 2)
          cand = cand.with_columns(hard.alias("hard"))
          n_pos = int(cand["y"].sum())
          n_hard = cand.filter((pl.col("y") == 0) & pl.col("hard")).height
          n_easy = cand.filter((pl.col("y") == 0) & ~pl.col("hard")).height
          q_hard = min(1.0, max(0.05, (TARGET_EXTRA_PAIRS * 0.8 - n_pos) / max(n_hard, 1)))
          q_easy = min(1.0, max(0.01, (TARGET_EXTRA_PAIRS - n_pos - q_hard * n_hard) / max(n_easy, 1)))
          log(f"[extra] {cand.height:,} pairs: {n_pos:,} pos, {n_hard:,} hard neg (keep {q_hard:.2f}), "
              f"{n_easy:,} easy neg (keep {q_easy:.3f})")
          rng = np.random.default_rng(RANDOM_SEED)
          u = rng.random(cand.height)
          q = np.where(cand["y"].to_numpy() == 1, 1.0, np.where(cand["hard"].to_numpy(), q_hard, q_easy))
          cand = cand.with_columns(pl.Series("w", (1.0 / q).astype(np.float32))).filter(pl.Series(u < q))
          log(f"[extra] kept {cand.height:,} pairs")
          lab = cand.select("s1_idx", "t_idx", "y", "w")
          feats = featurize(cand.drop("s1_id", "t_id", "y", "w", "hard"), s1, tg)
          feats = feats.join(lab, on=["s1_idx", "t_idx"])
          feats.write_parquet(EXTRA_FEAT)
          del feats, cand, lab
          gc.collect()

    # free the text frames; stage 2 later needs only target name/addr/id
    t_name, t_addr = tg["name_core"], tg["addr"]          # Arrow strings (compact); gathered by index later
    t_is3 = tg["entity_id"].str.starts_with("S3-").to_numpy()
    t_ent = tg["entity_id"]
    s1_ent = s1["entity_id"]
    del s1, tg
    gc.collect()

    # ---- assemble matrices ----------------------------------------------------
    base_keys = pl.read_parquet(BASE_FEAT, columns=["s1_idx", "t_idx"])
    base_keys = base_keys.with_columns(pl.Series("s1_id", s1_ent.gather(base_keys["s1_idx"])),
                                       pl.Series("t_id", t_ent.gather(base_keys["t_idx"])))
    truth = truth_all.join(pl.DataFrame({"s1_id": ids}), on="s1_id")
    base_keys = (base_keys.join(truth.with_columns(pl.lit(1, pl.Int8).alias("y")), on=["s1_id", "t_id"], how="left")
                 .with_columns(pl.col("y").fill_null(0)))
    is_val = base_keys["s1_id"].is_in(val_ids.implode()).to_numpy()
    cols = [c for c in pl.read_parquet_schema(BASE_FEAT) if c not in ("s1_idx", "t_idx")]
    extra_files = [SHARD_FEAT(k) for k in range(N_SHARDS)]
    extra_n = sum(pl.scan_parquet(f).select(pl.len()).collect().item() for f in extra_files)
    n_trn, n_val = int((~is_val).sum()), int(is_val.sum())
    log(f"[data] base train {n_trn:,} + extra {extra_n:,} pairs; val {n_val:,}; {len(cols)} features")

    X = np.empty((n_trn + extra_n, len(cols)), np.float32)
    Xv = np.empty((n_val, len(cols)), np.float32)
    base = pl.read_parquet(BASE_FEAT, columns=cols)
    to_numpy_chunked(base.filter(pl.Series(~is_val)), cols, X, 0)
    to_numpy_chunked(base.filter(pl.Series(is_val)), cols, Xv, 0)
    base_extra = base.select(["rk_rev"] + [c for c in EXTRA_COLS if c != "rk_rev"])   # for stage 2
    del base
    gc.collect()
    ys, ws, pos = [base_keys.filter(pl.Series(~is_val))["y"].to_numpy()], [np.ones(n_trn, np.float32)], n_trn
    for f in extra_files:
        ex = pl.read_parquet(f)
        pos = to_numpy_chunked(ex, cols, X, pos)
        ys.append(ex["y"].to_numpy()); ws.append(ex["w"].to_numpy().astype(np.float32))
        del ex
        gc.collect()
    y = np.concatenate(ys).astype(np.float32)
    w = np.concatenate(ws)
    ex = None
    yv = base_keys.filter(pl.Series(is_val))["y"].to_numpy()
    del ex
    gc.collect()
    log("[data] matrices ready")

    # ---- stage 1 ---------------------------------------------------------------
    full = lgb.Dataset(X, y, weight=w, feature_name=cols, free_raw_data=False,
                       params={"max_bin": LGB_PARAMS["max_bin"]}).construct()
    dval = lgb.Dataset(Xv, yv, reference=full)
    if (OUT / "lgb_matcher.txt").exists():                     # resume: reuse the saved stage-1 model
        b1 = lgb.Booster(model_file=str(OUT / "lgb_matcher.txt"))
        n_round = b1.current_iteration()
        log(f"[stage1] reusing saved model ({n_round} rounds)")
    else:
        b1 = lgb.train(LGB_PARAMS, full, 3000, valid_sets=[dval],
                       callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(250)])
        n_round = b1.best_iteration
        b1.save_model(str(OUT / "lgb_matcher.txt"), num_iteration=n_round)
    log(f"[stage1] best iteration {n_round}")
    vkeys = base_keys.filter(pl.Series(is_val))
    val = vkeys.with_columns(pl.Series("p", b1.predict(Xv, num_iteration=n_round).astype(np.float32)))
    vtruth = truth.join(pl.DataFrame({"s1_id": val_ids}), on="s1_id")
    pol1 = tune_policy(val.select("s1_id", "t_id", "p"), vtruth, val_ids, exclusive=True, log=log)
    f1 = macro_f05(pol1.apply(val.select("s1_id", "t_id", "p")), vtruth, val_ids)
    log(f"[val] stage-1 macro F0.5 = {f1:.5f}   (v8 stage-1: 0.98327)")

    # ---- out-of-fold stage 1 for the base train rows (fold models also see extra) --
    tkeys = base_keys.filter(pl.Series(~is_val))
    fold = (tkeys["s1_idx"].hash(seed=1) % N_FOLDS).to_numpy()
    oof = np.zeros(n_trn, np.float32)
    extra_idx = np.arange(n_trn, n_trn + extra_n)
    for k in range(N_FOLDS):
        m = np.flatnonzero(fold == k)
        ck = OUT / f"oof_fold{k}_of{N_FOLDS}.npy"               # resume: each finished fold is saved
        if ck.exists():
            oof[m] = np.load(ck)
            log(f"[stage2] out-of-fold fold {k + 1}/{N_FOLDS} loaded from checkpoint")
            continue
        keep_idx = np.concatenate([np.flatnonzero(fold != k), extra_idx])
        b = lgb.train(LGB_PARAMS, full.subset(keep_idx), num_boost_round=n_round)
        oof[m] = b.predict(X[m])
        np.save(ck, oof[m])
        log(f"[stage2] out-of-fold fold {k + 1}/{N_FOLDS} done")
        del b
        gc.collect()
    del full, X
    gc.collect()

    # ---- stage 2 ---------------------------------------------------------------
    be = base_extra
    trn = pl.concat([tkeys, be.filter(pl.Series(~is_val))], how="horizontal").with_columns(pl.Series("p", oof))
    vv = pl.concat([vkeys, be.filter(pl.Series(is_val))], how="horizontal").with_columns(val["p"])
    T = stage2_frame(trn, t_name, t_addr, t_is3).join(
        trn.select(pl.col("s1_idx").alias("g"), pl.col("t_idx").alias("t"), "y"), on=["g", "t"])
    V = stage2_frame(vv, t_name, t_addr, t_is3).join(
        vv.select(pl.col("s1_idx").alias("g"), pl.col("t_idx").alias("t"), "y", "s1_id", "t_id"), on=["g", "t"])
    p2 = dict(LGB_PARAMS, num_leaves=63, learning_rate=0.03, min_data_in_leaf=200)
    b2 = lgb.train(p2, lgb.Dataset(T.select(STAGE2_COLS).to_numpy(), T["y"].to_numpy(), feature_name=STAGE2_COLS),
                   3000, valid_sets=[lgb.Dataset(V.select(STAGE2_COLS).to_numpy(), V["y"].to_numpy())],
                   callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(500)])
    V = V.with_columns(pl.Series("p", b2.predict(V.select(STAGE2_COLS).to_numpy(),
                                                 num_iteration=b2.best_iteration).astype(np.float32)))
    pol2 = tune_policy(V.select("s1_id", "t_id", "p"), vtruth, val_ids, exclusive=True, log=log)
    f2 = macro_f05(pol2.apply(V.select("s1_id", "t_id", "p")), vtruth, val_ids)
    b2.save_model(str(OUT / "lgb_stage2.txt"), num_iteration=b2.best_iteration)
    pol2.save(OUT / "policy.json")
    for f in ("translit_map.json", "native_states.json", "noise_tokens.json"):
        shutil.copy(ARTIFACT_DIR / f, OUT / f)
    V.select("s1_id", "t_id", "p", "y").write_parquet(WORK_DIR / f"val_scored_{OUT.name[-3:].strip('_')}.parquet")
    log(f"[val] {OUT.name} stage-2 macro F0.5 = {f2:.5f}   (v9: 0.98527, v8: 0.98499, same held-out entities)")


if __name__ == "__main__":
    main()
