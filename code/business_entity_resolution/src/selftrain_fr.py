"""
v11-FR: self-training for the unseen country (France).

France has no training labels.  The v10h stage-2 scores on French test pairs
provide confident pseudo-labels (p >= POS -> match, p <= NEG -> non-match; the
uncertain middle is left out).  The v10 stage-1 model is *continued* with extra
boosting rounds on those French rows mixed with US/India training rows, so it
keeps what it learned and adds what is French-specific.  Only French pairs are
re-scored; US/India keep their v10h scores.

    python -m src.selftrain_fr                       v11-FR (pseudo-labels from the v10h scores)
    python -m src.selftrain_fr v12fr frnz_merge      v12-FR (pseudo-labels from the noise-word scores,
                                                     French noise words switched on)
    python -m src.selftrain_fr v13fr v12fr fils      v13-FR (second round: pseudo-labels from v12-FR,
                                                     'fils' added to the French noise words)
Resumable: every step caches its output.
"""
from __future__ import annotations

import gc
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from . import pipeline as P
from .config import RANDOM_SEED, TEST_DIR, TRAIN_DIR, WORK_DIR
from .data import get_transliterator, gt_pairs, load_normalized, read_ground_truth
from .features import COUNTRY_NOISE, set_noise_tokens, set_vocab
from .model import LGB_PARAMS, macro_f05

W = WORK_DIR
POS, NEG = 0.95, 0.02
N_NEG = 3_000_000          # sampled confident negatives
N_BASE = 3_000_000         # US/India training rows mixed in (keeps the model anchored)
EXTRA_ROUNDS = 400
TAG = sys.argv[1] if len(sys.argv) > 1 else "v11fr"
PSEUDO_SRC = W / f"test_final_scores_{sys.argv[2] if len(sys.argv) > 2 else 'frq6'}.parquet"
# generator noise words of French true copies (see work/frnz.py); used by v12-FR onwards
FR_NOISE = ({"france", "groupe", "developpement", "associes"} if TAG != "v11fr" else set()) |     (set(sys.argv[3].split(",")) if len(sys.argv) > 3 else set())
OUT_MODEL = W / f"artifacts_{TAG}_stage1.txt"
PSEUDO_FEAT = W / ("fr_pseudo_feat.parquet" if TAG == "v11fr" else f"fr_pseudo_feat_{TAG}.parquet")
SCORES_DIR = W / ("test_scores_fr_v11" if TAG == "v11fr" else f"test_scores_fr_{TAG}")
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def main():
    tr = get_transliterator(TRAIN_DIR, log=log)
    s1, tg = load_normalized(TEST_DIR, "test", tr, log=log)
    s1, tg = P.add_name_freq(s1, tg)
    set_noise_tokens(json.loads(P.NOISE_PATH.read_text()))
    set_vocab(s1["name_core"])
    if FR_NOISE:
        COUNTRY_NOISE["France"] = set(FR_NOISE)
    fr = pl.Series(np.flatnonzero((s1["country"] == "France").to_numpy()).astype(np.int32))
    s1 = s1.select(P.SIDE_COLS + ["country", "entity_id", "s1_name_freq", "s1_raw_freq"])
    tg = tg.select(P.SIDE_COLS + ["entity_id", "t_name_freq", "t_raw_freq"])
    base_model = lgb.Booster(model_file=str(P.MODEL_PATH))            # v10 stage 1
    cols = base_model.feature_name()

    # ---- 1. pseudo-labelled French pairs ----------------------------------------
    if not PSEUDO_FEAT.exists():
        sc = pl.read_parquet(PSEUDO_SRC).filter(pl.col("s1_idx").is_in(fr.implode()))
        # positives: confident AND the record's best S1 (exclusivity); negatives: any confident
        # non-match, including records owned by another S1 (the informative look-alikes)
        best = pl.col("p") == pl.col("p").max().over("t_idx")
        pos = sc.filter(best & (pl.col("p") >= POS)).select("s1_idx", "t_idx").with_columns(pl.lit(1, pl.Int8).alias("y"))
        neg_all = sc.filter(pl.col("p") <= NEG).select("s1_idx", "t_idx")
        neg = neg_all.sample(min(N_NEG, neg_all.height), seed=RANDOM_SEED)
        w_neg = neg_all.height / max(neg.height, 1)          # restore the real French base rate
        neg = neg.with_columns(pl.lit(0, pl.Int8).alias("y"))
        lab = pl.concat([pos.with_columns(pl.lit(1.0, pl.Float32).alias("w")),
                         neg.with_columns(pl.lit(w_neg, pl.Float32).alias("w"))])
        log(f"[pseudo] France: {sc.height:,} pairs; {pos.height:,} confident matches, "
            f"{neg.height:,} of {neg_all.height:,} confident non-matches sampled (weight {w_neg:.2f})")
        cand = pl.read_parquet(W / "test_cand.parquet").join(lab.select("s1_idx", "t_idx"), on=["s1_idx", "t_idx"])
        feats = P.featurize(cand, s1, tg).join(lab, on=["s1_idx", "t_idx"])
        feats.write_parquet(PSEUDO_FEAT)
        del sc, cand, feats
        gc.collect()

    # ---- 2. continue the v10 stage-1 model -------------------------------------
    if not OUT_MODEL.exists():
        ps = pl.read_parquet(PSEUDO_FEAT)
        base = pl.read_parquet(W / "train_feat_150000_30000_d19.parquet")
        s1tr = pl.read_parquet(W / "train_s1_norm.parquet", columns=["entity_id"])
        s1tr = s1tr.filter(s1tr["entity_id"].hash(seed=RANDOM_SEED) % 10_000 >= 1900)
        ids = s1tr["entity_id"].sample(180_000, seed=RANDOM_SEED, shuffle=True)
        val_ids = ids[150_000:]
        tgt = pl.concat([pl.read_parquet(W / f"train_{p}_norm.parquet", columns=["entity_id"]) for p in ("s2", "s3")])["entity_id"]
        base = base.with_columns(pl.Series("s1_id", s1tr["entity_id"].gather(base["s1_idx"])),
                                 pl.Series("t_id", tgt.gather(base["t_idx"])))
        truth = gt_pairs(read_ground_truth(TRAIN_DIR)).join(pl.DataFrame({"s1_id": ids}), on="s1_id")
        base = (base.join(truth.with_columns(pl.lit(1, pl.Int8).alias("y")), on=["s1_id", "t_id"], how="left")
                .with_columns(pl.col("y").fill_null(0)))
        isv = base["s1_id"].is_in(val_ids.implode())
        trn = base.filter(~isv).sample(min(N_BASE, int((~isv).sum())), seed=RANDOM_SEED)
        val = base.filter(isv)
        X = np.vstack([trn.select(cols).to_numpy().astype(np.float32), ps.select(cols).to_numpy().astype(np.float32)])
        y = np.concatenate([trn["y"].to_numpy(), ps["y"].to_numpy()]).astype(np.float32)
        w = np.concatenate([np.ones(trn.height, np.float32), ps["w"].to_numpy().astype(np.float32)])
        log(f"[train] continue v10 stage 1 on {trn.height:,} US/IN + {ps.height:,} French pseudo rows")
        p = dict(LGB_PARAMS, learning_rate=0.03)
        b = lgb.train(p, lgb.Dataset(X, y, weight=w, feature_name=cols), num_boost_round=EXTRA_ROUNDS,
                      init_model=base_model, keep_training_booster=True)
        b.save_model(str(OUT_MODEL))
        # guard: US/India held-out at stage 1 must not degrade
        vt = truth.join(pl.DataFrame({"s1_id": val_ids}), on="s1_id")
        Xv = val.select(cols).to_numpy().astype(np.float32)
        for name, m in (("v10 stage 1", base_model), ("v11-FR stage 1", b)):
            d = val.select("s1_id", "t_id").with_columns(pl.Series("p", m.predict(Xv).astype(np.float32)))
            f = macro_f05(P.Policy(0.725, 0.725, True).apply(d), vt, val_ids)
            log(f"[guard] US/India held-out stage-1 macro F0.5 ({name}, tau .725): {f:.5f}")
        del X, y, trn, val, base, ps
        gc.collect()

    # ---- 3. re-score French pairs, stage 2, combine, leaderboard rule ----------
    booster = lgb.Booster(model_file=str(OUT_MODEL))
    P.MODEL_PATH = OUT_MODEL                         # cache key for score_candidates
    P.score_candidates(W / "test_cand.parquet", s1, tg, booster, cols, SCORES_DIR, pl.col("s1_idx").is_in(fr.implode()))
    fr_scores = P.apply_stage2(pl.scan_parquet(SCORES_DIR / "*.parquet"), tg, s1.height)
    rest = pl.read_parquet(W / "test_final_scores_v10h.parquet").filter(~pl.col("s1_idx").is_in(fr.implode()))
    sc = pl.concat([rest, fr_scores.select(rest.columns)])
    sc.write_parquet(W / f"test_final_scores_{TAG}.parquet")
    sc = sc.filter(pl.col("p") == pl.col("p").max().over("t_idx"))
    sc = sc.with_columns(pl.col("p").rank("ordinal", descending=True).over("s1_idx").alias("rk"))
    sel = sc.filter((pl.col("p") >= 0.86) | ((pl.col("rk") == 1) & (pl.col("p") >= 0.55))).select("s1_idx", "t_idx")
    P.write_id_lists(sel, s1["entity_id"], tg["entity_id"], W / f"var_{TAG.upper()}_E86_T55.tsv", "matched_entity_ids")
    log(f"[{TAG}] written: {sel.height:,} links")


if __name__ == "__main__":
    main()
