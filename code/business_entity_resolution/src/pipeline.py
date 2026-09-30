"""
End-to-end Business Entity Resolution pipeline.

    python -m src.pipeline --mode train     # sample train S1s, block, featurize, fit LightGBM, tune policy
    python -m src.pipeline --mode predict   # block all test S1s, score, write the two output TSVs
    python -m src.pipeline --mode all       # both

Run from code/business_entity_resolution/.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from .blocking import BlockingConfig, apply_cap, block_country
from .config import ARTIFACT_DIR, OUTPUT_DIR, RANDOM_SEED, TEST_DIR, TRAIN_DIR, WORK_DIR
from .data import gt_pairs, get_transliterator, load_normalized, read_ground_truth
import json

from .features import (COUNTRY_NOISE, NOISE_TOKENS, SIDE_COLS, learn_noise_tokens, pair_features,
                       set_noise_tokens, set_vocab)
from .model import LGB_PARAMS, LeaderboardRule, Policy, macro_f05, train_lgb, tune_policy
from .stage2 import EXTRA_COLS, STAGE2_COLS, stage2_frame

MODEL_PATH = ARTIFACT_DIR / "lgb_matcher.txt"
NOISE_PATH = ARTIFACT_DIR / "noise_tokens.json"
STAGE2_PATH = ARTIFACT_DIR / "lgb_stage2.txt"
POLICY_PATH = ARTIFACT_DIR / "policy.json"
RULE_PATH = ARTIFACT_DIR / "rule.json"          # final decision rule (LeaderboardRule)
# stage 1 continued on confident pseudo-labels of the unseen country (src/selftrain_fr.py);
# when present it re-scores that country's pairs with the noise words switched on
UNSEEN_MODEL_PATH = ARTIFACT_DIR / "lgb_matcher_unseen.txt"
# Generator noise words of true copies in countries unseen in training.  Found without labels
# on the test candidates: at the SAME house number and street these words are extra words
# 15-30x more often than the look-alike suffixes (Holding, Participations, ...), which almost
# always come with a changed house number.  Used only where the house numbers agree or the
# record has no address (see rescore_unseen_noise).
UNSEEN_NOISE = {"France": ["france", "groupe", "developpement", "associes", "fils"]}
FEAT_CHUNK = 1_000_000
# stage-1 columns kept next to p for the stage-2 group model
KEEP_FOR_STAGE2 = ["rk_rev"] + EXTRA_COLS

T0 = time.time()


def log(msg: str):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------

def add_name_freq(s1: pl.DataFrame, tg: pl.DataFrame):
    """How many S1 / targets in the same country share the core name (ambiguity)."""
    s1 = s1.with_columns(pl.len().over("country", "name_core").cast(pl.Int32).alias("s1_name_freq"))
    tg = tg.with_columns(pl.len().over("country", "name_core").cast(pl.Int32).alias("t_name_freq"))
    # same count on the name *as written* (legal form kept): "resolve india pvt ltd"
    # can be unique even when the core name "resolve india" is shared
    raw = (pl.col("business_name").str.to_lowercase().str.replace_all(r"\.", "")
           .str.replace_all(r"[^\p{L}\p{N}]+", " ").str.strip_chars())
    s1 = s1.with_columns(pl.len().over("country", raw).cast(pl.Int32).alias("s1_raw_freq"))
    tg = tg.with_columns(pl.len().over("country", raw).cast(pl.Int32).alias("t_raw_freq"))
    return s1, tg


def block_all(s1: pl.DataFrame, tg: pl.DataFrame, cfg: BlockingConfig,
              query_ids: pl.Series | None = None) -> pl.DataFrame:
    """Candidates for every country (open set). Returns global row indices."""
    s1 = s1.with_columns(pl.int_range(0, pl.len(), dtype=pl.Int32).alias("_g"))
    tg = tg.with_columns(pl.int_range(0, pl.len(), dtype=pl.Int32).alias("_g"))
    out = []
    for country in s1["country"].unique().sort().to_list():
        a = s1.filter(pl.col("country") == country)
        b = tg.filter(pl.col("country") == country)
        log(f"[block] {country}: {a.height:,} S1 vs {b.height:,} targets")
        if b.height == 0:
            continue
        mask = None if query_ids is None else a["entity_id"].is_in(query_ids.implode()).to_numpy()
        if mask is not None and not mask.any():
            continue
        r = block_country(a, b, cfg, query_mask=mask, log=log)
        ag, bg = a["_g"].to_numpy(), b["_g"].to_numpy()
        r = r.with_columns(pl.Series("s1_idx", ag[r["s1_idx"].to_numpy()]),
                           pl.Series("t_idx", bg[r["t_idx"].to_numpy()]))
        out.append(r)
        del a, b
        gc.collect()
    cand = pl.concat(out, how="diagonal_relaxed")
    cand = cand.with_columns(pl.len().over("s1_idx").cast(pl.Int32).alias("s1_ncand"))
    log(f"[block] {cand.height:,} candidate pairs")
    return cand


def featurize(cand: pl.DataFrame, s1: pl.DataFrame, tg: pl.DataFrame) -> pl.DataFrame:
    """Join fields and compute pair features in chunks. Keeps s1_idx/t_idx keys."""
    s1f = s1.select([pl.col(c).alias(f"{c}_1") for c in SIDE_COLS + ["country"]] + ["s1_name_freq", "s1_raw_freq"])
    tgf = tg.select([pl.col(c).alias(f"{c}_2") for c in SIDE_COLS] + ["t_name_freq", "t_raw_freq", "entity_id"])
    # large inputs: spill finished chunks to disk so peak RAM stays one chunk wide
    spill = WORK_DIR / "_feat_spill" if cand.height > 3 * FEAT_CHUNK else None
    if spill is not None:
        spill.mkdir(parents=True, exist_ok=True)
        for old in spill.glob("*.parquet"):
            old.unlink()
    parts = []
    for k, i in enumerate(range(0, cand.height, FEAT_CHUNK)):
        c = cand.slice(i, FEAT_CHUNK)
        si, ti = c["s1_idx"].to_numpy(), c["t_idx"].to_numpy()
        d = pl.concat([c, s1f[si], tgf[ti].rename({"entity_id": "t_id"})], how="horizontal")
        f = pl.concat([c.select("s1_idx", "t_idx"), pair_features(d)], how="horizontal")
        del d
        if spill is not None:
            f.write_parquet(spill / f"part_{k:04d}.parquet")
            del f
            gc.collect()
        else:
            parts.append(f)
        log(f"[features] {min(i + FEAT_CHUNK, cand.height):,}/{cand.height:,}")
    if spill is not None:
        return pl.read_parquet(spill / "*.parquet")
    return pl.concat(parts)


def feature_cols(df: pl.DataFrame):
    return [c for c in df.columns if c not in ("s1_idx", "t_idx", "y", "s1_id", "t_id", "p")]


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def run_train(args):
    tr = get_transliterator(Path(args.train_dir), log=log)
    s1, tg = load_normalized(Path(args.train_dir), "train", tr, log=log)
    if args.drop_s1_frac > 0:
        # Test-like conditions: the test set has ~2.3 unowned S2/S3 records per S1
        # vs ~1.2 in train.  Removing a fraction of train S1 entities turns their
        # S2/S3 records into unowned look-alikes and reproduces the test density.
        keep = s1["entity_id"].hash(seed=RANDOM_SEED) % 10_000 >= int(args.drop_s1_frac * 10_000)
        s1 = s1.filter(keep)
        log(f"[train] test-like mode: kept {s1.height:,} S1 (dropped {args.drop_s1_frac:.0%})")
    s1, tg = add_name_freq(s1, tg)
    ids = s1["entity_id"].sample(args.n_train + args.n_val, seed=RANDOM_SEED, shuffle=True)
    train_ids, val_ids = ids[: args.n_train], ids[args.n_train:]

    cand_path = WORK_DIR / f"train_cand_{args.n_train}_{args.n_val}_d{int(args.drop_s1_frac*100)}.parquet"
    if cand_path.exists():
        cand = pl.read_parquet(cand_path)
    else:
        cand = block_all(s1, tg, BlockingConfig(), query_ids=ids)
        cand.write_parquet(cand_path)
    # idempotent: cached candidates may predate the cap
    cand = apply_cap(cand.drop("s1_ncand", "t_ncand", strict=False), BlockingConfig())
    cand = cand.with_columns(pl.len().over("s1_idx").cast(pl.Int32).alias("s1_ncand"))

    # generic "noise" words the generator adds to true-match names, learned from
    # ground-truth pairs of the training S1s only (never the validation S1s)
    truth_tr = gt_pairs(read_ground_truth(Path(args.train_dir))).join(
        pl.DataFrame({"s1_id": train_ids}), on="s1_id")
    nm1 = dict(zip(s1["entity_id"].to_list(), s1["name_core"].to_list()))
    nm2 = tg.select("entity_id", "name_core").join(truth_tr.select(pl.col("t_id").alias("entity_id"),
                                                                   "s1_id"), on="entity_id")
    noise = learn_noise_tokens(pl.Series([nm1[x] for x in nm2["s1_id"].to_list()]), nm2["name_core"])
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    NOISE_PATH.write_text(json.dumps(noise))
    set_noise_tokens(noise)
    log(f"[train] {len(noise)} generic noise tokens learned, e.g. {noise[:20]}")

    set_vocab(s1["name_core"])
    feat_path = WORK_DIR / f"train_feat_{args.n_train}_{args.n_val}_d{int(args.drop_s1_frac*100)}.parquet"
    if feat_path.exists():
        feats = pl.read_parquet(feat_path)
    else:
        feats = featurize(cand, s1, tg)
        feats.write_parquet(feat_path)
    del cand
    gc.collect()

    # labels
    truth = gt_pairs(read_ground_truth(Path(args.train_dir)))
    truth = truth.join(pl.DataFrame({"s1_id": ids}), on="s1_id")
    feats = feats.with_columns(pl.Series("s1_id", s1["entity_id"].to_numpy()[feats["s1_idx"].to_numpy()]),
                               pl.Series("t_id", tg["entity_id"].to_numpy()[feats["t_idx"].to_numpy()]))
    feats = feats.join(truth.with_columns(pl.lit(1, pl.Int8).alias("y")), on=["s1_id", "t_id"], how="left")
    feats = feats.with_columns(pl.col("y").fill_null(0))
    is_val = feats["s1_id"].is_in(val_ids.implode())
    blocked_recall = feats["y"].sum() / truth.height
    log(f"[train] pairs {feats.height:,}  positives {feats['y'].sum():,}  blocking recall {blocked_recall:.4f}")

    cols = feature_cols(feats)
    trn, val = feats.filter(~is_val), feats.filter(is_val)
    del feats
    gc.collect()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    if args.mode == "stage2" and MODEL_PATH.exists():
        booster = lgb.Booster(model_file=str(MODEL_PATH))
    else:
        booster = train_lgb(trn.select(cols).to_numpy().astype(np.float32), trn["y"].to_numpy(), cols,
                            val.select(cols).to_numpy().astype(np.float32), val["y"].to_numpy(),
                            rounds=args.rounds, log=log)
        booster.save_model(str(MODEL_PATH), num_iteration=booster.best_iteration)
        imp = sorted(zip(booster.feature_importance("gain"), cols), reverse=True)
        log("[model] top features: " + ", ".join(f"{n}={g:.0f}" for g, n in imp[:15]))
    n_round = booster.best_iteration if booster.best_iteration > 0 else booster.current_iteration()

    val = val.with_columns(pl.Series("p", booster.predict(val.select(cols).to_numpy().astype(np.float32),
                                                          num_iteration=n_round).astype(np.float32)))
    vtruth = truth.join(pl.DataFrame({"s1_id": val_ids}), on="s1_id")
    pol1 = tune_policy(val.select("s1_id", "t_id", "p"), vtruth, val_ids, exclusive=True, log=log)
    f1 = macro_f05(pol1.apply(val.select("s1_id", "t_id", "p")), vtruth, val_ids)
    log(f"[val] stage-1 macro F0.5 = {f1:.5f} on {len(val_ids):,} held-out S1")

    # ---- stage 2: out-of-fold stage-1 probabilities -> group-context model ----
    oof_path = feat_path.with_name(feat_path.stem + "_oof.parquet")
    if oof_path.exists() and oof_path.stat().st_mtime > MODEL_PATH.stat().st_mtime:
        trn = trn.join(pl.read_parquet(oof_path), on=["s1_idx", "t_idx"])
    else:
        X = trn.select(cols).to_numpy().astype(np.float32)
        y = trn["y"].to_numpy()
        fold = (trn["s1_idx"].hash(seed=1) % 3).to_numpy()
        oof = np.zeros(len(y), np.float32)
        for k in range(3):
            m = fold == k
            b = lgb.train(LGB_PARAMS, lgb.Dataset(X[~m], y[~m], feature_name=cols), num_boost_round=n_round)
            oof[m] = b.predict(X[m])
            log(f"[stage2] out-of-fold stage-1 fold {k + 1}/3 done")
        del X, b
        gc.collect()
        trn = trn.with_columns(pl.Series("p", oof))
        trn.select("s1_idx", "t_idx", "p").write_parquet(oof_path)
    t_name, t_addr = tg["name_core"].to_numpy(), tg["addr"].to_numpy()
    t_is3 = tg["entity_id"].str.starts_with("S3-").to_numpy()
    T = stage2_frame(trn, t_name, t_addr, t_is3).join(
        trn.select(pl.col("s1_idx").alias("g"), pl.col("t_idx").alias("t"), "y"), on=["g", "t"])
    V = stage2_frame(val, t_name, t_addr, t_is3).join(
        val.select(pl.col("s1_idx").alias("g"), pl.col("t_idx").alias("t"), "y", "s1_id", "t_id"), on=["g", "t"])
    p2 = dict(LGB_PARAMS, num_leaves=63, learning_rate=0.03, min_data_in_leaf=200)
    b2 = lgb.train(p2, lgb.Dataset(T.select(STAGE2_COLS).to_numpy(), T["y"].to_numpy(), feature_name=STAGE2_COLS),
                   3000, valid_sets=[lgb.Dataset(V.select(STAGE2_COLS).to_numpy(), V["y"].to_numpy())],
                   callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(500)])
    V = V.with_columns(pl.Series("p", b2.predict(V.select(STAGE2_COLS).to_numpy(),
                                                 num_iteration=b2.best_iteration).astype(np.float32)))
    pol2 = tune_policy(V.select("s1_id", "t_id", "p"), vtruth, val_ids, exclusive=True, log=log)
    f2 = macro_f05(pol2.apply(V.select("s1_id", "t_id", "p")), vtruth, val_ids)
    imp = sorted(zip(b2.feature_importance("gain"), STAGE2_COLS), reverse=True)
    log("[stage2] top features: " + ", ".join(f"{n}={g:.0f}" for g, n in imp[:10]))
    log(f"[val] stage-2 macro F0.5 = {f2:.5f} (stage-1 {f1:.5f})")
    V.select("s1_id", "t_id", "p", "y").write_parquet(WORK_DIR / "val_scored.parquet")
    if f2 > f1:
        b2.save_model(str(STAGE2_PATH), num_iteration=b2.best_iteration)
        pol2.save(POLICY_PATH)
    else:
        STAGE2_PATH.unlink(missing_ok=True)
        pol1.save(POLICY_PATH)


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

def run_predict(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tr = get_transliterator(Path(args.train_dir), log=log)
    s1, tg = load_normalized(Path(args.test_dir), "test", tr, log=log)
    s1, tg = add_name_freq(s1, tg)
    booster = lgb.Booster(model_file=str(MODEL_PATH))
    set_noise_tokens(json.loads(NOISE_PATH.read_text()))
    set_vocab(s1["name_core"])
    policy = Policy.load(POLICY_PATH)
    cols = booster.feature_name()

    # memory diet: keep only what scoring needs; IDs stay as Arrow strings
    # (10M Python str objects would push a 16 GB machine into swap)
    s1 = s1.select(SIDE_COLS + ["country", "entity_id", "s1_name_freq", "s1_raw_freq"])
    tg = tg.select(SIDE_COLS + ["entity_id", "t_name_freq", "t_raw_freq"])
    gc.collect()
    s1_ids = s1["entity_id"]
    t_ids = tg["entity_id"]
    cand_path = WORK_DIR / "test_cand.parquet"
    cand_tsv = out_dir / "candidate_pairs.tsv"
    if not cand_path.exists():
        cand = block_all(s1, tg, BlockingConfig())
        cand.write_parquet(cand_path)
        del cand
        gc.collect()
    if not cand_tsv.exists() or cand_tsv.stat().st_mtime < cand_path.stat().st_mtime:
        write_id_lists(pl.read_parquet(cand_path, columns=["s1_idx", "t_idx"]), s1_ids, t_ids,
                       cand_tsv, "candidate_entity_ids")
        log("[predict] candidate_pairs.tsv written")

    # ---- stage 1: every candidate pair, streamed to per-chunk parquet files -----
    s1_dir = WORK_DIR / "test_scores_s1"
    score_candidates(cand_path, s1, tg, booster, cols, s1_dir, None)
    lf = pl.scan_parquet(s1_dir / "*.parquet")

    # ---- countries unseen in training (France): learn their generic noise words
    #      from confident matches, then re-score only that country's pairs --------
    unseen = sorted(set(s1["country"].unique()) - train_countries(Path(args.train_dir)))
    if unseen and args.adapt_unseen:
        rows = np.flatnonzero(s1["country"].is_in(unseen).to_numpy()).astype(np.int32)
        # address-anchored confident matches: same house number, near-identical
        # address, p >= 0.5 -- independent of how the name model treats the words
        anchor = (pl.col("p") >= 0.5) & (pl.col("a_tset") >= 95) & (pl.col("hn_rel") == 4)
        conf = lf.filter(anchor).select("s1_idx", "t_idx", "p").collect()
        conf = conf.filter(pl.col("s1_idx").is_in(pl.Series(rows).implode()))
        s1_country = s1["country"].to_numpy()
        for c in unseen:
            cc = conf.filter(pl.Series(s1_country[conf["s1_idx"].to_numpy()] == c))
            cc = cc.sample(min(cc.height, 300_000), seed=RANDOM_SEED)
            toks = learn_noise_tokens(pl.Series(s1["name_core"].to_numpy()[cc["s1_idx"].to_numpy()]),
                                      pl.Series(tg["name_core"].to_numpy()[cc["t_idx"].to_numpy()]), min_count=60)
            COUNTRY_NOISE[c] = set(toks) - NOISE_TOKENS
            log(f"[adapt] {c}: {len(COUNTRY_NOISE[c])} country-specific noise words from {cc.height:,} "
                f"confident matches: {sorted(COUNTRY_NOISE[c])[:40]}")
        (ARTIFACT_DIR / "country_noise_test.json").write_text(
            json.dumps({k: sorted(v) for k, v in COUNTRY_NOISE.items()}, ensure_ascii=False))
        del conf
        ad_dir = WORK_DIR / "test_scores_adapted"
        in_rows = pl.col("s1_idx").is_in(pl.Series(rows).implode())
        score_candidates(cand_path, s1, tg, booster, cols, ad_dir, in_rows)
        lf = pl.concat([lf.filter(~in_rows), pl.scan_parquet(ad_dir / "*.parquet")])

    # ---- stage 2 (group context) --------------------------------------------
    if STAGE2_PATH.exists():
        scored = apply_stage2(lf, tg, len(s1_ids))
    else:
        scored = lf.select("s1_idx", "t_idx", "p").collect()
    scored.write_parquet(WORK_DIR / "test_final_scores_base.parquet")

    # ---- unseen countries: generator noise words, guarded by the house number ---
    noise = {c: UNSEEN_NOISE[c] for c in unseen if c in UNSEEN_NOISE}
    if noise and not args.adapt_unseen:
        if UNSEEN_MODEL_PATH.exists():
            # the self-trained model learned the house-number guard from its pseudo-labels
            scored = rescore_unseen_noise(scored, s1, tg, lgb.Booster(model_file=str(UNSEEN_MODEL_PATH)),
                                          cols, cand_path, noise, guard=False)
        else:
            scored = rescore_unseen_noise(scored, s1, tg, booster, cols, cand_path, noise, guard=True)
    scored.write_parquet(WORK_DIR / "test_final_scores.parquet")

    # ---- decision rule -------------------------------------------------------
    if RULE_PATH.exists():
        noaddr = tg.select(pl.int_range(pl.len(), dtype=pl.Int32).alias("t_idx"),
                           (pl.col("addr").is_null() | (pl.col("addr").str.strip_chars() == "")).alias("noaddr"))
        sel = LeaderboardRule.load(RULE_PATH).apply(scored, noaddr)
        sel = quota_and_duplicates(sel, scored, tg)
    else:
        sel = policy.apply(scored.rename({"s1_idx": "s1_id", "t_idx": "t_id"})).rename(
            {"s1_id": "s1_idx", "t_id": "t_idx"})
    write_id_lists(sel, s1_ids, t_ids, out_dir / "matching_results.tsv", "matched_entity_ids")
    log(f"[predict] matching_results.tsv written: {sel.height:,} matches, "
        f"{sel['s1_idx'].n_unique():,}/{len(s1_ids):,} S1 with >=1 match")


def rescore_unseen_noise(scored: pl.DataFrame, s1: pl.DataFrame, tg: pl.DataFrame, booster, cols,
                         cand_path: Path, noise: dict, guard: bool = True) -> pl.DataFrame:
    """
    Re-score the pairs of countries unseen in training with their generator noise words
    switched on (COUNTRY_NOISE), run stage 2 on them, and take the new probability only
    where the S1 and target house numbers agree or the target has no address.  Look-alike
    records reuse the same words with a *different* house number, so every other pair keeps
    its original score.  With guard=False (self-trained model) every pair of the country
    takes the new score.
    """
    rows = pl.Series(np.flatnonzero(s1["country"].is_in(list(noise)).to_numpy()).astype(np.int32))
    in_rows = pl.col("s1_idx").is_in(rows.implode())
    model_id = "" if guard else str(UNSEEN_MODEL_PATH.stat().st_mtime_ns)
    key = hashlib.md5((json.dumps(noise, sort_keys=True) + str(guard) + model_id).encode()).hexdigest()[:8]
    nz_dir = WORK_DIR / f"test_scores_unseen_noise_{key}"       # cache keyed on the word list
    COUNTRY_NOISE.clear()
    COUNTRY_NOISE.update({c: set(w) for c, w in noise.items()})
    score_candidates(cand_path, s1, tg, booster, cols, nz_dir, in_rows)
    COUNTRY_NOISE.clear()
    nz = apply_stage2(pl.scan_parquet(nz_dir / "*.parquet"), tg, s1.height)
    if not guard:
        log(f"[unseen] {noise}: {nz.height:,} pairs re-scored with the self-trained model")
        return pl.concat([scored.join(nz.select("s1_idx").unique(), on="s1_idx", how="anti"), nz.select(scored.columns)])
    num = lambda c: pl.col(c).str.extract(r"(\d+)", 1).cast(pl.Int64, strict=False)
    n1 = s1.select(pl.int_range(pl.len(), dtype=pl.Int32).alias("s1_idx"), num("addr").alias("n1"))
    n2 = tg.select(pl.int_range(pl.len(), dtype=pl.Int32).alias("t_idx"), num("addr").alias("n2"),
                   (pl.col("addr").is_null() | (pl.col("addr").str.strip_chars() == "")).alias("noaddr"))
    nz = nz.join(n1, on="s1_idx").join(n2, on="t_idx")
    ok = (nz.filter(pl.col("noaddr") | (pl.col("n1").is_not_null() & (pl.col("n1") == pl.col("n2"))))
          .select("s1_idx", "t_idx", pl.col("p").alias("p_nz")))
    log(f"[unseen] {noise}: {ok.height:,} pairs take the noise-word score")
    return (scored.join(ok, on=["s1_idx", "t_idx"], how="left")
            .with_columns(pl.coalesce("p_nz", "p").alias("p")).drop("p_nz"))


def quota_and_duplicates(sel: pl.DataFrame, scored: pl.DataFrame, tg: pl.DataFrame) -> pl.DataFrame:
    """
    Two structural facts of the training ground truth, applied after the decision rule:
    * no S1 has more than 5 matches in Source 2 or more than 6 in Source 3, so only an S1's
      5 best S2 links and 6 best S3 links are kept (affects ~70 links on test);
    * identical S2/S3 records (same raw name and address) belong to the same business, so an
      unselected candidate that duplicates a selected record of the same S1, and whose best
      S1 is that S1, is added (~40 links on test).
    """
    t_is3 = tg["entity_id"].str.starts_with("S3").to_numpy()
    sel = sel.join(scored, on=["s1_idx", "t_idx"])
    sel = sel.with_columns(pl.Series("is3", t_is3[sel["t_idx"].to_numpy()]))
    sel = sel.with_columns(pl.col("p").rank("ordinal", descending=True).over("s1_idx", "is3").alias("rk_src"))
    over = (~pl.col("is3") & (pl.col("rk_src") > 5)) | (pl.col("is3") & (pl.col("rk_src") > 6))
    n_over = sel.filter(over).height
    sel = sel.filter(~over).select("s1_idx", "t_idx")
    key = pl.concat_str([pl.col("business_name").fill_null("").str.to_lowercase().str.strip_chars(),
                         pl.col("business_address").fill_null("").str.to_lowercase().str.strip_chars()], separator="|")
    tk = tg.select(pl.int_range(pl.len(), dtype=pl.Int32).alias("t_idx"), key.alias("tkey"))
    anchors = sel.join(tk, on="t_idx").select("s1_idx", "tkey").unique()
    best = scored.filter(pl.col("p") == pl.col("p").max().over("t_idx")).join(sel.select("t_idx"), on="t_idx", how="anti")
    dup = best.join(tk, on="t_idx").join(anchors, on=["s1_idx", "tkey"]).select("s1_idx", "t_idx")
    log(f"[rule] source quota removed {n_over} links; exact duplicates added {dup.height}")
    return pl.concat([sel, dup])


def train_countries(train_dir: Path) -> set:
    return set(pl.scan_csv(train_dir / "train_source1.tsv", separator="\t", quote_char=None,
                           infer_schema=False).select("country").unique().collect()["country"].to_list())


def apply_stage2(lf: pl.LazyFrame, tg: pl.DataFrame, n_s1: int, step: int = 50_000) -> pl.DataFrame:
    """Replace stage-1 p by the stage-2 (group context) probability, chunked by S1 groups."""
    b2 = lgb.Booster(model_file=str(STAGE2_PATH))
    t_name, t_addr = tg["name_core"].to_numpy(), tg["addr"].to_numpy()
    t_is3 = tg["entity_id"].str.starts_with("S3-").to_numpy()
    out = []
    for a in range(0, n_s1, step):
        d = lf.filter((pl.col("s1_idx") >= a) & (pl.col("s1_idx") < a + step)).collect()
        if d.height == 0:
            continue
        F = stage2_frame(d, t_name, t_addr, t_is3)
        out.append(F.select(pl.col("g").alias("s1_idx"), pl.col("t").alias("t_idx")).with_columns(
            pl.Series("p", b2.predict(F.select(STAGE2_COLS).to_numpy()).astype(np.float32))))
        if (a // step) % 8 == 7 or a + step >= n_s1:
            log(f"[stage2] S1 {min(a + step, n_s1):,}/{n_s1:,}")
        del d, F
        gc.collect()
    return pl.concat(out)


def score_candidates(cand_path: Path, s1, tg, booster, cols, out_dir: Path, where: pl.Expr | None):
    """
    Stream candidates (filtered by `where`) from parquet in chunks, featurize and
    score them, writing one parquet per chunk to out_dir (bounded RAM).  Chunks
    already written after the model/candidates were last changed are reused.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    newest_input = max(cand_path.stat().st_mtime, MODEL_PATH.stat().st_mtime)
    for old in out_dir.glob("*.parquet"):
        if old.stat().st_mtime < newest_input:
            old.unlink()
    if where is None:
        # plain row slices push down into the parquet reader (no full re-read)
        lf = pl.scan_parquet(cand_path)
        n = lf.select(pl.len()).collect().item()
        get = lambda i, step: lf.slice(i, step).collect()
    else:
        sub = pl.scan_parquet(cand_path).filter(where).collect()
        n = sub.height
        get = lambda i, step: sub.slice(i, step)
    step = FEAT_CHUNK * 2
    for k, i in enumerate(range(0, n, step)):
        path = out_dir / f"part_{k:04d}.parquet"
        if path.exists():
            continue
        f = featurize(get(i, step), s1, tg)
        p = booster.predict(f.select(cols).to_numpy().astype(np.float32))
        (f.select("s1_idx", "t_idx", *KEEP_FOR_STAGE2).with_columns(pl.Series("p", p.astype(np.float32)))
         .write_parquet(path))
        del f
        gc.collect()
        log(f"[predict] {out_dir.name}: scored {min(i + step, n):,}/{n:,}")


def write_id_lists(pairs: pl.DataFrame, s1_ids: np.ndarray, t_ids: np.ndarray, path: Path, col: str):
    """One row per S1 (all S1 included), comma-joined unique target ids."""
    u = pairs.select("s1_idx", "t_idx").unique(maintain_order=True)
    g = (u.with_columns(pl.Series("t", t_ids.gather(u["t_idx"]) if isinstance(t_ids, pl.Series)
                                  else t_ids[u["t_idx"].to_numpy()]))
         .group_by("s1_idx").agg(pl.col("t").unique().sort().str.join(",").alias(col)))
    base = pl.DataFrame({"s1_idx": np.arange(len(s1_ids), dtype=np.int32),
                         "source1_entity_id": s1_ids})
    out = (base.join(g.with_columns(pl.col("s1_idx").cast(pl.Int32)), on="s1_idx", how="left")
           .select("source1_entity_id", pl.col(col).fill_null("")))
    out.write_csv(path, separator="\t", quote_style="never")


def parse_args():
    p = argparse.ArgumentParser(description="Business Entity Resolution pipeline")
    p.add_argument("--mode", choices=["all", "train", "stage2", "predict"], default="all")
    p.add_argument("--train-dir", default=str(TRAIN_DIR))
    p.add_argument("--test-dir", default=str(TEST_DIR))
    p.add_argument("--output-dir", default=str(OUTPUT_DIR))
    p.add_argument("--n-train", type=int, default=80_000, help="train S1 entities sampled for fitting")
    p.add_argument("--n-val", type=int, default=20_000, help="held-out S1 entities for policy tuning")
    p.add_argument("--adapt-unseen", action="store_true",
                   help="old label-free noise-word learning for unseen countries (hurt France on the leaderboard); "
                        "replaces the curated UNSEEN_NOISE step when set")
    p.add_argument("--drop-s1-frac", type=float, default=0.19,
                   help="fraction of train S1 removed to mimic the test distractor density")
    p.add_argument("--rounds", type=int, default=3000)
    return p.parse_args()


def main():
    args = parse_args()
    if args.mode in ("train", "stage2", "all"):
        run_train(args)
    if args.mode in ("predict", "all"):
        run_predict(args)


if __name__ == "__main__":
    main()
