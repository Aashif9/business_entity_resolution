"""
Matcher (LightGBM) and entity-level decision policy.

The decision policy turns pair probabilities into a set of matches per S1:

  1. exclusivity : every target belongs to at most one S1 in the ground truth,
                   so a target is only kept for the S1 where it scores highest
                   (applied when all S1s are scored, i.e. at inference time);
  2. threshold   : keep pairs with p >= tau;
  3. singleton   : if the best pair of an S1 is below tau_top, predict empty.

tau / tau_top are chosen by grid search on the exact macro F0.5 of a held-out
set of S1 entities (true pairs lost in blocking count as misses).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import lightgbm as lgb
import numpy as np
import polars as pl

LGB_PARAMS = dict(
    objective="binary",
    learning_rate=0.05,
    num_leaves=127,
    min_data_in_leaf=100,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    max_bin=255,
    verbose=-1,
    num_threads=0,
    seed=42,
)


def train_lgb(X: np.ndarray, y: np.ndarray, feat_names: List[str], X_val=None, y_val=None,
              rounds: int = 2000, log=print) -> lgb.Booster:
    dtr = lgb.Dataset(X, y, feature_name=feat_names, free_raw_data=True)
    valid = [dtr]
    names = ["train"]
    if X_val is not None:
        valid.append(lgb.Dataset(X_val, y_val, reference=dtr))
        names.append("val")
    cb = [lgb.log_evaluation(200)]
    if X_val is not None:
        cb.append(lgb.early_stopping(100, verbose=False))
    booster = lgb.train(LGB_PARAMS, dtr, num_boost_round=rounds, valid_sets=valid,
                        valid_names=names, callbacks=cb)
    log(f"[model] best iteration {booster.best_iteration}")
    return booster


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------

def macro_f05(pred: pl.DataFrame, truth: pl.DataFrame, s1_ids: pl.Series) -> float:
    """
    Exact challenge metric.
    pred  : (s1_id, t_id) predicted pairs
    truth : (s1_id, t_id) true pairs
    s1_ids: all evaluated S1 ids (singletons included)
    """
    base = pl.DataFrame({"s1_id": s1_ids})
    npred = pred.group_by("s1_id").len("n_pred")
    ntrue = truth.group_by("s1_id").len("n_true")
    tp = pred.join(truth, on=["s1_id", "t_id"]).group_by("s1_id").len("tp")
    d = (base.join(npred, on="s1_id", how="left").join(ntrue, on="s1_id", how="left")
         .join(tp, on="s1_id", how="left").fill_null(0))
    p = d["tp"] / d["n_pred"].clip(1)
    r = d["tp"] / d["n_true"].clip(1)
    f = (1.25 * p * r / (0.25 * p + r)).fill_nan(0.0)
    f = pl.Series(np.where((d["n_true"] == 0).to_numpy(), (d["n_pred"] == 0).to_numpy().astype(float),
                           f.to_numpy()))
    return float(f.mean())


# ---------------------------------------------------------------------------
# Decision policy
# ---------------------------------------------------------------------------

@dataclass
class LeaderboardRule:
    """
    Final decision rule (every threshold confirmed on the public leaderboard):
    1. exclusivity: a record goes only to its highest-scoring S1;
    2. records WITH an address: keep links with p >= extras (and, optionally, whose best S1
       holds >= min_share_addr of the record's probability mass);
    3. records WITHOUT an address (~98% are real copies): keep the link if the best S1 is
       dominant -- (p, share) passes any of noaddr_tiers, share = p / sum of p over the
       record's S1 candidates;
    4. each S1's single best candidate is kept if p >= top1.
    """
    extras: float = 0.86
    top1: float = 0.55
    noaddr_tiers: tuple = ((0.60, 0.70), (0.40, 0.90), (0.30, 0.95))
    min_share_addr: float = 0.0

    def apply(self, scored: pl.DataFrame, noaddr: pl.DataFrame | None = None) -> pl.DataFrame:
        """scored: (s1_idx, t_idx, p); noaddr: (t_idx, noaddr) -> selected (s1_idx, t_idx)."""
        tot = scored.group_by("t_idx").agg(pl.col("p").sum().alias("p_sum"))
        d = scored.filter(pl.col("p") == pl.col("p").max().over("t_idx"))
        d = d.with_columns(pl.col("p").rank("ordinal", descending=True).over("s1_idx").alias("rk"))
        d = d.join(tot, on="t_idx").with_columns((pl.col("p") / pl.col("p_sum")).alias("share"))
        addr_ok = (pl.col("p") >= self.extras) & (pl.col("share") >= self.min_share_addr)
        keep_extra = addr_ok
        if noaddr is not None:
            d = d.join(noaddr, on="t_idx", how="left").with_columns(pl.col("noaddr").fill_null(False))
            na_ok = pl.lit(False)
            for p_min, share_min in self.noaddr_tiers:
                na_ok = na_ok | ((pl.col("p") >= p_min) & (pl.col("share") >= share_min))
            keep_extra = pl.when(pl.col("noaddr")).then(na_ok).otherwise(addr_ok)
        keep = keep_extra | ((pl.col("rk") == 1) & (pl.col("p") >= self.top1))
        return d.filter(keep).select("s1_idx", "t_idx")

    def save(self, path: Path):
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "LeaderboardRule":
        cfg = json.loads(Path(path).read_text())
        if "noaddr_tiers" in cfg:
            cfg["noaddr_tiers"] = tuple(tuple(t) for t in cfg["noaddr_tiers"])
        return cls(**cfg)


@dataclass
class Policy:
    tau: float = 0.5
    tau_top: float = 0.5
    exclusive: bool = True

    def apply(self, scored: pl.DataFrame) -> pl.DataFrame:
        """scored: (s1_id, t_id, p) -> selected (s1_id, t_id)."""
        d = scored
        if self.exclusive:
            d = d.filter(pl.col("p") == pl.col("p").max().over("t_id"))
        d = d.with_columns(pl.col("p").max().over("s1_id").alias("pmax"))
        d = d.filter((pl.col("p") >= self.tau) & (pl.col("pmax") >= self.tau_top))
        return d.select("s1_id", "t_id")

    def save(self, path: Path):
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "Policy":
        return cls(**json.loads(Path(path).read_text()))


def tune_policy(scored: pl.DataFrame, truth: pl.DataFrame, s1_ids: pl.Series,
                exclusive: bool, log=print) -> Policy:
    best = (-1.0, None)
    for tau in np.arange(0.20, 0.96, 0.025):
        for tau_top in np.arange(tau, 0.99, 0.05):
            pol = Policy(float(tau), float(tau_top), exclusive)
            s = macro_f05(pol.apply(scored), truth, s1_ids)
            if s > best[0]:
                best = (s, pol)
    log(f"[policy] best macro F0.5 {best[0]:.5f} with {best[1]}")
    return best[1]
