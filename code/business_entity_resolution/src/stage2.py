"""
Stage 2: entity-level (group) context on top of stage-1 pair probabilities.

The candidates of one S1 entity are scored jointly.  Most of the residual
errors of the pair model are *ambiguous in isolation* — typically targets with
an empty address and a name shared by several businesses (26% of S2/S3 records
belong to businesses absent from S1, many of them with colliding names).  The
S1's other candidates disambiguate them: if the same-name records that carry an
address match this S1 confidently, an address-less one probably belongs to it
too; if same-name records at another address were rejected, it more likely
belongs to that other business.

Stage 2 is trained on *out-of-fold* stage-1 probabilities so that its inputs
have the same distribution at training and inference time.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from rapidfuzz import distance, fuzz, process

GROUP_FEATS = [
    "p", "g_max", "g_second", "g_n50", "g_n90", "g_rank", "p_rel", "p_gap_other",
    "same_n", "same_hi", "same_lo", "same_maxp_other",
    "top_addr_tset", "top_name_ratio", "t_addr_empty2", "g_n_empty", "g_ncand",
    "rk_rev", "rev_rel2", "rev_gap2",
    "hn_share", "hn_share_hi", "addr_share", "addr_share_hi", "name_share_hi", "hn_eq_top",
    "hi_name_max", "hi_name_mean", "hi_addr_max", "hi_addr_mean", "n_hi_sib",
    "lo_name_max", "lo_addr_max", "lo_both_max", "n_lo_sib",
    "same_src_hi", "other_src_hi", "t_is3",
]


def _cp(a, b, scorer):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


def group_features(d: pl.DataFrame) -> pl.DataFrame:
    """
    d: one row per candidate with columns
       g (S1 key), t (target key), p (stage-1 prob), tname, taddr,
       rk_rev, sc_rev, rev_top1, rev_top2
    Returns d's keys + GROUP_FEATS.  Every S1 group must be complete in d.
    """
    d = d.with_columns(
        pl.col("p").max().over("g").alias("g_max"),
        (pl.col("p") > 0.5).sum().over("g").alias("g_n50"),
        (pl.col("p") > 0.9).sum().over("g").alias("g_n90"),
        pl.col("p").rank("ordinal", descending=True).over("g").alias("g_rank"),
        pl.len().over("g").alias("g_ncand"),
        (pl.col("taddr") == "").alias("t_addr_empty2"),
    )
    d = d.with_columns(
        pl.col("p").filter(pl.col("g_rank") == 2).first().over("g").fill_null(0.0).alias("g_second"),
        (pl.col("t_addr_empty2")).sum().over("g").alias("g_n_empty"),
    )
    d = d.with_columns(
        (pl.col("p") / pl.col("g_max").clip(1e-6)).alias("p_rel"),
        (pl.col("p") - pl.when(pl.col("g_rank") == 1).then(pl.col("g_second")).otherwise(pl.col("g_max")))
        .alias("p_gap_other"),
    )
    # same-name siblings inside the S1's candidate list (excluding self)
    hi = (pl.col("p") > 0.8).cast(pl.Int32)
    lo = (pl.col("p") < 0.3).cast(pl.Int32)
    d = d.with_columns(
        (pl.len().over("g", "tname") - 1).alias("same_n"),
        (hi.sum().over("g", "tname") - hi).alias("same_hi"),
        (lo.sum().over("g", "tname") - lo).alias("same_lo"),
    )
    # best p among *other* same-name siblings: max over group of p excluding self
    top2 = d.group_by("g", "tname").agg(pl.col("p").top_k(2).alias("_tk"))
    d = d.join(top2, on=["g", "tname"], how="left").with_columns(
        pl.when(pl.col("_tk").list.len() < 2).then(pl.lit(-1.0))
        .when(pl.col("p") >= pl.col("_tk").list.get(0)).then(pl.col("_tk").list.get(1))
        .otherwise(pl.col("_tk").list.get(0)).alias("same_maxp_other")
    ).drop("_tk")
    # similarity to the group's most confident *other* candidate
    ref = d.filter(pl.col("g_rank") <= 2).select("g", "g_rank", pl.col("tname").alias("rn"), pl.col("taddr").alias("ra"))
    r1 = ref.filter(pl.col("g_rank") == 1).drop("g_rank")
    r2 = ref.filter(pl.col("g_rank") == 2).drop("g_rank").rename({"rn": "rn2", "ra": "ra2"})
    d = d.join(r1, on="g", how="left").join(r2, on="g", how="left")
    d = d.with_columns(
        pl.when(pl.col("g_rank") == 1).then(pl.col("rn2")).otherwise(pl.col("rn")).fill_null("").alias("rn"),
        pl.when(pl.col("g_rank") == 1).then(pl.col("ra2")).otherwise(pl.col("ra")).fill_null("").alias("ra"),
    ).drop("rn2", "ra2")
    d = d.with_columns(
        pl.Series("top_addr_tset", _cp(d["taddr"].to_list(), d["ra"].to_list(), fuzz.token_set_ratio)),
        pl.Series("top_name_ratio", _cp(d["tname"].to_list(), d["rn"].to_list(), fuzz.ratio)),
    ).with_columns(
        pl.when(pl.col("taddr") == "").then(-1.0).otherwise(pl.col("top_addr_tset")).alias("top_addr_tset")
    )
    d = d.with_columns(
        (pl.col("sc_rev") / pl.col("rev_top1")).alias("rev_rel2"),
        (pl.col("rev_top1") - pl.col("rev_top2")).alias("rev_gap2"),
    )
    # consensus: a business's own records agree with each other (even where they
    # disagree with S1); a deviation carried by a single record is a look-alike cue
    hi5 = (pl.col("p") > 0.5).cast(pl.Int32)
    d = d.with_columns(pl.col("taddr").str.extract(r"(\d+)").alias("thn"))
    d = d.with_columns(
        pl.when(pl.col("thn").is_null()).then(-1)
        .otherwise(pl.len().over("g", "thn") - 1).alias("hn_share"),
        pl.when(pl.col("thn").is_null()).then(-1)
        .otherwise(hi5.sum().over("g", "thn") - hi5).alias("hn_share_hi"),
        pl.when(pl.col("taddr") == "").then(-1)
        .otherwise(pl.len().over("g", "taddr") - 1).alias("addr_share"),
        pl.when(pl.col("taddr") == "").then(-1)
        .otherwise(hi5.sum().over("g", "taddr") - hi5).alias("addr_share_hi"),
        (hi5.sum().over("g", "tname") - hi5).alias("name_share_hi"),
    )
    top_hn = (d.filter(pl.col("g_rank") <= 2)
              .select("g", "g_rank", pl.col("thn").alias("_h")))
    d = (d.join(top_hn.filter(pl.col("g_rank") == 1).drop("g_rank"), on="g", how="left")
         .join(top_hn.filter(pl.col("g_rank") == 2).drop("g_rank").rename({"_h": "_h2"}), on="g", how="left"))
    ref = pl.when(pl.col("g_rank") == 1).then(pl.col("_h2")).otherwise(pl.col("_h"))
    d = d.with_columns(pl.when(pl.col("thn").is_null() | ref.is_null()).then(-1)
                       .otherwise((pl.col("thn") == ref).cast(pl.Int32)).alias("hn_eq_top"))
    # cross-source slots: 80% of businesses have records in both S2 and S3, so a
    # candidate from a source where the S1 has no confident record fills a gap
    if "is3" in d.columns:
        h = (pl.col("p") > 0.5).cast(pl.Int32)
        d = d.with_columns((h * pl.col("is3").cast(pl.Int32)).sum().over("g").alias("_h3"),
                           (h * (~pl.col("is3")).cast(pl.Int32)).sum().over("g").alias("_h2"))
        d = d.with_columns(
            (pl.when(pl.col("is3")).then(pl.col("_h3")).otherwise(pl.col("_h2")) - h).alias("same_src_hi"),
            pl.when(pl.col("is3")).then(pl.col("_h2")).otherwise(pl.col("_h3")).alias("other_src_hi"),
            pl.col("is3").cast(pl.Int32).alias("t_is3"))
    else:
        d = d.with_columns(pl.lit(-1).alias("same_src_hi"), pl.lit(-1).alias("other_src_hi"), pl.lit(-1).alias("t_is3"))
    d = d.join(sibling_features(d), on=["g", "t"], how="left").with_columns(
        [pl.col(c).fill_null(-1.0) for c in SIB_FEATS])
    return d.select("g", "t", *[pl.col(c).cast(pl.Float32) for c in GROUP_FEATS])


SIB_FEATS = ["hi_name_max", "hi_name_mean", "hi_addr_max", "hi_addr_mean", "n_hi_sib",
             "lo_name_max", "lo_addr_max", "lo_both_max", "n_lo_sib"]


def sibling_features(d: pl.DataFrame, hi_p: float = 0.5, lo_lo: float = 0.02, lo_hi: float = 0.3,
                     max_lo: int = 8) -> pl.DataFrame:
    """
    Record-to-record agreement inside each S1 group.
    hi siblings : the S1's confident candidates (p > hi_p) other than t
    lo siblings : plausible-but-rejected candidates (lo_lo < p < lo_hi), top max_lo by p
    A true record resembles the confident records; a look-alike resembles the
    rejected records of the other business it actually belongs to.
    """
    base = d.select("g", "t", "p", "tname", "taddr")
    hi = base.filter(pl.col("p") > hi_p).select("g", pl.col("t").alias("t2"), pl.col("tname").alias("n2"),
                                               pl.col("taddr").alias("a2"))
    lo = (base.filter((pl.col("p") > lo_lo) & (pl.col("p") < lo_hi))
          .sort("p", descending=True).group_by("g", maintain_order=True).head(max_lo)
          .select("g", pl.col("t").alias("t2"), pl.col("tname").alias("n2"), pl.col("taddr").alias("a2")))
    out = base.select("g", "t")
    for tag, sib in (("hi", hi), ("lo", lo)):
        x = base.select("g", "t", "tname", "taddr").join(sib, on="g").filter(pl.col("t") != pl.col("t2"))
        if x.height == 0:
            continue
        ns = _cp(x["tname"].to_list(), x["n2"].to_list(), fuzz.ratio)
        a_ok = ((x["taddr"] != "") & (x["a2"] != "")).to_numpy()
        as_ = np.where(a_ok, _cp(x["taddr"].to_list(), x["a2"].to_list(), fuzz.token_set_ratio), np.nan)
        x = x.select("g", "t").with_columns(pl.Series("ns", ns), pl.Series("as", as_, nan_to_null=True))
        if tag == "hi":
            agg = x.group_by("g", "t").agg(pl.col("ns").max().alias("hi_name_max"), pl.col("ns").mean().alias("hi_name_mean"),
                                           pl.col("as").max().alias("hi_addr_max"), pl.col("as").mean().alias("hi_addr_mean"),
                                           pl.len().alias("n_hi_sib"))
        else:
            agg = x.group_by("g", "t").agg(pl.col("ns").max().alias("lo_name_max"), pl.col("as").max().alias("lo_addr_max"),
                                           ((pl.col("ns") + pl.col("as").fill_null(0)) / 2).max().alias("lo_both_max"),
                                           pl.len().alias("n_lo_sib"))
        out = out.join(agg, on=["g", "t"], how="left")
    for c in SIB_FEATS:
        if c not in out.columns:
            out = out.with_columns(pl.lit(None, pl.Float32).alias(c))
    return out


# stage-1 columns (stored next to p at inference) that stage 2 may use directly
EXTRA_COLS = ["t_addr_empty", "n_exact", "n_tset", "a_tset", "hn_rel", "tk_extra2_real", "tk_miss1_80",
              "tk_extra2_vocab", "tk_extra2_oov", "raw_ratio", "lg_t_extra", "lg_equal",
              "s1_name_freq", "t_name_freq", "s1_raw_freq", "t_raw_freq", "sc_rev", "rev_top1", "rev_top2"]
STAGE2_COLS = GROUP_FEATS + [c for c in EXTRA_COLS if c not in GROUP_FEATS]


def stage2_frame(d: pl.DataFrame, t_name: np.ndarray, t_addr: np.ndarray, t_is3: np.ndarray) -> pl.DataFrame:
    """d: (s1_idx, t_idx, p, EXTRA_COLS..., rk_rev) for complete S1 groups -> (g, t, STAGE2_COLS)."""
    g = d.select(pl.col("s1_idx").alias("g"), pl.col("t_idx").alias("t"), "p", "rk_rev",
                 *[c for c in EXTRA_COLS])
    ti = g["t"].to_numpy()
    g = g.with_columns(pl.Series("tname", t_name[ti]), pl.Series("taddr", t_addr[ti]), pl.Series("is3", t_is3[ti]))
    gf = group_features(g)
    extra = [c for c in EXTRA_COLS if c not in GROUP_FEATS]
    return g.select("g", "t", *extra).join(gf, on=["g", "t"]).select("g", "t", *STAGE2_COLS)
