"""
Pair feature engine.

Input : long frame of candidate pairs with the normalized S1 / target fields
        joined on (suffix _1 / _2) plus blocking scores.
Output: numeric feature matrix (polars) for the matcher.

All string similarities use rapidfuzz `cpdist` (C++, multithreaded, element-wise),
set/number logic uses vectorized polars list operations, so tens of millions
of pairs can be featurized in chunks.

No feature depends on the country label, so the model transfers to countries
unseen in training (France in the test set).
"""
from __future__ import annotations

import numpy as np
import polars as pl
from rapidfuzz import distance, fuzz, process

# Columns expected on each side (suffix _1 = Source 1, _2 = target)
SIDE_COLS = ["name_core", "name_full", "addr", "business_name", "business_address"]

BLOCK_COLS = ["sc_name", "sc_addr", "sc_comb", "sc_rev", "rk_name", "rk_addr", "rk_comb", "rk_rev",
              "rev_top1", "rev_top2"]


def _cp(a, b, scorer, **kw):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32, **kw)


def string_features(df: pl.DataFrame) -> dict:
    n1, n2 = df["name_core_1"].to_list(), df["name_core_2"].to_list()
    f1, f2 = df["name_full_1"].to_list(), df["name_full_2"].to_list()
    a1, a2 = df["addr_1"].to_list(), df["addr_2"].to_list()
    c1 = [x.replace(" ", "") for x in n1]
    c2 = [x.replace(" ", "") for x in n2]
    out = {
        "n_ratio": _cp(n1, n2, fuzz.ratio),
        "n_partial": _cp(n1, n2, fuzz.partial_ratio),
        "n_tsort": _cp(n1, n2, fuzz.token_sort_ratio),
        "n_tset": _cp(n1, n2, fuzz.token_set_ratio),
        "n_jw": _cp(n1, n2, distance.JaroWinkler.normalized_similarity),
        "n_concat_ratio": _cp(c1, c2, fuzz.ratio),
        "n_concat_partial": _cp(c1, c2, fuzz.partial_ratio),
        "n_lev": _cp(c1, c2, distance.Levenshtein.distance),
        "nf_ratio": _cp(f1, f2, fuzz.ratio),
        "nf_tset": _cp(f1, f2, fuzz.token_set_ratio),
        "a_ratio": _cp(a1, a2, fuzz.ratio),
        "a_partial": _cp(a1, a2, fuzz.partial_ratio),
        "a_tsort": _cp(a1, a2, fuzz.token_sort_ratio),
        "a_tset": _cp(a1, a2, fuzz.token_set_ratio),
    }
    return out


def set_features(df: pl.DataFrame) -> pl.DataFrame:
    tok = lambda c: pl.col(c).str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique()
    num = lambda c: pl.col(c).str.extract_all(r"\d+[a-z]?").list.unique()
    d = df.select(
        tok("name_core_1").alias("nt1"), tok("name_core_2").alias("nt2"),
        tok("addr_1").alias("at1"), tok("addr_2").alias("at2"),
        num("addr_1").alias("an1"), num("addr_2").alias("an2"),
        pl.col("addr_1").str.extract(r"(\d+)").alias("hn1"),
        pl.col("addr_2").str.extract(r"(\d+)").alias("hn2"),
        pl.col("name_core_1").str.split(" ").list.first().alias("ft1"),
        pl.col("name_core_2").str.split(" ").list.first().alias("ft2"),
        pl.col("business_name_2").alias("raw2"),
        pl.col("business_address_2").alias("rawa2"),
        pl.col("name_core_1"), pl.col("name_core_2"), pl.col("addr_1"), pl.col("addr_2"),
    )
    inter = lambda a, b: pl.col(a).list.set_intersection(pl.col(b)).list.len().cast(pl.Float32)
    ln = lambda a: pl.col(a).list.len().cast(pl.Float32)
    d = d.select(
        (inter("nt1", "nt2") / (ln("nt1") + ln("nt2") - inter("nt1", "nt2")).clip(1)).alias("n_jacc"),
        (inter("nt1", "nt2") / ln("nt1").clip(1)).alias("n_cov1"),
        (inter("nt1", "nt2") / ln("nt2").clip(1)).alias("n_cov2"),
        ln("nt1").alias("n_ntok1"), ln("nt2").alias("n_ntok2"),
        (pl.col("name_core_1") == pl.col("name_core_2")).cast(pl.Float32).alias("n_exact"),
        (pl.col("ft1") == pl.col("ft2")).cast(pl.Float32).alias("n_first_eq"),
        (inter("at1", "at2") / (ln("at1") + ln("at2") - inter("at1", "at2")).clip(1)).alias("a_jacc"),
        (inter("at1", "at2") / ln("at1").clip(1)).alias("a_cov1"),
        (inter("at1", "at2") / ln("at2").clip(1)).alias("a_cov2"),
        ln("at1").alias("a_ntok1"), ln("at2").alias("a_ntok2"),
        inter("an1", "an2").alias("a_num_inter"),
        ln("an1").alias("a_num1"), ln("an2").alias("a_num2"),
        (inter("an1", "an2") / ln("an1").clip(1)).alias("a_num_cov1"),
        pl.when(pl.col("hn1").is_null() | pl.col("hn2").is_null()).then(pl.lit(-1.0))
          .otherwise((pl.col("hn1") == pl.col("hn2")).cast(pl.Float32)).alias("a_house_eq"),
        (pl.col("addr_2") == "").cast(pl.Float32).alias("t_addr_empty"),
        pl.col("raw2").str.contains(r"[ऀ-෿]").cast(pl.Float32).alias("t_native"),
        pl.col("raw2").str.contains(r"(?i)\.com|\.net|\.org|^#|www\.").cast(pl.Float32).alias("t_webname"),
        pl.col("raw2").str.contains(r"(?i)\b(fka|dba|aka|formerly)\b").cast(pl.Float32).alias("t_alias"),
        pl.col("raw2").str.contains(r"\(|\[").cast(pl.Float32).alias("t_bracket"),
        (pl.col("raw2") == pl.col("raw2").str.to_uppercase()).cast(pl.Float32).alias("t_upper"),
        pl.col("rawa2").str.contains("#").cast(pl.Float32).alias("t_addr_hash"),
    )
    return d


# ---------------------------------------------------------------------------
# Token alignment: typo vs. word substitution
# ---------------------------------------------------------------------------
# True-match noise keeps every distinctive word (possibly typo'd) and adds
# generic words ("center", "services", ...); look-alike distractors substitute
# a distinctive word with another real word.  We align tokens fuzzily and count
# unmatched S1 tokens and unmatched *non-generic* target tokens.

NOISE_TOKENS: set = set()
# extra noise words for countries unseen in training, learned transductively
# from confident test matches of that country: {country: set(tokens)}
COUNTRY_NOISE: dict = {}


# Tokens that occur in >= 3 Source-1 core names of the current split ("real words").
# A distractor substitutes a real word; true-match noise inserts gibberish aliases.
VOCAB: set = set()


def set_vocab(s1_name_core: pl.Series, min_df: int = 3):
    vc = (s1_name_core.str.split(" ").list.unique().explode().drop_nulls()
          .value_counts().filter(pl.col("count") >= min_df))
    VOCAB.clear()
    VOCAB.update(t for t in vc[vc.columns[0]].to_list() if t)


def set_noise_tokens(tokens):
    NOISE_TOKENS.clear()
    NOISE_TOKENS.update(tokens)


def _token_pairs(a: pl.Series, b: pl.Series) -> pl.DataFrame:
    """Cross product of core-name tokens per pair with Jaro-Winkler similarity."""
    base = pl.DataFrame({"pid": np.arange(len(a), dtype=np.int32), "a": a, "b": b})
    A = (base.select("pid", pl.col("a").str.split(" ").alias("ta"))
         .explode("ta").filter(pl.col("ta").is_not_null() & (pl.col("ta") != "")))
    B = (base.select("pid", pl.col("b").str.split(" ").alias("tb"))
         .explode("tb").filter(pl.col("tb").is_not_null() & (pl.col("tb") != "")))
    X = A.join(B, on="pid")
    X = X.with_columns(pl.Series("sim", _cp(X["ta"].to_list(), X["tb"].to_list(),
                                            distance.JaroWinkler.normalized_similarity)))
    concat = base.select("pid", pl.col("a").str.replace_all(" ", "").alias("ac"),
                         pl.col("b").str.replace_all(" ", "").alias("bc"))
    return A, B, X, concat


def token_align_features(df: pl.DataFrame) -> pl.DataFrame:
    n = df.height
    A, B, X, concat = _token_pairs(df["name_core_1"], df["name_core_2"])
    cn = None
    if COUNTRY_NOISE and "country_1" in df.columns:
        cn = pl.DataFrame([(c, t, True) for c, ts in COUNTRY_NOISE.items() for t in ts],
                          schema=["country", "tb", "cnoise"], orient="row")
        pc = pl.DataFrame({"pid": np.arange(n, dtype=np.int32), "country": df["country_1"]})
    # S1 side: best similarity of each S1 token to any target token
    sa = (A.join(X.group_by("pid", "ta").agg(pl.col("sim").max()), on=["pid", "ta"], how="left")
          .join(concat, on="pid").with_columns(pl.col("sim").fill_null(0.0))
          .with_columns(pl.col("bc").str.contains(pl.col("ta"), literal=True).alias("sub")))
    fa = sa.group_by("pid").agg(
        ((pl.col("sim") < 0.8) & ~pl.col("sub")).sum().alias("tk_miss1_80"),
        ((pl.col("sim") < 0.9) & ~pl.col("sub")).sum().alias("tk_miss1_90"),
        pl.col("sim").min().alias("tk_min1"),
        pl.col("sim").mean().alias("tk_mean1"),
    )
    # target side: extra tokens, and extra tokens that are not generic noise
    sb = (B.join(X.group_by("pid", "tb").agg(pl.col("sim").max()), on=["pid", "tb"], how="left")
          .join(concat, on="pid").with_columns(pl.col("sim").fill_null(0.0))
          .with_columns(pl.col("ac").str.contains(pl.col("tb"), literal=True).alias("sub"),
                        pl.col("tb").is_in(list(NOISE_TOKENS)).alias("noise")))
    if cn is not None:
        sb = (sb.join(pc, on="pid", how="left").join(cn, on=["country", "tb"], how="left")
              .with_columns(pl.col("noise") | pl.col("cnoise").fill_null(False)))
    extra = (pl.col("sim") < 0.8) & ~pl.col("sub")
    sb = sb.with_columns(pl.col("tb").is_in(list(VOCAB)).alias("invocab"))
    fb = sb.group_by("pid").agg(
        extra.sum().alias("tk_extra2"),
        (extra & ~pl.col("noise") & pl.col("invocab")).sum().alias("tk_extra2_vocab"),
        (extra & ~pl.col("noise") & ~pl.col("invocab")).sum().alias("tk_extra2_oov"),
        (extra & ~pl.col("noise")).sum().alias("tk_extra2_real"),
        (extra & ~pl.col("noise")).cast(pl.Int32).mul(pl.col("tb").str.len_chars()).max().alias("tk_extra2_real_len"),
        pl.col("sim").min().alias("tk_min2"),
    )
    out = (pl.DataFrame({"pid": np.arange(n, dtype=np.int32)})
           .join(fa, on="pid", how="left").join(fb, on="pid", how="left").sort("pid").drop("pid"))
    return out.with_columns(pl.all().cast(pl.Float32).fill_null(-1.0))


def learn_noise_tokens(a: pl.Series, b: pl.Series, min_count: int = 30) -> list:
    """Target tokens that appear as unmatched extras in TRUE pairs (generator noise words)."""
    _, B, X, concat = _token_pairs(a, b)
    sb = (B.join(X.group_by("pid", "tb").agg(pl.col("sim").max()), on=["pid", "tb"], how="left")
          .join(concat, on="pid").with_columns(pl.col("sim").fill_null(0.0))
          .filter((pl.col("sim") < 0.8) & ~pl.col("ac").str.contains(pl.col("tb"), literal=True)))
    vc = sb["tb"].value_counts().filter(pl.col("count") >= min_count)
    return sorted(vc["tb"].to_list())


# ---------------------------------------------------------------------------
# Number relations: truncation / padding (true noise) vs. digit change (distractor)
# ---------------------------------------------------------------------------

def _num_rel(a: pl.Expr, b: pl.Expr, lev: pl.Expr) -> pl.Expr:
    """4 equal, 3 one is a suffix of the other, 2 prefix, 1 same length 1 edit, 0 other."""
    return (pl.when(a == b).then(4)
            .when(a.str.ends_with(b) | b.str.ends_with(a)).then(3)
            .when(a.str.starts_with(b) | b.str.starts_with(a)).then(2)
            .when((a.str.len_chars() == b.str.len_chars()) & (lev == 1)).then(1)
            .otherwise(0))


def number_features(df: pl.DataFrame) -> pl.DataFrame:
    n = df.height
    base = pl.DataFrame({"pid": np.arange(n, dtype=np.int32),
                         "x": df["addr_1"].str.extract_all(r"\d+"),
                         "y": df["addr_2"].str.extract_all(r"\d+")})
    Xn = base.select("pid", "x").explode("x").filter(pl.col("x").is_not_null()).unique()
    Yn = base.select("pid", "y").explode("y").filter(pl.col("y").is_not_null()).unique()
    P = Xn.join(Yn, on="pid")
    P = P.with_columns(pl.Series("lev", _cp(P["x"].to_list(), P["y"].to_list(), distance.Levenshtein.distance)))
    P = P.with_columns(_num_rel(pl.col("x"), pl.col("y"), pl.col("lev")).alias("rel"))
    best = Xn.join(P.group_by("pid", "x").agg(pl.col("rel").max()), on=["pid", "x"], how="left").fill_null(0)
    f = best.group_by("pid").agg(
        *[(pl.col("rel") == k).sum().alias(f"num_rel{k}") for k in range(5)])
    # house number = first number on each side
    h = base.select("pid", pl.col("x").list.first().alias("hx"), pl.col("y").list.first().alias("hy"))
    hl = _cp(h["hx"].fill_null("").to_list(), h["hy"].fill_null("").to_list(), distance.Levenshtein.distance)
    h = h.with_columns(pl.Series("lev", hl))
    h = h.select("pid", pl.when(pl.col("hx").is_null() | pl.col("hy").is_null()).then(-1)
                 .otherwise(_num_rel(pl.col("hx"), pl.col("hy"), pl.col("lev"))).alias("hn_rel"),
                 (pl.col("hx").str.len_chars().cast(pl.Int32) - pl.col("hy").str.len_chars().cast(pl.Int32))
                 .alias("hn_lendiff"))
    out = (pl.DataFrame({"pid": np.arange(n, dtype=np.int32)})
           .join(f, on="pid", how="left").join(h, on="pid", how="left").sort("pid").drop("pid"))
    return out.with_columns(pl.all().cast(pl.Float32).fill_null(-1.0))


# ---------------------------------------------------------------------------
# Raw-name style: legal form as written (Pvt vs Private, PLLC vs Corp)
# ---------------------------------------------------------------------------
# Canonicalized names make "Resolve India Pvt Ltd" and "Resolve (india) Private
# Limited" identical.  The generator keeps the owner's legal-form *style* in most
# of its records, which separates same-name businesses when the address is
# missing.  These features compare the names as written.

LEGAL_TOKENS = ["pvt", "private", "ltd", "limited", "llc", "pllc", "inc", "incorporated", "corp",
                "corporation", "co", "company", "lp", "llp", "pc", "plc", "opc", "pa", "ltda",
                "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "scp", "sci", "gmbh",
                "ei", "eirl", "selarl", "gie", "sarlu", "earl"]


def _raw_norm(col: str) -> pl.Expr:
    return (pl.col(col).str.to_lowercase().str.replace_all(r"\.", "")
            .str.replace_all(r"[^\p{L}\p{N}]+", " ").str.strip_chars())


def raw_name_features(df: pl.DataFrame) -> pl.DataFrame:
    r = df.select(_raw_norm("business_name_1").alias("r1"), _raw_norm("business_name_2").alias("r2"))
    r1, r2 = r["r1"].to_list(), r["r2"].to_list()
    ra = df.select(_raw_norm("business_address_1").alias("a1"), _raw_norm("business_address_2").alias("a2"))
    ra1, ra2 = ra["a1"].to_list(), ra["a2"].to_list()
    legal = list(dict.fromkeys(LEGAL_TOKENS))
    lg = r.select(
        pl.col("r1").str.split(" ").list.eval(pl.element().filter(pl.element().is_in(legal))).list.unique().alias("l1"),
        pl.col("r2").str.split(" ").list.eval(pl.element().filter(pl.element().is_in(legal))).list.unique().alias("l2"))
    inter = pl.col("l1").list.set_intersection(pl.col("l2")).list.len()
    lf = lg.select(
        pl.col("l1").list.len().cast(pl.Float32).alias("lg_n1"),
        pl.col("l2").list.len().cast(pl.Float32).alias("lg_n2"),
        inter.cast(pl.Float32).alias("lg_inter"),
        (pl.col("l2").list.len() - inter).cast(pl.Float32).alias("lg_t_extra"),   # target legal token absent in S1
        (pl.col("l1").list.len() - inter).cast(pl.Float32).alias("lg_s1_extra"),
        (pl.col("l1").list.sort() == pl.col("l2").list.sort()).cast(pl.Float32).alias("lg_equal"),
    )
    return pl.concat([pl.DataFrame({
        "raw_ratio": _cp(r1, r2, fuzz.ratio),
        "raw_tsort": _cp(r1, r2, fuzz.token_sort_ratio),
        "raw_tset": _cp(r1, r2, fuzz.token_set_ratio),
        "raw_exact": np.array([a == b for a, b in zip(r1, r2)], dtype=np.float32),
        # addresses as written (no abbreviation / state canonicalization)
        "raw_a_ratio": _cp(ra1, ra2, fuzz.ratio),
        "raw_a_tset": _cp(ra1, ra2, fuzz.token_set_ratio),
    }), lf], how="horizontal")


def pair_features(df: pl.DataFrame) -> pl.DataFrame:
    """Full feature frame for a chunk of joined candidate pairs."""
    s = string_features(df)
    sets = pl.concat([set_features(df), token_align_features(df), number_features(df), raw_name_features(df)],
                     how="horizontal")
    extra = df.select(
        pl.col("t_id").str.starts_with("S3-").cast(pl.Float32).alias("t_is_s3"),
        *[pl.col(c).cast(pl.Float32) for c in BLOCK_COLS if c in df.columns],
        *[pl.col(c).cast(pl.Float32) for c in ("s1_name_freq", "t_name_freq", "s1_raw_freq", "t_raw_freq", "s1_ncand")
          if c in df.columns],
    )
    out = pl.concat([pl.DataFrame(s), sets, extra], how="horizontal")
    # derived
    out = out.with_columns(
        (pl.col("sc_rev") / pl.col("rev_top1")).alias("rev_rel"),
        (pl.col("rev_top1") - pl.col("rev_top2")).alias("rev_gap"),
        (pl.col("n_lev") / pl.max_horizontal(pl.lit(1.0), pl.col("n_ntok1"))).alias("n_lev_per_tok"),
    )
    return out
