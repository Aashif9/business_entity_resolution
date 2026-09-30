"""
Scalable high-recall blocking.

For every Source-1 record we retrieve the top-K Source-2/3 records of the same
country under three sparse TF-IDF cosine "channels":

  * name     : core-name word tokens + char 4-grams of the space-less core name
               (robust to typos, word reordering, concatenated domain names)
  * address  : address tokens + adjacent-token bigrams (house number + street)
  * combined : name + address features together

Retrieval is an inverted-index accumulation written in numba: posting lists
of features with document frequency above `max_df` are dropped (they carry
almost no signal and dominate cost), each thread owns a dense score
accumulator, and top-K is taken with a partition.  This scales linearly and
handles millions of records per country in minutes on a laptop.

The union of the channel top-Ks is the candidate set fed to the matcher
(after the light cap in `cap_candidates`).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numba as nb
import numpy as np
import polars as pl

from .normalize import ADDR_STOPWORDS


# ---------------------------------------------------------------------------
# Feature generation (vectorized, polars)
# ---------------------------------------------------------------------------

def _explode_tokens(df: pl.DataFrame, col: str, prefix: str, stop=None) -> pl.DataFrame:
    e = pl.col(col).str.split(" ")
    out = df.select(pl.col("rid"), e.alias("f")).explode("f").filter(pl.col("f") != "")
    if stop:
        out = out.filter(~pl.col("f").is_in(list(stop)))
    return out.select("rid", (pl.lit(prefix) + pl.col("f")).alias("f"))


def _explode_bigrams(df: pl.DataFrame, col: str, prefix: str, stop=None) -> pl.DataFrame:
    toks = pl.col(col).str.split(" ")
    if stop:
        toks = toks.list.eval(pl.element().filter(~pl.element().is_in(list(stop)) & (pl.element() != "")))
    t = df.select(pl.col("rid"), toks.alias("t"))
    t = t.with_columns(pl.col("t").list.slice(1).alias("u"), pl.col("t").list.slice(0, (pl.col("t").list.len().cast(pl.Int64) - 1).clip(0, None)).alias("t"))
    t = t.explode(["t", "u"]).filter(pl.col("u").is_not_null())
    return t.select("rid", (pl.lit(prefix) + pl.col("t") + "_" + pl.col("u")).alias("f"))


def _explode_ngrams(df: pl.DataFrame, col: str, prefix: str, n: int) -> pl.DataFrame:
    s = df.select(pl.col("rid"), pl.col(col).str.replace_all(" ", "").alias("s"))
    s = s.filter(pl.col("s").str.len_chars() > 0)
    s = s.with_columns(pl.int_ranges(0, (pl.col("s").str.len_chars().cast(pl.Int64) - n + 1).clip(1, None)).alias("o"))
    s = s.explode("o")
    return s.select("rid", (pl.lit(prefix) + pl.col("s").str.slice(pl.col("o"), n)).alias("f"))


def record_features(df: pl.DataFrame, channel: str) -> pl.DataFrame:
    """(rid, f) rows for one channel; df must have rid, name_core, addr."""
    if channel == "name":
        parts = [_explode_tokens(df, "name_core", "w:"),
                 _explode_ngrams(df, "name_core", "g:", 4)]
    elif channel == "addr":
        parts = [_explode_tokens(df, "addr", "a:", ADDR_STOPWORDS),
                 _explode_bigrams(df, "addr", "b:", ADDR_STOPWORDS)]
    else:
        raise ValueError(channel)
    out = pl.concat(parts).select("rid", (pl.col("f").hash(seed=17) & 0xFFFFFFFF).cast(pl.UInt32).alias("h"))
    return out.unique(["rid", "h"])


def features_chunked(df: pl.DataFrame, channel: str, chunk: int = 500_000) -> pl.DataFrame:
    """record_features over row chunks to bound peak memory."""
    parts = [record_features(df.slice(i, chunk), channel) for i in range(0, df.height, chunk)]
    return pl.concat(parts).with_columns(pl.col("rid").cast(pl.Int32))


# ---------------------------------------------------------------------------
# Index construction
# ---------------------------------------------------------------------------

@dataclass
class ChannelIndex:
    vocab: np.ndarray          # sorted feature hashes (only kept features)
    idf: np.ndarray            # idf per kept feature
    indptr: np.ndarray         # CSR over features -> targets
    tgt: np.ndarray
    w: np.ndarray              # normalized target weights


@nb.njit(cache=True)
def _counting_order(keys, n_keys):
    """Stable counting-sort permutation + CSR indptr for integer keys in [0, n_keys)."""
    counts = np.zeros(n_keys + 1, np.int64)
    for k in keys:
        counts[k + 1] += 1
    for i in range(n_keys):
        counts[i + 1] += counts[i]
    indptr = counts.copy()
    order = np.empty(len(keys), np.int64)
    for i in range(len(keys)):
        k = keys[i]
        order[counts[k]] = i
        counts[k] += 1
    return order, indptr


def build_index(target_feats: pl.DataFrame, n_targets: int, max_df: int) -> ChannelIndex:
    h = target_feats["h"].to_numpy()
    rid = target_feats["rid"].to_numpy().astype(np.int32)
    vocab, inv, df = np.unique(h, return_inverse=True, return_counts=True)
    idf_all = np.log((n_targets + 1) / (df + 1)).astype(np.float32) + 1.0
    w_all = idf_all[inv]
    # L2 norm per target over *all* features (before df pruning)
    norm = np.sqrt(np.bincount(rid, weights=w_all.astype(np.float64) ** 2, minlength=n_targets)).astype(np.float32)
    norm[norm == 0] = 1.0
    keep = df[inv] <= max_df
    inv, rid, w_all = inv[keep], rid[keep], w_all[keep] / norm[rid[keep]]
    kept_feats = np.flatnonzero(df <= max_df)
    remap = np.full(len(vocab), -1, np.int64)
    remap[kept_feats] = np.arange(len(kept_feats))
    fid = remap[inv]
    order, indptr = _counting_order(fid, len(kept_feats))
    return ChannelIndex(vocab=vocab[kept_feats], idf=idf_all[kept_feats], indptr=indptr,
                        tgt=rid[order], w=w_all[order].astype(np.float32))


@nb.njit(cache=True)
def _trim_rows(indptr, fid, w, max_feats):
    """Keep at most max_feats highest-weight (rarest) features per query row."""
    n_q = len(indptr) - 1
    new_ptr = np.zeros(n_q + 1, np.int64)
    out_f = np.empty(len(fid), np.int64)
    out_w = np.empty(len(fid), np.float32)
    m = 0
    for q in range(n_q):
        lo, hi = indptr[q], indptr[q + 1]
        if hi - lo <= max_feats:
            for j in range(lo, hi):
                out_f[m] = fid[j]
                out_w[m] = w[j]
                m += 1
        else:
            order = np.argsort(-w[lo:hi])[:max_feats]
            for j in order:
                out_f[m] = fid[lo + j]
                out_w[m] = w[lo + j]
                m += 1
        new_ptr[q + 1] = m
    return new_ptr, out_f[:m], out_w[:m]


def query_matrix(q_feats: pl.DataFrame, n_q: int, idx: ChannelIndex, max_feats: int = 0,
                 weight: float = 1.0):
    """CSR (query -> kept feature ids, weights). Query weights use target idf.
    max_feats > 0 keeps only the rarest features per query (search budget)."""
    h = q_feats["h"].to_numpy()
    rid = q_feats["rid"].to_numpy().astype(np.int64)
    pos = np.searchsorted(idx.vocab, h)
    pos[pos >= len(idx.vocab)] = 0
    found = idx.vocab[pos] == h
    # unseen / pruned features still count in the query norm with max idf
    w = np.where(found, idx.idf[pos], idx.idf.max() if len(idx.idf) else 1.0).astype(np.float32)
    norm = np.sqrt(np.bincount(rid, weights=w.astype(np.float64) ** 2, minlength=n_q)).astype(np.float32)
    norm[norm == 0] = 1.0
    rid, fid, w = rid[found], pos[found], w[found] / norm[rid[found]] * weight
    order, indptr = _counting_order(rid, n_q)
    fid, w = fid[order].astype(np.int64), w[order].astype(np.float32)
    if max_feats > 0:
        indptr, fid, w = _trim_rows(indptr, fid, w, max_feats)
    return indptr, fid, w


# ---------------------------------------------------------------------------
# numba top-K search
# ---------------------------------------------------------------------------

@nb.njit(parallel=True, cache=True)
def _topk_search(q_indptr, q_fid, q_w, q_off, p_indptr, p_tgt, p_w, n_targets, k, n_blocks):
    nq = len(q_indptr) - 1
    out_idx = np.full((nq, k), -1, np.int32)
    out_sc = np.zeros((nq, k), np.float32)
    for b in nb.prange(n_blocks):
        acc = np.zeros(n_targets, np.float32)
        touched = np.empty(n_targets, np.int32)
        for q in range(b, nq, n_blocks):
            nt = 0
            for j in range(q_indptr[q], q_indptr[q + 1]):
                f = q_fid[j] + q_off[j]
                wq = q_w[j]
                for p in range(p_indptr[f], p_indptr[f + 1]):
                    t = p_tgt[p]
                    if acc[t] == 0.0:
                        touched[nt] = t
                        nt += 1
                    acc[t] += wq * p_w[p]
            if nt == 0:
                continue
            if nt <= k:
                sc = np.empty(nt, np.float32)
                for i in range(nt):
                    sc[i] = acc[touched[i]]
                order = np.argsort(-sc)
                for i in range(nt):
                    out_idx[q, i] = touched[order[i]]
                    out_sc[q, i] = sc[order[i]]
            else:
                sc = np.empty(nt, np.float32)
                for i in range(nt):
                    sc[i] = acc[touched[i]]
                kth = np.partition(-sc, k - 1)[k - 1]
                sel = np.empty(k, np.int32)
                ssc = np.empty(k, np.float32)
                m = 0
                # strictly greater first, then ties
                for i in range(nt):
                    if -sc[i] < kth and m < k:
                        sel[m] = touched[i]
                        ssc[m] = sc[i]
                        m += 1
                for i in range(nt):
                    if -sc[i] == kth and m < k:
                        sel[m] = touched[i]
                        ssc[m] = sc[i]
                        m += 1
                order = np.argsort(-ssc)
                for i in range(k):
                    out_idx[q, i] = sel[order[i]]
                    out_sc[q, i] = ssc[order[i]]
            for i in range(nt):
                acc[touched[i]] = 0.0
    return out_idx, out_sc


def _search(qs: List[Tuple], idxs: List[ChannelIndex], n_targets: int, k: int):
    """Search with the concatenation of one or more channels (scores add)."""
    offs = np.cumsum([0] + [len(ix.indptr) - 1 for ix in idxs])[:-1]
    ptr_off = np.cumsum([0] + [len(ix.tgt) for ix in idxs])[:-1]
    p_indptr = np.concatenate([ix.indptr[:-1] + po for ix, po in zip(idxs, ptr_off)] +
                              [np.array([sum(len(ix.tgt) for ix in idxs)], np.int64)])
    p_tgt = np.concatenate([ix.tgt for ix in idxs])
    p_w = np.concatenate([ix.w for ix in idxs])
    n_q = len(qs[0][0]) - 1
    # merge query CSRs
    counts = sum(np.diff(q[0]) for q in qs)
    indptr = np.zeros(n_q + 1, np.int64)
    np.cumsum(counts, out=indptr[1:])
    fid = np.empty(indptr[-1], np.int64)
    w = np.empty(indptr[-1], np.float32)
    off = np.empty(indptr[-1], np.int64)
    fill = indptr[:-1].copy()
    for (qi, qf, qw), o in zip(qs, offs):
        c = np.diff(qi)
        rows = np.repeat(np.arange(n_q), c)
        dst = fill[rows] + (np.arange(len(qf)) - qi[rows])
        fid[dst] = qf
        w[dst] = qw
        off[dst] = o
        fill += c
    n_blocks = max(nb.get_num_threads(), 1)
    return _topk_search(indptr, fid, w, off, p_indptr, p_tgt, p_w, n_targets, k, n_blocks)


def _to_long(ii: np.ndarray, ss: np.ndarray, rows: np.ndarray, name: str, reverse: bool) -> pl.DataFrame:
    k = ii.shape[1]
    q = np.repeat(rows.astype(np.int32), k)
    other = ii.ravel()
    keep = other >= 0
    s1_idx, t_idx = (other, q) if reverse else (q, other)
    return pl.DataFrame({
        "s1_idx": s1_idx[keep], "t_idx": t_idx[keep],
        f"sc_{name}": ss.ravel()[keep],
        f"rk_{name}": np.tile(np.arange(1, k + 1, dtype=np.int16), len(rows))[keep],
    })


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class BlockingConfig:
    k_name: int = 30
    k_addr: int = 30
    k_comb: int = 40
    k_rev: int = 20             # reverse search depth (ranks 6-20 kept only for address-less targets)
    max_df_name: int = 30000
    max_df_addr: int = 30000
    max_df_rev: int = 10000
    max_feats_fwd: int = 24     # rarest features used per query per channel
    max_feats_rev: int = 12
    # final candidate cap (the set written to candidate_pairs.tsv and scored)
    cap_rev: int = 5
    cap_rev_noaddr: int = 20    # adaptive K: address-less targets have only the name to go on
    cap_comb: int = 20
    cap_single: int = 10


def apply_cap(cand: pl.DataFrame, cfg: BlockingConfig) -> pl.DataFrame:
    big = 32000
    r = lambda c: pl.col(c).fill_null(big) if c in cand.columns else pl.lit(big)
    noaddr = pl.col("t_noaddr").fill_null(False) if "t_noaddr" in cand.columns else pl.lit(False)
    return cand.filter((r("rk_rev") <= cfg.cap_rev) | (noaddr & (r("rk_rev") <= cfg.cap_rev_noaddr)) |
                       (r("rk_comb") <= cfg.cap_comb) |
                       (r("rk_name") <= cfg.cap_single) | (r("rk_addr") <= cfg.cap_single))


def block_country(s1: pl.DataFrame, tg: pl.DataFrame, cfg: BlockingConfig,
                  query_mask: np.ndarray | None = None, log=print) -> pl.DataFrame:
    """
    Candidate generation for one country.

    s1, tg      : normalized frames (all S1 / all targets of the country).
    query_mask  : optional bool mask over s1 rows to restrict output to
                  (used for validation on a sample; the reverse channel still
                  uses the full S1 index so competition between S1s is real).

    Returns a long frame (s1_idx, t_idx, sc_*/rk_* per channel) with row
    indices into s1 / tg.  Channels: name, addr, comb (S1 -> targets) and
    rev (targets -> S1, combined features).
    """
    t0 = time.time()
    s1 = s1.select(pl.int_range(0, pl.len()).alias("rid"), "name_core", "addr")
    tg = tg.select(pl.int_range(0, pl.len()).alias("rid"), "name_core", "addr")
    n_t, n_s = tg.height, s1.height
    q_rows = np.arange(n_s) if query_mask is None else np.flatnonzero(query_mask)
    s1q = s1[q_rows].with_columns(pl.int_range(0, pl.len()).alias("rid"))

    frames = []
    # ---- forward: S1 -> targets --------------------------------------------
    idx, qs = {}, {}
    for ch, mdf in (("name", cfg.max_df_name), ("addr", cfg.max_df_addr)):
        idx[ch] = build_index(features_chunked(tg, ch), n_t, mdf)
        qs[ch] = query_matrix(features_chunked(s1q, ch), len(q_rows), idx[ch], cfg.max_feats_fwd)
    log(f"  [block] target index {n_t:,} rows: {time.time() - t0:.0f}s")
    for name, chans, k in (("name", ["name"], cfg.k_name), ("addr", ["addr"], cfg.k_addr),
                           ("comb", ["name", "addr"], cfg.k_comb)):
        t1 = time.time()
        ii, ss = _search([qs[c] for c in chans], [idx[c] for c in chans], n_t, k)
        frames.append(_to_long(ii, ss, q_rows, name, reverse=False))
        log(f"  [block] {name}: {len(q_rows):,} queries in {time.time() - t1:.0f}s")
    del idx, qs

    # ---- reverse: targets -> S1 (every target looks for its best S1s) ------
    if cfg.k_rev > 0:
        t1 = time.time()
        idx, qs = {}, {}
        for ch in ("name", "addr"):
            idx[ch] = build_index(features_chunked(s1, ch), n_s, cfg.max_df_rev)
            qs[ch] = query_matrix(features_chunked(tg, ch), n_t, idx[ch], cfg.max_feats_rev)
        ii, ss = _search([qs["name"], qs["addr"]], [idx["name"], idx["addr"]], n_s, cfg.k_rev)
        del idx, qs
        # each target's best / second-best S1 score: context for "is this S1 the
        # target's most likely owner" (targets belong to at most one S1)
        rev_top = pl.DataFrame({"t_idx": np.arange(n_t, dtype=np.int32),
                                "rev_top1": ss[:, 0],
                                "rev_top2": ss[:, 1] if ss.shape[1] > 1 else np.zeros(n_t, np.float32)})
        rev = _to_long(ii, ss, np.arange(n_t), "rev", reverse=True)
        t_noaddr = (tg["addr"] == "").to_numpy()
        rev = rev.with_columns(pl.Series("t_noaddr", t_noaddr[rev["t_idx"].to_numpy()]))
        rev = rev.filter((pl.col("rk_rev") <= cfg.cap_rev) | (pl.col("t_noaddr") & (pl.col("rk_rev") <= cfg.cap_rev_noaddr)))
        if query_mask is not None:
            rev = rev.filter(pl.Series(query_mask[rev["s1_idx"].to_numpy()]))
        frames.append(rev)
        log(f"  [block] rev: {n_t:,} queries in {time.time() - t1:.0f}s")

    out = frames[0]
    for f in frames[1:]:
        out = out.join(f, on=["s1_idx", "t_idx"], how="full", coalesce=True)
    if cfg.k_rev > 0:
        out = out.join(rev_top, on="t_idx", how="left")
    return apply_cap(out, cfg)
