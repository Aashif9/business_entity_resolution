# Step 2 of the rescue: stage 2 for India owners that received a new pair with stage-1 p >= 0.05, then the final rule;
# writes work/rescue/partial/matching_results_partial_rescue.tsv and candidate_pairs_partial_rescue.tsv.
"""Fast partial rescue: stage 2 only for India owners that received a new pair with stage-1 p >= 0.05;
every other owner keeps its baseline score.  Writes a NEW matching file; baseline untouched."""
import os, sys, time
os.environ["POLARS_MAX_THREADS"] = "20"
from pathlib import Path
import numpy as np
import polars as pl
import lightgbm as lgb
T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:.0f}s] {m}", flush=True)
P = Path(__file__).resolve().parents[4]; w = P / "work"; exp = w / "rescue"; run = exp / "test_India"
out = exp / "partial"; out.mkdir(exist_ok=True)
sys.path.insert(0, str(P / "code/business_entity_resolution"))
from src.stage2 import stage2_frame, STAGE2_COLS
from src.model import LeaderboardRule
arts = P / "code/business_entity_resolution/artifacts"

new = pl.read_parquet(run / "new_stage1.parquet")
R = new.filter(pl.col("p") >= 0.05)["s1_idx"].unique()
log(f"owners with a promising new pair: {len(R):,}")
new = new.filter(pl.col("s1_idx").is_in(R.implode()))
old = pl.scan_parquet(w / "test_scores_s1/*.parquet").filter(pl.col("s1_idx").is_in(R.implode())).collect()
full = pl.concat([old, new.select(old.columns)], how="vertical_relaxed"); del old, new
log(f"pairs for stage 2: {full.height:,}")
tg = pl.concat([pl.read_parquet(w / f"test_s{i}_norm.parquet", columns=["entity_id", "name_core", "addr", "country", "business_name", "business_address"]) for i in (2, 3)]).with_row_index("t_idx").with_columns(pl.col("t_idx").cast(pl.Int32))
tn, ta = tg["name_core"].to_numpy(), tg["addr"].to_numpy(); is3 = tg["entity_id"].str.starts_with("S3-").to_numpy()
b2 = lgb.Booster(model_file=str(arts / "lgb_stage2.txt"))
res = []
for start in range(0, len(R), 20000):
    d = full.filter(pl.col("s1_idx").is_in(R.slice(start, 20000).implode()))
    f = stage2_frame(d, tn, ta, is3)
    res.append(f.select(pl.col("g").alias("s1_idx"), pl.col("t").alias("t_idx")).with_columns(pl.Series("p", b2.predict(f.select(STAGE2_COLS).to_numpy(), num_threads=20).astype(np.float32))))
    log(f"stage2 {min(start + 20000, len(R)):,}/{len(R):,}")
changed = pl.concat(res); changed.write_parquet(out / "affected_scores.parquet")

s1 = pl.read_parquet(w / "test_s1_norm.parquet", columns=["entity_id", "country"]).with_row_index("s1_idx").with_columns(pl.col("s1_idx").cast(pl.Int32))
ids = s1.filter(pl.col("country") == "India")["s1_idx"]
base = pl.scan_parquet(w / "test_final_scores.parquet").filter(pl.col("s1_idx").is_in(ids.implode())).collect()
revised = pl.concat([base.filter(~pl.col("s1_idx").is_in(R.implode())), changed], how="vertical_relaxed")
tgi = tg.filter(pl.col("country") == "India")
noaddr = tgi.select("t_idx", (pl.col("addr") == "").alias("noaddr"))
rule = LeaderboardRule()


def select(sc):
    sel = rule.apply(sc, noaddr)
    sel = sel.join(sc, on=["s1_idx", "t_idx"]).sort(["p", "s1_idx"], descending=[True, False]).unique("t_idx", keep="first", maintain_order=True)
    sel = sel.join(tgi.select("t_idx", pl.col("entity_id").str.starts_with("S3-").alias("is3")), on="t_idx")
    sel = sel.with_columns(pl.col("p").rank("ordinal", descending=True).over("s1_idx", "is3").alias("r"))
    sel = sel.filter(pl.col("r") <= pl.when(pl.col("is3")).then(6).otherwise(5)).select("s1_idx", "t_idx")
    key = pl.concat_str([pl.col("business_name").fill_null("").str.to_lowercase().str.strip_chars(), pl.col("business_address").fill_null("").str.to_lowercase().str.strip_chars()], separator="|")
    tk = tgi.select("t_idx", key.alias("key"))
    anchors = sel.join(tk, on="t_idx").select("s1_idx", "key").unique()
    best = sc.sort(["p", "s1_idx"], descending=[True, False]).unique("t_idx", keep="first", maintain_order=True)
    dup = best.join(sel.select("t_idx"), on="t_idx", how="anti").join(tk, on="t_idx").join(anchors, on=["s1_idx", "key"]).select("s1_idx", "t_idx")
    return pl.concat([sel, dup]).unique()


original = pl.read_csv(P / "output/matching_results.tsv", separator="\t", infer_schema=False, quote_char=None).fill_null("")
orig = (original.join(s1.filter(pl.col("country") == "India").select(pl.col("entity_id").alias("source1_entity_id"), "s1_idx"), on="source1_entity_id")
        .select("s1_idx", pl.col("matched_entity_ids").str.split(",").alias("entity_id")).explode("entity_id").filter(pl.col("entity_id") != "")
        .join(tgi.select("entity_id", "t_idx"), on="entity_id").select("s1_idx", "t_idx"))
calc = select(base)
mm = calc.join(orig, on=["s1_idx", "t_idx"], how="anti").height + orig.join(calc, on=["s1_idx", "t_idx"], how="anti").height
assert mm == 0, f"baseline mismatch {mm}"
sel = select(revised)
added = sel.join(orig, on=["s1_idx", "t_idx"], how="anti"); removed = orig.join(sel, on=["s1_idx", "t_idx"], how="anti")
log(f"baseline reproduced; India links +{added.height:,} / -{removed.height:,}")
mapped = sel.join(tgi.select("t_idx", "entity_id"), on="t_idx").group_by("s1_idx").agg(pl.col("entity_id").unique().sort().str.join(",").alias("new_matches"))
repl = s1.filter(pl.col("country") == "India").join(mapped, on="s1_idx", how="left").select(pl.col("entity_id").alias("source1_entity_id"), pl.col("new_matches").fill_null(""))
output = original.join(repl, on="source1_entity_id", how="left").with_columns(pl.coalesce("new_matches", "matched_entity_ids").alias("matched_entity_ids")).select("source1_entity_id", "matched_entity_ids")
output.write_csv(out / "matching_results_partial_rescue.tsv", separator="\t", quote_style="never")
log(f"written {out / 'matching_results_partial_rescue.tsv'}  rows {output.height:,}")
# candidate file: baseline candidates + the new pairs that were actually selected
newsel = added.join(tgi.select("t_idx", pl.col("entity_id").alias("t_id")), on="t_idx").join(s1.select("s1_idx", pl.col("entity_id").alias("s1_id")), on="s1_idx")
extra = dict(newsel.group_by("s1_id").agg(pl.col("t_id")).iter_rows())
with (P / "output/candidate_pairs.tsv").open(encoding="utf-8") as fi, (out / "candidate_pairs_partial_rescue.tsv").open("w", encoding="utf-8", newline="") as fo:
    fo.write(next(fi))
    for line in fi:
        sid, _, tail = line.rstrip("\r\n").partition("\t")
        if sid in extra:
            t = set(tail.split(",")) if tail else set(); t.update(extra.pop(sid)); fo.write(sid + "\t" + ",".join(sorted(t)) + "\n")
        else:
            fo.write(line)
log("candidate file written")
