"""Control for the India validation: recompute stage 1 + stage 2 for the SAME touched owners WITHOUT the
new candidates, with the same pred() logic.  If this alone changes the score, the rescue gain is a
recomputation artefact; if it is ~0, the gain comes from the added candidates."""
import json, os, sys
os.environ.setdefault("POLARS_MAX_THREADS", "4")
from pathlib import Path
import numpy as np
import polars as pl
import lightgbm as lgb

P = Path(__file__).resolve().parents[4]; w = P / "work"; out = w / "rescue" / "val_India"
sys.path.insert(0, str(P / "code/business_entity_resolution"))
from src.pipeline import KEEP_FOR_STAGE2
from src.stage2 import stage2_frame, STAGE2_COLS
from src.model import LeaderboardRule, macro_f05

arts = w / "artifacts_v9"
s1 = pl.read_parquet(w / "train_s1_norm.parquet", columns=["entity_id"])
s1 = s1.filter(s1["entity_id"].hash(seed=42) % 10000 >= 1900).with_row_index("_g").with_columns(pl.col("_g").cast(pl.Int32))
ids = s1["entity_id"].sample(180000, seed=42, shuffle=True)[150000:]
tg = pl.concat([pl.read_parquet(w / f"train_s{i}_norm.parquet", columns=["entity_id", "name_core", "addr"]) for i in (2, 3)])
tg = tg.with_row_index("_g").with_columns(pl.col("_g").cast(pl.Int32))
gt = (pl.read_csv(P / "dataset/train/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema=False)
      .select(pl.col("source1_entity_id").alias("s1_id"), pl.col("matched_entity_ids").str.split(",").alias("t_id"))
      .explode("t_id").filter(pl.col("t_id").is_not_null() & (pl.col("t_id") != "")))
vt = gt.filter(pl.col("s1_id").is_in(ids.implode()))
touched = pl.read_parquet(out / "affected_scores.parquet", columns=["s1_idx"])["s1_idx"].unique()
b1 = lgb.Booster(model_file=str(arts / "lgb_matcher.txt"))
oldf = pl.scan_parquet(w / "train_feat_150000_30000_d19.parquet").filter(pl.col("s1_idx").is_in(touched.implode())).collect()
p = b1.predict(oldf.select(b1.feature_name()).to_numpy().astype(np.float32), num_threads=4)
oldf = oldf.select("s1_idx", "t_idx", *KEEP_FOR_STAGE2).with_columns(pl.Series("p", p.astype(np.float32)))
b2 = lgb.Booster(model_file=str(arts / "lgb_stage2.txt"))
tn, ta = tg["name_core"].to_numpy(), tg["addr"].to_numpy(); is3 = tg["entity_id"].str.starts_with("S3-").to_numpy()
ctrl = []
for start in range(0, len(touched), 5000):
    d = oldf.filter(pl.col("s1_idx").is_in(touched.slice(start, 5000).implode()))
    f = stage2_frame(d, tn, ta, is3)
    ctrl.append(f.select(pl.col("g").alias("s1_idx"), pl.col("t").alias("t_idx")).with_columns(pl.Series("p", b2.predict(f.select(STAGE2_COLS).to_numpy(), num_threads=4).astype(np.float32))))
ctrl = pl.concat(ctrl)
baseline = pl.read_parquet(w / "val_scored_v9.parquet")
base = (baseline.join(s1.select(pl.col("entity_id").alias("s1_id"), pl.col("_g").alias("s1_idx")), on="s1_id")
        .join(tg.select(pl.col("entity_id").alias("t_id"), pl.col("_g").alias("t_idx")), on="t_id").select("s1_idx", "t_idx", "p"))
control = pl.concat([base.filter(~pl.col("s1_idx").is_in(touched.implode())), ctrl], how="vertical_relaxed")
na = tg.select(pl.col("_g").alias("t_idx"), (pl.col("addr") == "").alias("noaddr"))
rule = LeaderboardRule()


def pred(sc):
    z = rule.apply(sc, na)
    z = z.join(sc, on=["s1_idx", "t_idx"]).sort(["p", "s1_idx"], descending=[True, False]).unique("t_idx", keep="first", maintain_order=True)
    z = z.with_columns(pl.Series("src3", tg["entity_id"].gather(z["t_idx"]).str.starts_with("S3-")))
    z = z.with_columns(pl.col("p").rank("ordinal", descending=True).over("s1_idx", "src3").alias("r"))
    z = z.filter(pl.col("r") <= pl.when(pl.col("src3")).then(6).otherwise(5))
    return z.select(pl.Series("s1_id", s1["entity_id"].gather(z["s1_idx"])), pl.Series("t_id", tg["entity_id"].gather(z["t_idx"])))


summ = json.loads((out / "summary.json").read_text())
res = {}
for label, vids in [("all", ids), ("half_A", ids[:15000]), ("half_B", ids[15000:])]:
    res[label] = {"baseline": summ["metrics"][label]["baseline"], "control_no_new_candidates": macro_f05(pred(control), vt, vids),
                  "revised_with_rescue": summ["metrics"][label]["revised"]}
    r = res[label]
    r["control_minus_baseline"] = r["control_no_new_candidates"] - r["baseline"]
    r["rescue_minus_control"] = r["revised_with_rescue"] - r["control_no_new_candidates"]
(out / "control_summary.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
