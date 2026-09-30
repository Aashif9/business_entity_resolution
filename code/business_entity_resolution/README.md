# Business Entity Resolution Pipeline (team daijobu)

End-to-end solution: normalization → multi-channel inverted-index blocking → stage-1 pair
features + LightGBM → stage-2 group-context LightGBM → French noise words and self-training →
decision rule (exclusivity, thresholds, source quota, duplicates) → deep reverse rescue for India.

**Final submission: public leaderboard macro F0.5 ≈ 0.985** (the leaderboard showed "0.985"; the
previous submission without the India rescue scored 0.983987).
`Business_Entity_Resolution.ipynb` walks through the pipeline with explanations, and
`../../Documentation_template.md` describes the full method and results.

## Project structure

```
business_entity_resolution/
├── src/
│   ├── config.py        # paths (dataset/, output/, work/, artifacts/), seed
│   ├── normalize.py     # name/address normalization, learned Indic transliteration,
│   │                    # India state / France region, French legal forms and address junk
│   ├── data.py          # TSV loading, learned native-script state names, parquet caching
│   ├── blocking.py      # numba inverted-index TF-IDF search: name / addr / comb / reverse
│   │                    # channels, adaptive K for address-less records, candidate cap
│   ├── features.py      # pair features: rapidfuzz, token alignment (+ country noise words),
│   │                    # number relations, raw-name / legal-form style, name frequency
│   ├── stage2.py        # group-context features (consensus, siblings, source slots)
│   ├── model.py         # LightGBM training, exact macro F0.5, LeaderboardRule (final rule)
│   ├── pipeline.py      # train / predict entry point; French noise words; quota + duplicates
│   ├── bigtrain.py      # larger-data stage 1 (extra S1, hard negatives kept) + stage 2
│   ├── selftrain_fr.py  # French self-training on confident pseudo-labels
│   └── rescue/          # deep reverse retrieval rescue for India (final step)
│       ├── reverse_rescue.py   # deeper target → S1 search, new pairs, stage-1 scoring
│       ├── partial_rescue.py   # stage 2 for affected owners, rule, output files
│       └── control_val.py      # validation control (same groups without new candidates)
├── artifacts/           # final models and learned resources (see below)
├── Business_Entity_Resolution.ipynb
├── README.md
└── requirements.txt     # pinned versions
```

`artifacts/`: `translit_map.json`, `native_states.json`, `noise_tokens.json`,
`lgb_matcher.txt` (v10 stage 1), `lgb_stage2.txt` (v9 stage 2),
`lgb_matcher_unseen.txt` (French self-trained stage 1, round 3), `rule.json` (final decision rule),
`policy.json` (older single-threshold policy, kept for reference).

## Setup

Python 3.14 (Windows 11, 16 GB RAM, CPU only; any recent 64-bit Python should work).

```bash
pip install -r requirements.txt
```

## Reproduction

Run every command from this folder (`code/business_entity_resolution/`). Data is read from
`../../dataset/{train,test}`, outputs go to `../../output/`, caches to `../../work/`.
Run one heavy job at a time on a 16 GB machine.

### Fast path: shipped models (about 3.5 h)

```bash
# 1. blocking, stage 1, stage 2, French step with the self-trained model, final rule
#    -> ../../output/candidate_pairs.tsv, ../../output/matching_results.tsv (the 0.983987 file)
python -m src.pipeline --mode predict

# 2. India deep reverse rescue (about 20 min retrieval + 30 min stage-1 scoring + 3 min)
python src/rescue/reverse_rescue.py --project ../.. --out ../../work/rescue --country India --mode test
#    the script stops after stage 1 for test; step 2 re-runs stage 2 only for owners that
#    received a promising new pair and writes the final files
python src/rescue/partial_rescue.py
cp ../../work/rescue/partial/matching_results_partial_rescue.tsv ../../output/matching_results.tsv
cp ../../work/rescue/partial/candidate_pairs_partial_rescue.tsv  ../../output/candidate_pairs.tsv
```

### Full path: retrain everything

```bash
python -m src.pipeline --mode train --n-train 150000 --n-val 30000   # maps, noise words, base features (~80 min)
python -m src.bigtrain 1     # v9: stage 1 with 300k extra S1 + stage 2 on out-of-fold probabilities
python -m src.bigtrain 2     # v10: stage 1 with 600k extra S1
# final models = v10 stage 1 + v9 stage 2: copy them into artifacts/
cp ../../work/artifacts_v10/lgb_matcher.txt artifacts/ && cp ../../work/artifacts_v9/lgb_stage2.txt artifacts/
python -m src.pipeline --mode predict                                # without lgb_matcher_unseen.txt: guarded noise words
# French self-training, three rounds (pseudo-labels from the previous round's scores)
cp ../../work/test_final_scores.parquet ../../work/test_final_scores_frnz_merge.parquet
python -m src.selftrain_fr v12fr frnz_merge
python -m src.selftrain_fr v13fr v12fr fils
python -m src.selftrain_fr v14fr v13fr fils
cp ../../work/artifacts_v14fr_stage1.txt artifacts/lgb_matcher_unseen.txt
python -m src.pipeline --mode predict                                # re-uses cached stage-1 scores
# then the India rescue exactly as in the fast path (step 2)
```

Measured runtimes on the reference machine: predict ≈ 2.5 h (blocking ≈ 70 min, scoring
77.8M pairs ≈ 60 min, stage 2 ≈ 25 min); each French self-training round ≈ 30–40 min;
India rescue ≈ 55 min at 20 threads.

## Validation

```bash
python ../../utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir ../../dataset/test --check-ids
```

The India rescue can be checked on labelled held-out data with
`reverse_rescue.py --mode val` followed by `src/rescue/control_val.py`. That check gave
+0.00079 macro F0.5 on the 30k held-out S1 (+0.00059 / +0.00099 on the two halves). The control
(same groups without the new candidates) reproduces the baseline exactly.
