# Current Status — Amazon ML Challenge 2026: Business Entity Resolution

*Last updated: 2026-09-27, ~22:10*

## 1. Where we stand

| | Value |
|---|---|
| **Best public leaderboard score** | **0.983987** (`work/var_V14FR_Q_SHARE3.tsv` = `output/matching_results.tsv`, packaged in `TEAM_NAME_submission.zip`) — FINAL |
| Best held-out validation (test-like, 30k S1, US/India only) | 0.98499 (v8), 0.98492 (v9 with the leaderboard rule) |
| Leaderboard target | 0.99+ (top team: 0.9907) |
| File currently in `output/` | **V14FR_Q_NA60** (`matching_results.tsv`) + the 77.8M-pair `candidate_pairs.tsv`; validator PASS |
| Fallback | `work/var_FRNZ2M_Q_E86_T55.tsv` (LB 0.982631), `work/var_FRNZM_E86_T55.tsv` (LB 0.982206) |

**Important:** v8 changed the candidate set (+3.6M adaptive-K pairs). `output/candidate_pairs.tsv` must be the v8 version whenever v8's `matching_results.tsv` is used. It is a superset of the v7 candidates, so v7's matches also validate against it.

---

## 2. Every version: what we expected vs what we got

"Held-out" = exact macro F0.5 on S1 entities held out from training, blocked against the full training pool. From v4 on, validation is **test-like**: 19% of training S1 are removed so that the density of unowned look-alike records matches the test set.

| Version | What changed | Held-out F0.5 | What I expected on LB | **Actual LB** |
|---|---|---|---|---|
| v1 | Blocking + LightGBM pair model + exclusivity + threshold | 0.9733 (train-density, 20k S1) | about the same as held-out | **0.965** |
| v4 | Test-like training; token-substitution and house-number-relation features; stage-2 group model (consensus + sibling similarity); France noise-word adaptation | 0.98226 | clearly above 0.965 | **0.964** (dropped) |
| Hybrid | v4 for US/India + v1 for France | – | above 0.965 if France was the problem | **0.965918** |
| B | v5 (+ cross-source slot features) for US/India + v1 for France | 0.98261 | above hybrid | **0.967366** |
| C | v5 for all countries, France adaptation **off** | 0.98261 | – | **0.96944** |
| PROBE | C with every French row empty (diagnostic only) | – | – | **0.834796** |
| v7 | + raw-name / legal-form features ("Pvt" vs "Private", "PLLC" vs "Corp") | 0.98462 | modest gain (held-out was +0.20) | **0.977636** (+0.82) |
| v8 | + India state / France region normalization, raw-name frequency, raw-address similarity, adaptive K for address-less records, 1.5× training data | 0.98499 | similar or larger gain than v7 | **0.978406** (+0.08) |
| v9 | stage 1 on 150k + 300k extra S1 (hard negatives kept, easy negatives sampled) | – | small gain | **0.979268** |
| v9 rules | exclusivity; extras τ 0.60 / 0.72 / 0.77 / 0.82 / 0.87; singleton gate 0.90 | – | – | 0.978864 / 0.9797 / 0.979884 / **0.979945** / 0.979638; gate 0.979157 |
| v9 + top-1 | extras 0.82 + each S1's best candidate if p ≥ 0.45 / 0.55 / 0.65 | – | – | 0.98049 / **0.980496** / 0.980436 |
| v10h | stage 1 on 150k + 600k extra S1, rule 0.82 / 0.55 | – | small gain | **0.980742** |
| v10h E78 / E86 | extras 0.78 / 0.86, top-1 0.55 | – | – | 0.980536 / **0.980833** |
| V11FR | French self-training (pseudo-labels p ≥ 0.95 / ≤ 0.02, 400 extra rounds), E86 rule | – | small gain | **0.981384** (+0.00055) |
| **FRNZM** | **French generator noise words** (france, groupe, developpement, associes) used only where house numbers agree or no address; E86 rule | – | +0.001 to +0.003 | **0.982206** (+0.00137) |
| FRNZ2M_Q | FRNZM + *fils* as fifth noise word + source quota (≤ 5 S2, ≤ 6 S3 per S1) + exact-duplicate recovery | – | small gain | **0.982631** (+0.00043) |
| FRNZ2M_QN | FRNZ2M_Q − 10,260 French same-street links with a changed house number (external audit's rule) | held-out analogue −0.052 | lower | 0.982209 (−0.00042) |
| **V12FR** | French self-training round 1 on the FRNZM scores, noise words on | – | small gain | **0.982798** (+0.00059) |
| V12FR_Q | V12FR + quota + duplicates | – | tiny | 0.982802 |
| **V13FR_Q** | self-training round 2 (pseudo-labels from V12FR, + *fils*) + quota + duplicates | – | small gain | **0.983159** (+0.00036) |
| V13FR_Q_FR80 | France extras 0.80 | – | – | 0.983138 |
| V13FR_Q_NA70 / NA60 / NA50 / NA40 | extras for address-less records 0.70 / 0.60 / 0.50 / 0.40 | +0.0001–0.0003 (held-out) | gain | 0.983552 / **0.983579** / 0.983512 / 0.983512 |
| V14FR_Q_NA60 | self-training round 3 + address-less 0.60 + quota + duplicates | – | small gain | 0.983732 |
| V14FR_Q_NASHARE | address-less decided by dominance: (p ≥ .60, share ≥ .70) or (p ≥ .40, share ≥ .90) | – | gain | 0.98396 |
| V14FR_Q_SHARE2A | NASHARE − with-address extras with share < 0.60 | – | – | 0.983942 |
| V14FR_Q_SHARE2B | SHARE2A + address-less (p ≥ .30, share ≥ .95) | – | – | 0.983969 |
| **V14FR_Q_SHARE3** | NASHARE + address-less (p ≥ .30, share ≥ .95) — **FINAL** in `output/` and the zip | – | ≈ 0.98399 | **0.983987** |

### Where my expectations were wrong
- **v4:** I expected the held-out gain (+0.9) to show on the leaderboard. It didn't (−0.1), because the French noise adaptation hurt France.
- **France estimate:** I estimated France at ≈0.89 before the probe. The probe showed it was higher: v1-France ≈0.93, C-France ≈0.943.
- **v7:** I expected a small gain. The leaderboard gave +0.82, about 4× the held-out gain.
- **v8:** I expected the same-name features to repeat v7's jump. They gave +0.08.
- **Held-out vs LB:** held-out scores consistently overestimate the leaderboard by roughly 0.7–1.6 points. The test-like simulation closes only part of that gap.

### Per-country breakdown (derived from the France-empty probe)
France is 259,452 of 1,732,544 test S1 (14.97%); about 5.5% of French S1 are singletons.

| Model | France (LB) | US + India (LB) |
|---|---|---|
| v1 France / v4 US-IN (hybrid) | ≈0.929 | ≈0.972 |
| C | ≈0.943 | ≈0.974 |
| v7 / v8 | not probed | not probed |

---

## 3. The pipeline as it stands (v8)

Code: `code/business_entity_resolution/src/`. Run from `code/business_entity_resolution/`:

```
python -m src.pipeline --mode train --n-train 150000 --n-val 30000
python -m src.pipeline --mode predict --output-dir ../../output
```

### 3.1 Normalization (`normalize.py`, `data.py`)
- **Names:**
  - Indic-script tokens are transliterated with a dictionary learned from training pairs (1,347 tokens; 96.4% coverage of test Indic tokens). The uncovered 3.6% are category words that the generator inserts into look-alike records.
  - Accent stripping; OCR digit fixes (0→o, 1→l); web prefixes/suffixes removed.
  - Legal forms are canonicalized for the *core* name. **Raw names are also kept** for style comparison.
- **Addresses:**
  - Street, state and ordinal abbreviations canonicalized (US, India, France); leading zeros stripped.
  - **India states** in English, as codes, or in native script map to one canonical token. 16 native-script state names were learned from training pairs.
  - **France departments** map to their region (Nord → Hauts-de-France, Gironde → Nouvelle-Aquitaine, Loire-Atlantique → Pays de la Loire).

### 3.2 Blocking (`blocking.py`) — custom numba inverted-index TF-IDF
- **Channels:**
  - name (words + char 4-grams): K=30
  - address (tokens + bigrams): K=30
  - combined: K=40
  - **reverse** (each S2/S3 record searches for its best S1): K=5, **K=20 for address-less records** (adaptive K, new in v8)
- **Cap:** keep a pair if reverse rank ≤ 5 (≤ 20 when the target has no address), OR combined rank ≤ 20, OR name/address rank ≤ 10.
- **Held-out blocking recall:** 0.9844 → **0.9874** with adaptive K.
- **Test:** 77.8M candidate pairs (~45 per S1).

### 3.3 Stage-1 pair model (`features.py`, LightGBM)
About 100 features:
- rapidfuzz string similarity on names and addresses.
- Token alignment: typo vs real-word substitution; in-vocabulary substitution vs gibberish alias.
- Number relations: equal / truncated / prefix / one digit changed.
- **Raw-name similarity and legal-form agreement.**
- **Raw-name frequency** (how many S1 share the name as written).
- Raw-address similarity.
- Blocking scores and ranks; name-ambiguity counts; record-style flags.

### 3.4 Stage-2 group model (`stage2.py`)
Trained on 3-fold out-of-fold stage-1 probabilities. Features:
- Rank of the candidate within its S1, and gaps to the other candidates.
- **Consensus:** does this candidate's house number / address agree with the S1's other confident records?
- **Sibling similarity:** to the S1's confident candidates and to its rejected candidates.
- Cross-source slots: does the S1 already have a confident S2 / S3 record?

### 3.5 Decision policy
- **Exclusivity:** each S2/S3 record goes only to its highest-scoring S1.
- **Threshold:** τ = 0.675 (tuned on held-out macro F0.5).
- **Singleton gate:** S1 with no candidate above τ get an empty prediction.

---

## 4. What worked (measured)

| Change | Held-out effect | LB effect |
|---|---|---|
| Reverse blocking channel | recall 97.7% → 98.5%; strongest feature | – |
| Token-substitution + number-relation features | +0.39 | mixed (see v4) |
| Stage-2 group model | +0.10 | – |
| Consensus features (house number agrees with the S1's other records) | +0.21 | – |
| Sibling similarity; cross-source slots | +0.04; +0.04 | +0.14 (US/India, B vs hybrid) |
| France adaptation **off** | – | +0.21 (C vs B) |
| **Legal-form / raw-name features** | +0.20 | **+0.82** |
| States / raw frequency / adaptive K / 1.5× data | +0.04 | +0.08 |

## 5. What was tested and rejected (measured, not guessed)

| Idea | Result |
|---|---|
| Hard veto on house-number conflicts | Held-out 0.9826 → 0.9302. Accepted pairs with conflicting numbers are 98.4% correct; the generator perturbs numbers in true records too. |
| Per-S1 expected-F0.5 subset selection | ±0.0001 vs the tuned threshold; kept the simpler threshold. |
| Pseudo-label transliteration of unseen Indic words | Those words occur almost only in look-alike records, which are already rejected. Dropped. |
| France noise-word adaptation | Hurt France on the LB. Turned off; the code keeps it behind `--adapt-unseen`. |
| Rule: "gibberish name + exact address = match" | Only 42% precise (different businesses in one building). Not added. |
| Leakage checks (file row order, ID numbers) | No structure in train or test; all correlations ≈ 0. |
| Separate thresholds for address-less vs address-present records (v8, tuned on half A, checked on half B) | −0.00023 on B / +0.00002 on A: noise. The model is already calibrated for address-less records. |
| v7 + v8 stage-2 score ensemble (weights 0.3–1.0, two halves) | Best case +0.0001 on one half and −0.0003 on the other: no consistent gain. The two models make the same errors. |
| Higher-capacity stage 1 (255 leaves, learning rate 0.03), same v8 features and held-out entities | 0.98342 vs 0.98327: +0.00015, within noise; not worth the 2.5 h to retrain stage 2 and re-score test. |

### v8 error attribution (held-out, 30k S1; full table in `work/error_breakdown.csv`)
| Error category | Pairs | Recoverable macro-F0.5 points |
|---|---|---|
| True match in the candidates, rejected by the model | 2,112 | 0.72 (address-less 0.45) |
| True match never in the candidates | 1,318 | 0.44 |
| False: extra record owned by no S1 | 212 | 0.16 |
| False: record owned by another S1 | 199 | 0.14 |
| False: matches on a true singleton | 23 | 0.06 |
| Lost to exclusivity | 0 | 0 |

v8 ceilings: 0.99219 with zero model false negatives; 0.99582 with perfect decisions on the current candidates.

## 6. Ceiling analysis (v7 held-out)
- Current: 0.98462
- With zero model false negatives: **0.99231**
- With perfect decisions on the current candidates (blocking ceiling): **0.99511**

The largest remaining false-negative bucket is **address-less records whose exact name is shared by 3+ S1 businesses**, about 935 per 30k S1. The model is well calibrated there (true match rate ≈ 0.07–0.23), so with the current information these look genuinely ambiguous.

## 7. Problems hit along the way
- **Output writer bug:** `unique()` was called twice without preserving order, which scrambled target IDs. The validator caught it before any upload; fixed.
- **Out-of-memory / swapping on the 16 GB machine** when two heavy jobs ran together, or when all test pairs were held in RAM. Fixed with streamed, chunked scoring, spilling features to disk, keeping IDs as Arrow strings, and running one heavy job at a time.
- Scores cached from an older model were once reused with a newer policy. The cache is now keyed on the model file's timestamp.

## 8. File map

| Path | Content |
|---|---|
| `output/matching_results.tsv`, `output/candidate_pairs.tsv` | **FRNZM** (current best, LB 0.982206) |
| `work/matching_results_v8.tsv` | v8 copy |
| `work/matching_results_v7.tsv` | v7 (LB 0.977636) |
| `work/matching_results_C_v5all_noadaptFR.tsv` | C (LB 0.96944) |
| `work/matching_results_B_v5USIN_v1FR.tsv` | B (LB 0.967366) |
| `work/matching_results_hybrid_v4US_IN_v1FR.tsv` | Hybrid (LB 0.965918) |
| `work/matching_results_v4.tsv`, `work/matching_results_pass1_v1.tsv` | v4 (0.964), v1 (0.965) |
| `work/PROBE_france_empty.tsv` | Diagnostic probe (0.834796). Do not submit as final. |
| `work/var_fr85.tsv`, `var_all85.tsv`, `var_all65.tsv` | **v7**-based threshold variants (built, not submitted) |
| `work/test_final_scores.parquet` | v8 stage-2 test scores (for fast threshold variants) |
| `code/business_entity_resolution/artifacts/` | v8 models (also in `work/artifacts_v8/`); v7 in `work/artifacts_v7/`; v5 (C) in `work/artifacts_v5/` |
| `TEAM_NAME_submission.zip` | **Final package, v8** (rebuilt 08:15, validator passed). Rename it with your team name, or rebuild with `python utils/package_submission.py --team-name <Team>`. |
| `Documentation_template.md` | Methodology write-up, updated to v8 |

## 9. Findings of 2026-09-27 afternoon (the "where is the 0.02" analysis)

1. **The generator is country-independent.** Train US and India have identical link-count distributions (3.46 links per S1, 5.6 % singletons, same shares for 1…5+ links; S2 1.67, S3 1.79 per S1). On test we predicted 3.36 (US), 3.32 (India), 3.16 (France) links per S1, so France was clearly under-linked.
2. **French noise words.** The generator appends *France, Groupe, Développement, Associés* to real French copies, often after dropping one descriptive word ("Lille Ecole SARL" → "Lille SARL Développement", same address). Look-alikes use the same suffixes but almost always change the house number (same-number share 0.4–0.7 % for pure look-alike suffixes vs 15–18 % for these words, and 94 % for *associés*). The US/India equivalent (same number, one word dropped, a noise word added) is a true match 93 % of the time on held-out. Re-scoring France with these words as country noise and using the new score **only where house numbers agree or the record has no address** added 27.9k French links: LB 0.980833 → **0.982206**. Without the house-number guard, look-alikes with a different number rose to mean p ≈ 0.4, so the guard is essential.
3. **Address-less records.** In train, 97.7 % of address-less S2/S3 records are real copies (look-alikes almost always carry an address), and the test has the same number of address-less records per S1. We link only 43–51 % of them. Most unlinked ones share their name with several S1 businesses (chains). The true owner is uniformly spread over the family, so these cannot be resolved from the data (≈ 0.004 of LB for every team). Records whose exact name is unique are already linked 91 % of the time.
4. **Held-out precision / recall with the final rule (US/India):** records with an address: precision 99.8 %, recall 98.75 % (US) / 97.75 % (India), blocking recall 99.76 % / 98.81 %. Records without an address: precision 97–98 %, recall 47–51 %, blocking recall 85 %.
5. **No leaks:** row order and ID numbers carry no signal (correlation ≈ 0.001).

## 10. External audit reports (evaluated 18:40)

Two audit reports recommended extra rules. Each was tested on the 30k labelled US/India held-out S1 with the exact metric, on top of the final rule (baseline 0.98492):

| Recommendation | Links removed | Share of removed that were correct | Held-out effect | Decision |
|---|---|---|---|---|
| Drop pairs on the same street with different house numbers | 14,555 | 99.5 % | −0.0764 | rejected (France probe QN also lost −0.00042 on LB) |
| … only non-truncation number changes | 10,070 | 99.4 % | −0.0515 | rejected |
| Drop pairs with an unrelated name (name sim < 0.5) at the same address | 2,680 | 98.7 % | −0.0074 | rejected: invented names are generator noise on true copies |
| Cap S2 ≤ 5 and S3 ≤ 6 matches per S1 (train never exceeds this) | 0 on held-out, 69–78 on test | – | 0 | **applied** |
| Exact-duplicate recovery (same raw name + address as a selected record) | +1 on held-out, +34–38 on test | 100 % | 0 | **applied** |
| Escalating thresholds by rank (0.55 / 0.82 / 0.90) | 127 | 89 % | −0.00002 | rejected |

Claims in the audits that the data contradicts:
- "5.0–5.2M links expected": train gives 3.46 links per S1, i.e. ≈ 6.0M on test.
- "France over-accepts house-number conflicts": only 1.5 % of French links have a changed house number, against 11.0 % (US) and 20.9 % (India), which are 99 %+ correct on held-out. The French same-street pairs we reject carry a *different legal form* (SARL → SAS) plus a nearby number, which is the French look-alike pattern.

## 11. Root cause: why the score is 0.984 and not 0.99+ (22:10)

**Leaderboard loss 0.0160** = US/India ≈ 0.85 × 0.0140 (≈ 0.0119) + France ≈ 0.15 × 0.027 (≈ 0.0041).

US/India breakdown from the labelled held-out set, run in a test-like world where every owner is present (`work/root_cause.py`, `work/root_cause_heldout_testlike.csv`; macro F0.5 0.98596):

| Cause | Links (30k S1) | Points recoverable | Share | Fixable? |
|---|---|---|---|---|
| Never retrieved by blocking: no address | 706 | 0.22 | 16 % | only with deeper candidate search per name family (hours of compute) |
| Never retrieved by blocking: with address | 612 | 0.22 | 16 % | same; mostly India records with rewritten addresses |
| Rejected: no address, name shared by 2+ S1 (chains) | 734 | 0.24 | 17 % | **no**: nothing in the data identifies the branch |
| Rejected: no address, unique name | 87 | 0.04 | 3 % | small |
| Rejected: with address, model not confident | 1,016 | 0.33 | 23 % | partly, with a stronger model or more training data |
| Look-alike S1 took the record | 48 | 0.02 | 2 % | – |
| Wrong link: record owned by another S1 | 242 | 0.17 | 12 % | partly |
| Wrong link on a singleton S1 | 35 | 0.10 | 7 % | partly |
| Wrong link: look-alike owned by no S1 | 115 | 0.09 | 6 % | partly |

What this means:
- **0.999 is impossible with this data:** the chain ambiguity alone (0.24 points) is unidentifiable. Blocking misses (0.45 points) cap any classifier at ≈ 0.9955, and with chains ≈ 0.993.
- **Top teams (≈ 0.9907)** are about 0.007 above us. The only buckets large enough are **blocking recall** (0.45 points) and **with-address model misses** (0.33 points). A deeper candidate search, especially for India and for address-less records in large name families, plus a stronger stage 1, would be the next version's work. Each needs hours of re-blocking and re-scoring.
- **France** is still ≈ 0.013 below US/India. Its remaining gap is French look-alikes (same name, different legal form, nearby house number) that are hard to separate from real copies without labels.

## 12. In progress (2026-09-27, ~19:15)
- **v13-FR** (`python -m src.selftrain_fr v13fr v12fr fils`): second self-training round, pseudo-labels from v12-FR, *fils* added to the noise words. Then `audit_variants.py v13fr V13FR` writes `work/var_V13FR_Q_E86_T55.tsv` (+ quota and duplicates).
- `work/var_V12FR_Q_E86_T55.tsv` = V12FR + quota + duplicates.
- Code: `src/pipeline.py` now applies the unseen-country step (self-trained model `artifacts/lgb_matcher_unseen.txt` if present, otherwise noise words with the house-number guard), the leaderboard rule (`artifacts/rule.json`) and the quota / duplicate step.
