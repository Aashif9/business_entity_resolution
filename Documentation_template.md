# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** daijobu
**Team Members:** [List all team members]
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

The pipeline has six stages, and it treats matching as a two-sided assignment problem rather than as independent pair scoring:

1. **Normalization**, including a native-script → Latin token dictionary learned from the training pairs.
2. **A custom numba inverted-index blocker** with four TF-IDF channels. One is a **reverse channel**, in which every Source-2/3 record searches for its most likely Source-1 owner.
3. **A LightGBM pair matcher.** Its features detect how the data generator perturbs true records (typos, generic word insertions, gibberish aliases, truncated house numbers) as opposed to how look-alike records differ (a real word substituted, a digit changed).
4. **A stage-2 group model** that re-scores each candidate using the context of all the S1 entity's candidates.
5. **An entity-level decision policy** with **target exclusivity** (each S2/S3 record goes to at most one S1), with thresholds tuned on exact macro F0.5.
6. **A deep reverse rescue for India**: records that are not confidently matched search much deeper for their owner, and only the affected businesses are re-scored.

Training and validation replicate the test set's higher density of look-alike records. On 30,000 held-out training S1 entities under those conditions, the system scores **macro F0.5 ≈ 0.985**. The final submission scores **≈ 0.985** on the public leaderboard (displayed as 0.985; the version without the India rescue scored 0.983987).

The two most valuable findings came from studying the data generator:
1. **The legal form as written** ("Pvt" vs "Private", "PLLC" vs "Corp") separates businesses that share a name. Canonicalizing legal forms, which is the usual practice, destroys that signal; restoring it raised the leaderboard score by +0.82 points.
2. **The generator is country-independent**, so the number of matches per S1 is the same in every country (3.46 on average). France, which has no training data, was clearly under-linked. The cause was a set of French filler words (*France, Groupe, Développement, Associés*) that the generator adds to real copies. Look-alikes use the same words but change the house number. Treating these words as noise only where the house numbers agree raised the leaderboard score by +0.0014. Then two rounds of French self-training on the resulting confident predictions, plus two structural checks from the training ground truth (at most 5 S2 and 6 S3 matches per S1; exact duplicate records share an owner), brought it to 0.983159. A lower extras threshold for address-less records (0.60) added +0.0004 (0.983579); a third self-training round brought it to 0.983732, and a share-based rule for address-less records (keep only records whose best S1 is dominant) to 0.983987. Finally, a deep reverse search for India recovered true matches that the original blocking never retrieved (+0.00079 on held-out, validated on both halves), bringing the leaderboard to **≈ 0.985**.

---

## 2. Methodology

### 2.1 Problem Analysis

- **Scale:** train has 2.2M S1 and 10.3M S2+S3 records; test has 1.73M S1 and 9.97M S2+S3 records. Blocking must be sub-quadratic and fit in 16 GB of RAM.
- **Structure of the ground truth:**
  - Every S2/S3 record belongs to **at most one** S1 (the 7.64M matched target IDs are all unique).
  - S1 entities have 3.46 matches on average (maximum 11), and 5.6% are singletons.
  - All matches are within the same country.
  - **The distribution is identical for US and India** (3.46 vs 3.46 matches per S1; singletons 5.58% vs 5.58%; same shares for 1…5+ matches; 1.67 in S2 and 1.79 in S3). The generator does not depend on country, so the expected number of matches in France is known even without labels.
  - **Address-less records are almost always real copies** (97.7% have an owner, against 73% of records with an address): look-alikes nearly always carry an address.
- **Records of businesses absent from S1 (look-alikes).** 26% of training S2/S3 records belong to no S1 entity. In test the share is about 40%: roughly 2.3 unowned records per S1, against 1.2 in train. This distribution shift explained our initial gap between validation and the leaderboard (0.973 vs 0.965). We now reproduce it during training (§4.4).
- **How the generator perturbs *true* records:**
  - *Names:* legal-suffix swaps; word reordering; inserted generic words (center, services, enterprises…); OCR-like substitutions (`C0ok`); accents; domain-style names (`clevelandcreativeoil.com`); **invented gibberish aliases** (`Novinovinex`, `X fka Y`); honorifics; ID or phone suffixes.
  - *Addresses:* truncated or zero-padded house numbers (`1585→585`, `98→0098`), reordering, abbreviations, missing addresses.
- **How *look-alikes* differ:** a distinctive word is **replaced by another real word** (Hernandez *Silver* → Hernandez *Couriers*; Ram *Marketing* → राम *ट्रेडर्स* "Traders"), and/or **one digit of the house number changes** (`302→312`).
- **Multilingual data:**
  - About 27% of India targets carry the name in one of 9 Indic scripts: Devanagari, Tamil, Telugu, Kannada, Bengali, Gujarati, Malayalam, Oriya and Gurmukhi.
  - About 25% of India targets carry the state name in a native script.
  - France (test only) uses accented Latin text and French street and legal abbreviations.
- **Ambiguity:** names are drawn from a small vocabulary, so many distinct businesses share a core name. Address-less targets with a shared name were the largest single source of residual misses.

### 2.2 Solution Strategy

**Approach Type:** High-recall multi-channel blocking, a gradient-boosted pair classifier, a group-context second stage, and an assignment-aware decision policy.

**Core Innovations:**
1. A reverse "target → best S1" blocking channel, used both for recall and as the strongest feature.
2. Features that separate generator noise from look-alike substitutions.
3. **Test-like training.** 19% of train S1 entities are removed so that their records become look-alikes, matching the test density.
4. A stage-2 model over each S1's full candidate set, trained on out-of-fold stage-1 predictions.
5. Label-free handling of the unseen country (France). A link-count check (matches per S1 should match the country-independent generator) exposes under-linking. A same-vs-different house-number test then identifies which French words are generator noise, and self-training on confident French predictions adapts stage 1 to France.
6. A leaderboard-confirmed decision rule: exclusivity, extras p ≥ 0.86, and each S1's best candidate if p ≥ 0.55.

---

## 3. Candidate Generation (Blocking)

Blocking runs per country, and countries are taken from the data as an open set, never hard-coded.

Each channel turns records into L2-normalized TF-IDF vectors of hashed features. Search uses an inverted index written in numba: each thread owns a dense score accumulator, and top-K is taken with a partition. Two cost controls keep it fast:
- Posting lists with a document frequency above a cap are dropped.
- Each query uses only its rarest 24 features (12 in the reverse channel). This keeps 98.6% top-1 agreement with the unbudgeted search at about 1/13 of the cost.

| Channel | Direction | Features | K |
|---|---|---|---|
| name | S1 → targets | core-name words + char 4-grams of the space-less core name | 30 |
| addr | S1 → targets | address tokens + adjacent-token bigrams (e.g. `2417_indiana`) | 30 |
| comb | S1 → targets | name + address features | 40 |
| **rev** | **targets → S1** | name + address features, searched against **all** S1s | 5, or **20 for address-less targets** (adaptive K) |

**Final candidate set** (written to `candidate_pairs.tsv` and scored by the model): pairs with `rev rank ≤ 5` (≤ 20 when the target has no address) OR `comb rank ≤ 20` OR `name/addr rank ≤ 10`.

- **Blocking keys used:** TF-IDF over name words, name char 4-grams, address tokens and address bigrams; a reverse target→S1 search with adaptive depth.
- **Candidate pairs generated on test:** 77.8M in total, about 45 per S1 on average. A full cross product would be 17.3 trillion pairs.
- **How true matches were kept:**
  - Recall was always measured against the *full* training pool.
  - True-pair recall is 98.51% before the cap and 98.4% after it. The cap removes only 0.06% of the positives the model would accept.
  - **Adaptive K** (reverse depth 20 for records with no address) raised recall to **98.74%** for +3.6M test pairs. Address-less records have only their name to go on, and when many S1 businesses share that name the true owner often ranks below 5.
  - Contribution of each channel: comb@40 96.7%, addr@30 87.9%, name@30 64.9%, rev@5 97.95%. The reverse channel alone raised union recall from 97.7% to 98.5%. It rescues address-less targets with common names, which rank below dozens of same-named records when searched from the S1 side.
  - A misses analysis drove normalization fixes, for example stripping leading zeros from house numbers.

**Deep reverse rescue for India (final step, `src/rescue/`).** About a third of the remaining held-out loss came from true matches the blocker never retrieved. For India, every S2/S3 record whose best current score is below 0.90 (2.04M test records) searches again with the reverse channel, to depth 100, against all Indian S1 records.
- A cheap filter keeps a pair only if it passes one of two checks: records with an address need address token-set similarity ≥ 60; address-less records need name similarity ≥ 90.
- Pairs already in the candidate set are dropped. That leaves 20.6M genuinely new test pairs, which are scored by the unchanged stage-1 model.
- Stage 2 is then re-run for the 66,390 businesses that received a promising new pair (stage-1 p ≥ 0.05), and the final rule is applied. Every other business keeps its score.
- The selected new pairs are appended to `candidate_pairs.tsv`, so every submitted match is a candidate.
- On test this added 17,336 India links and removed 466.
- **Validation** (30k held-out S1, India changed): 0.98093 → 0.98172 (**+0.00079**; +0.00059 / +0.00099 on the two halves). 231 of the 261 added links were correct (88.5%).
- A control that re-scores the same groups *without* the new candidates reproduces the baseline exactly, so the gain comes from the recovered candidates.
- Leaderboard: 0.983987 → **≈ 0.985**.
- **Why an exact-duplicate search was not enough:** we also tried seeding retrieval from exact duplicates of already-accepted siblings. It recovered only 3 of 1,318 held-out misses, because missed records are mostly heavily corrupted or sit in large same-name families.

---

## 4. Matching Model

### 4.1 Normalization

**Names**
- **Native-script tokens** are transliterated with a dictionary **learned from training pairs**. Matched S1/target names with equal token counts are aligned by position, and each native token maps to the majority Latin token. This gives 1,347 tokens and covers 96.4% of Indic token occurrences in test. A cleaned `unidecode` output is the fallback.
  - The remaining 3.6% are business-category words (ट्रेडर्स "traders", बेकरी "bakery"…). We verified that these occur almost exclusively in *look-alike* records, where the generator inserts them. The model rejects those records, and the unmapped token is itself evidence.
- Accents are stripped.
- OCR digits inside words are fixed (`0→o`, `1→l`).
- Web prefixes and suffixes are removed.
- Legal forms are canonicalized for US, India and France (LLC, Pvt Ltd, SARL, SAS, EURL…), and a *core name* is built without legal forms or honorifics.
- **The raw name is kept as well**, lower-cased with punctuation removed but legal forms *not* canonicalized, for the raw-style features in 4.2.

**Addresses**
- Street, state and ordinal variants are canonicalized for US, India and France (Road/Rd, R./Rue, Av./Avenue, Bd/Boulevard, Saint/St, Tenth/10th, US state name vs. code).
- Leading zeros are stripped from numbers.
- **India states:** S1 writes them in English ("Maharashtra"), while S3 mostly uses codes ("MH") or native script ("महाराष्ट्र"). All three map to one token. The 16 native-script state names are **learned** from the last address component of true training pairs.
- **France:** S1 writes the region ("Nouvelle-Aquitaine"), while S2/S3 often write the department ("Gironde"). Departments map to their region.
- Both mappings apply only to rows of that country, so for example "OR" (Oregon) in US addresses is untouched.

### 4.2 Pair features (≈100; none use the country label)

- **Raw name style and legal form (largest leaderboard gain):**
  - Raw-name ratio, token-sort and token-set similarity, plus exact raw equality.
  - Legal-form tokens as written (pvt / private / ltd / limited / llc / pllc / inc / corp / sarl / sas …) on each side: how many are shared, **how many the target has that the S1 lacks**, and whether the two sets are identical.
  - **Raw-name frequency:** how many S1s and targets in the country share the name *as written*. "resolve india pvt ltd" can be unique even when the core name "resolve india" is shared.
  - Raw-address ratio and token-set similarity, with no abbreviation canonicalization.

- **String similarity (rapidfuzz, multithreaded):**
  - Core name: ratio, partial ratio, token-sort, token-set, Jaro-Winkler.
  - Space-less name: ratio, partial ratio, Levenshtein (for domain names).
  - Full name: ratio and token-set.
  - Address: ratio, partial ratio, token-sort, token-set.
- **Token alignment (typo vs. substitution):**
  - Each S1 core token is fuzzily aligned to the target tokens (Jaro-Winkler ≥ 0.8, or a substring of the concatenated name).
  - We count **missing S1 words** and **extra target words**.
  - Extra words are split three ways: *generic noise* words (learned automatically as the tokens most often inserted into true matches), *in-vocabulary* words (the word occurs in ≥3 S1 names of the split, which signals a real-word substitution and so a look-alike), and *out-of-vocabulary* words (gibberish aliases, which signal a true match).
- **Number relations:**
  - Every S1 address number is compared with the target's numbers, and the best relation is kept: equal, **suffix** (truncated, the true-noise pattern), prefix, **same length with one edit** (the look-alike pattern), or none.
  - The same relation is computed for the house number, along with its length difference.
- **Set features:** token Jaccard and coverage for name and address; overlap of number sets.
- **Blocking context:**
  - Cosine score and rank in each channel.
  - Reverse rank, score relative to the target's best S1, and the target's margin between its best and second-best S1.
- **Ambiguity:** how many S1s and targets share the core name.
- **Record style:** native script, web-style name, alias marker, brackets, upper case, `#` in the address, empty address, S2 vs S3.

### 4.3 Models

- **Stage 1:** LightGBM binary classifier (127 leaves, learning rate 0.05, early stopping), 86 features.
  - It is trained on the candidate pairs of the base 150k training S1 entities **plus 600k extra training S1 entities** (`src/bigtrain.py`, 2 shards of 300k).
  - For the extra entities, every positive and every hard negative is kept, and easy negatives are subsampled with inverse-probability weights. This lets stage 1 see 5× more businesses within 16 GB of RAM.
  - Every blocked candidate is a potential training pair, so hard negatives come from the blocker itself.
- **Stage 2 (group context):** a LightGBM model re-scores each candidate using its S1's full candidate list. Its inputs are:
  - The stage-1 probability, its rank within the S1, its ratio and gap to the best *other* candidate, and the S1's counts above 0.5 and 0.9.
  - **Same-name siblings:** how many candidates with the same core name were scored confidently or rejected, and the best probability among them.
  - **Consensus:** whether this candidate's house number / full address agrees with the S1's *other* candidates, especially its confident ones. A business's own records agree with each other even where they disagree with S1; a deviation carried by a single record points to a look-alike (+0.21 held-out).
  - **Sibling similarity:** name and address similarity to the S1's confident candidates and to its plausible-but-rejected candidates.
  - **Cross-source slots:** whether the S1 already has a confident record from the candidate's source and from the other source.
  - Similarity to the S1's most confident *other* candidate; reverse-channel context; selected stage-1 features (legal form, raw-name frequency, address agreement).

  Stage 2 (57 features) is trained on **3-fold out-of-fold** stage-1 probabilities of the base 150k entities (the v9 run, whose fold models also saw 300k extra entities), so its inputs are distributed the same way at training and inference time. It resolves address-less and shared-name records by comparing them against the S1's other evidence.

### 4.4 Test-like training and validation

We remove 19% of training S1 entities (a deterministic hash) before blocking. Their S2/S3 records become look-alikes, so unowned records per S1 rise from 1.2 to about 2.3, matching the test set. Blocking, the reverse channel, name-frequency features, training and threshold tuning all run under these conditions.

### 4.5 Unseen country (France): noise words guarded by the house number

France is absent from training, so its handling relies on country-agnostic features, on French normalization (street abbreviations, department → region, `N°`/`bis`/`ter` removal, French legal forms), and on one data-driven step.

1. **Link-count check.** The generator is country-independent (§2.1), so France should have about 3.46 matches per S1. We predicted 3.16, against 3.36 for the US, which shows France is under-linked.
2. **Finding the French noise words.** We took each French record's best candidate on the same street and counted the extra words it adds to the S1 name. We compared how often each word occurs at the *same* house number and at a *different* one:
   - Pure look-alike suffixes (*Participations, Holding, Distribution, International*) are at the same number only 0.4–0.7% of the time. The generator builds a look-alike by adding one of them and changing the house number.
   - *France, Groupe, Développement* are at the same number 15–18% of the time, and *Associés* 94% of the time.
   - So these words are also **added to real copies**, like *services / center / partners* in the US. A typical case is "Lille Ecole SARL, 23 Rue Delphin Petit" → "Lille SARL Développement, N° 23 Rue Delphin Petit".
   - The US/India version of this pattern (same house number, one word dropped, a noise word added) is a true match 93% of the time on held-out.
3. **Guarded re-scoring.** French pairs are re-scored with these words passed as country-specific noise words to the token-alignment features, then run through stage 2. The new probability is used **only where the S1 and record house numbers agree or the record has no address**. Every other pair keeps its original score, because the same words on a different house number mark a look-alike.
   - Without the guard, look-alikes at a different number rose to a mean p of about 0.4.
   - With the guard, the step adds 27.9k French links and removes 1.3k, and raised the leaderboard from 0.980833 to **0.982206**.

4. **Self-training (three rounds, `src/selftrain_fr.py`).** The guarded scores give confident French pseudo-labels: p ≥ 0.95 on a record's best S1 counts as a match. Non-matches are the p ≤ 0.02 pairs; 3M of about 10.7M are sampled and weighted back to the real base rate.
   - The stage-1 model is **continued** for 400 boosting rounds (learning rate 0.03) on these rows, mixed with 3M US/India training rows as an anchor, with the French noise words (*France, Groupe, Développement, Associés, Fils*) switched on.
   - The adapted model re-scores every French pair, and stage 2 is applied. It learns the house-number guard from its pseudo-labels, so no explicit guard is needed at this point.
   - US/India keep their original scores. The US/India held-out stage-1 check stays at 0.983–0.984.
   - Round 1 (pseudo-labels from the guarded scores) scored 0.982798 on the leaderboard (+0.00059). Round 2 uses round 1's scores as pseudo-labels and adds *Fils* ("& Fils") as a fifth noise word. *Fils* sits at the same house number 14 times more often than at a different one.

An earlier attempt learned French noise words automatically from confident matches, with no guard. It picked up look-alike words too and **lowered** the score, and it remains in the code only behind `--adapt-unseen`.

### 4.6 Decision policy

1. **Exclusivity:** a record is kept only for the S1 where its probability is highest. This is applied across all test S1s.
2. **Extras:** keep every remaining link with p ≥ 0.86 for records with an address. For records **without** an address (97.7% real copies in train, since look-alikes carry an address), the decision uses how dominant the best S1 is. share = p / (sum of p over all S1 candidates of the record), and the link is kept if (p ≥ 0.60 and share ≥ 0.70) or (p ≥ 0.40 and share ≥ 0.90) or (p ≥ 0.30 and share ≥ 0.95). A record with two plausible owners is a coin flip and loses at F0.5; a record with one dominant owner is almost always right. Leaderboard, each change on top of the previous one: flat address-less threshold 0.70 / 0.60 / 0.50 / 0.40 → 0.983552 / 0.983579 / 0.983512 / 0.983512 (on round-2 scores); the share tiers (0.60, 0.70) and (0.40, 0.90) → 0.98396 (on round-3 scores); the (0.30, 0.95) tier → +0.00003. The same share test on records *with* an address (drop extras with share < 0.60) lost 0.00002, so it is not used.
3. **Top-1 rescue:** also keep each S1's single best candidate if p ≥ 0.55. An S1 with true matches scores 0 when we predict nothing, so a moderately confident best candidate is worth keeping.
4. **Source quota:** no training S1 has more than 5 Source-2 or 6 Source-3 matches, so only an S1's 5 best S2 links and 6 best S3 links are kept (about 70 links removed on test).
5. **Exact-duplicate recovery:** an unselected candidate whose raw name and address are identical to a record already selected for the same S1 (and whose best S1 is that S1) is added (about 40 links).

The thresholds were first grid-searched on the **exact challenge metric** over the held-out S1 entities, then confirmed on the leaderboard:
- extras 0.60 / 0.72 / 0.77 / 0.82 / 0.87 → 0.97886 / 0.97970 / 0.97988 / 0.97995 / 0.97964;
- adding top-1 at 0.45 / 0.55 / 0.65 → 0.98049 / 0.98050 / 0.98044;
- extras 0.78 / 0.86 with top-1 0.55 → 0.98054 / 0.98083;
- France-only extras 0.80: −0.00002 (not used).

Alternatives tested and rejected, each checked on two disjoint halves of the held-out entities:
- Per-entity expected-F0.5 subset selection: ±0.0001.
- Separate thresholds for address-less vs address-present records: −0.00023 / +0.00002.
- A v7 + v8 score ensemble: best +0.0001 on one half and −0.0003 on the other.
- Rules from an external audit of our output, all measured on the labelled held-out (baseline 0.98492):
  - Dropping links on the same street whose house number differs: **−0.076**. The 14,555 links removed were 99.5% correct, because the generator changes house numbers on real copies.
  - Keeping only the non-truncated number changes: −0.052. The same rule applied to France on the leaderboard: −0.00042.
  - Rejecting unrelated names at an identical address: −0.0074. Invented aliases are generator noise on real copies.
  - Rank-dependent thresholds: −0.00002.

**Threshold selection method:** F0.5 optimization on test-like held-out S1 entities, followed by leaderboard confirmation of the two thresholds.

---

## 5. Results & Error Analysis

| Version | Held-out macro F0.5 (US/India) | Public leaderboard |
|---|---|---|
| v1: pair model | 0.9733 (train-density validation) | 0.965 |
| v4: + substitution/number features, test-like training, stage 2, France noise adaptation | 0.9823 | 0.964 |
| Hybrid: v4 US/India + v1 France | – | 0.965918 |
| B: v5 US/India + v1 France | 0.9826 | 0.967366 |
| C: v5 for all countries, France without noise adaptation | 0.9826 | 0.96944 |
| v7: + raw-name / legal-form features | 0.98462 | 0.977636 |
| v8: + India state / France region normalization, raw-name frequency, raw-address similarity, adaptive K, 1.5× training data | 0.98499 | 0.978406 |
| v9: stage 1 on 150k + 300k extra S1 (hard negatives kept) | 0.98527 | 0.979268 |
| v9 + decision rule (exclusivity, extras 0.82, top-1 0.55) | – | 0.980496 |
| v10h: stage 1 on 150k + 600k extra S1, v9 stage 2, extras 0.86 / top-1 0.55 | – | 0.980833 |
| v10h + French self-training | – | 0.981384 |
| v10h + French noise words guarded by the house number | – | 0.982206 |
| + *Fils*, source quota, exact-duplicate recovery | – | 0.982631 |
| + French self-training round 1 | – | 0.982798 |
| round 1 + source quota + exact-duplicate recovery | – | 0.982802 |
| self-training round 2 (+ *Fils*) + quota + duplicates | – | 0.983159 |
| + address-less extras threshold 0.60 | – | 0.983579 |
| self-training round 3 + address-less 0.60 + quota + duplicates | – | 0.983732 |
| + share-based address-less rule (tiers (0.60, 0.70), (0.40, 0.90)) | – | 0.98396 |
| + share tier (0.30, 0.95), and share < 0.60 for records with an address | – | 0.983969 |
| share tiers (0.60, 0.70), (0.40, 0.90), (0.30, 0.95); records with an address unchanged | 0.98596 (test-like US/India) | 0.983987 |
| **Final: + deep reverse rescue for India** | +0.00079 (held-out, India changed) | **≈ 0.985** |

**Per-country leaderboard decomposition (for model C).** We submitted one probe with every French prediction left empty; it scored 0.834796. France is 14.97% of test S1 entities, and roughly 5.5% of French S1s are singletons, which score 1.0 when predicted empty. From this probe and the submissions above we estimate **France ≈ 0.943** and **US + India ≈ 0.974** for model C. These are conditional estimates: they assume the public subset has the full test set's country mix and singleton rate.

**Where the remaining loss comes from (final rule, 30k labelled US/India held-out S1, test-like: every owner present; macro F0.5 0.98596):**

| Cause | Links | F0.5 points recoverable | Share of loss |
|---|---|---|---|
| True match never retrieved by blocking, record without address | 706 | 0.22 | 16% |
| True match never retrieved by blocking, record with address | 612 | 0.22 | 16% |
| Rejected, no address, name shared by 2+ S1 (chains: owner not identifiable) | 734 | 0.24 | 17% |
| Rejected, no address, unique name | 87 | 0.04 | 3% |
| Rejected, with address (heavy noise on both fields) | 1,016 | 0.33 | 23% |
| Retrieved but a look-alike S1 scored higher | 48 | 0.02 | 2% |
| Wrong link to a record owned by another S1 | 242 | 0.17 | 12% |
| Wrong link on a singleton S1 | 35 | 0.10 | 7% |
| Wrong link to a look-alike owned by no S1 | 115 | 0.09 | 6% |

France (no labels) is estimated at ≈ 0.973 from the leaderboard, against ≈ 0.986 for US/India.

**Held-out precision / recall with the E86 rule (US/India, 30k S1):**

| Records | Blocking recall | Recall | Precision |
|---|---|---|---|
| With address, US | 99.76% | 98.75% | 99.84% |
| With address, India | 98.81% | 97.75% | 99.83% |
| Without address | ≈85% | 47–51% | 97–98% |

Most unmatched address-less records share their exact name with several S1 businesses (chains). In train, the true owner is spread evenly over those businesses (the owner is the one with the most other matches only 26% of the time), so nothing in the provided fields identifies it. Leaving these records out is the F0.5-optimal choice.

**Lessons from the leaderboard:**
1. The label-free French noise-word adaptation *hurt*. In France the generator swaps generic organisation words (Club / Collectif / Groupement) in true records, while French look-alikes sit at neighbouring house numbers. Treating the words as harmless let in look-alikes, so it was removed.
2. A hard "house-number conflict → reject" rule was tested on labeled US/India data and rejected. The model accepts about 10% of its links with a conflicting number, those links are 98.4% correct, and vetoing them lowers held-out F0.5 from 0.9826 to 0.9302.
3. US/India reach about 0.974 on the leaderboard against 0.983 held-out, so part of the test distribution is harder than our test-like simulation.
4. Features that separate **same-name businesses** gain far more on the leaderboard than held-out (legal form: +0.20 held-out, +0.82 LB). This suggests the test set has more same-name ambiguity than even our test-like validation.

- **Loss decomposition (v8, held-out, in macro-F0.5 points):**

| Error category | Pairs (30k S1) | Recoverable points |
|---|---|---|
| True match in the candidates, rejected by the model | 2,112 | 0.72 (0.45 of it address-less records) |
| True match never in the candidates | 1,318 | 0.44 |
| Extra record that belongs to no S1 | 212 | 0.16 |
| Record that belongs to another S1 | 199 | 0.14 |
| Matches given to a true singleton | 23 | 0.06 |

  Ceilings: 0.99219 with zero model false negatives; 0.99582 with perfect decisions on the current candidates. The largest remaining bucket is address-less records whose exact name is shared by 3+ S1 businesses. The model is calibrated there (actual match rate 0.07–0.23), so the provided fields do not identify the owner.
- **Common false positives (wrong merges):**
  - Look-alike records at the same address whose only difference is one generic word or digit.
  - Address-less records whose name is shared by another business that is absent from S1.
- **Common false negatives (missed matches):**
  - True records whose house number was changed by the generator (`2335→2333`), combined with name corruption.
  - Address-less true records with very common names.

---

## 6. Conclusion

The largest gains came from understanding the data generator and the test distribution rather than from model capacity:
- comparing names **as written** (legal-form style), not only after canonicalization;
- the reverse "who owns this record" channel and the exclusivity rule;
- features that separate generator noise from look-alike substitutions;
- training under test-like look-alike density;
- a group-context second stage;
- recovering blocking misses with a deeper, targeted reverse search instead of a global re-blocking.

The linear-time custom blocker keeps the full 1.7M × 10M problem tractable on a 16 GB laptop. The multilingual and unseen-country handling is learned entirely from the provided data.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`:

- `src/normalize.py`: normalization and the learned transliteration dictionary
- `src/data.py`: TSV loading and parquet caching
- `src/blocking.py`: numba inverted-index blocker (4 channels) and candidate cap
- `src/features.py`: pair features, token alignment, number relations, noise and vocabulary learning
- `src/stage2.py`: group-context features for stage 2
- `src/model.py`: LightGBM, exact macro F0.5, decision policy and tuning
- `src/pipeline.py`: entry point for train, stage2 and predict; unseen-country noise words with the house-number guard (`rescore_unseen_noise`)
- `src/bigtrain.py`: larger-data stage-1 training (extra S1 with hard negatives kept) and stage 2 on out-of-fold probabilities
- `src/selftrain_fr.py`: French self-training on confident pseudo-labels (used in the final submission)
- `src/rescue/reverse_rescue.py`, `src/rescue/partial_rescue.py`, `src/rescue/control_val.py`: deep reverse rescue for India, its final step, and its validation control
- `Business_Entity_Resolution.ipynb`: explained walkthrough of the pipeline
- `artifacts/`: transliteration map, native-script state names, noise tokens, stage-1 and stage-2 models, the French self-trained stage 1 (`lgb_matcher_unseen.txt`), `rule.json` (final decision rule)

To reproduce, run from `code/business_entity_resolution/`:

```bash
pip install -r requirements.txt
python -m src.pipeline --mode train --n-train 150000 --n-val 30000   # base blocking/features, noise words, maps (~80 min)
python -m src.bigtrain 1          # v9: stage 1 with 300k extra S1 + stage 2 on out-of-fold probabilities
python -m src.bigtrain 2          # v10: stage 1 with 600k extra S1 (its lgb_matcher.txt is used with the v9 stage 2)
python -m src.pipeline --mode predict                                # blocking, stage 1, stage 2, French noise words, final rule
# French self-training (label-free), starting from the guarded noise-word scores of the run above:
cp ../../work/test_final_scores.parquet ../../work/test_final_scores_frnz_merge.parquet
python -m src.selftrain_fr v12fr frnz_merge                          # round 1
python -m src.selftrain_fr v13fr v12fr fils                         # round 2
python -m src.selftrain_fr v14fr v13fr fils                         # round 3
cp ../../work/artifacts_v14fr_stage1.txt artifacts/lgb_matcher_unseen.txt
python -m src.pipeline --mode predict                                # re-uses cached stage-1 scores; France re-scored by the adapted model
# India deep reverse rescue (final step)
python src/rescue/reverse_rescue.py --project ../.. --out ../../work/rescue --country India --mode test
python src/rescue/partial_rescue.py
cp ../../work/rescue/partial/matching_results_partial_rescue.tsv ../../output/matching_results.tsv
cp ../../work/rescue/partial/candidate_pairs_partial_rescue.tsv  ../../output/candidate_pairs.tsv
```

The shipped `artifacts/` already contain the final models: `lgb_matcher.txt` (v10 stage 1), `lgb_stage2.txt` (v9 stage 2), `lgb_matcher_unseen.txt` (the French self-trained stage 1) and `rule.json` (extras 0.86, share tiers for address-less records, top-1 0.55). So `--mode predict` followed by the two rescue scripts reproduces the submission. The one exception is France's pre-self-training base scores, which in our run predate the French `N°`/`bis`/`ter` address rules; this affects about 0.2% of French links. `Business_Entity_Resolution.ipynb` walks through every step with explanations.

Prediction streams the 77.8M candidate pairs in 2M-pair chunks and keeps IDs as Arrow strings, so it fits in 16 GB of RAM (~75 min on 22 threads).

### B. Additional Results

| Stage | Metric | Value |
|---|---|---|
| Blocking (uncapped) | true-pair recall | 98.51% |
| Blocking (final cap + adaptive K, ~45 cands/S1) | true-pair recall | 98.74% |
| Transliteration dictionary | Indic token coverage (test) | 96.4% |
| Stage 1 (test-like validation) | held-out macro F0.5 | 0.98327 |
| Stage 1 + stage 2 (test-like validation) | held-out macro F0.5 | 0.98499 |
| Test inference | candidate pairs | 77,812,265 |
| Submission before the India rescue (round-3 self-training, share-based address-less rule) | public leaderboard | 0.983987 |
| **Final submission (+ deep reverse rescue for India)** | public leaderboard | **≈ 0.985** |
