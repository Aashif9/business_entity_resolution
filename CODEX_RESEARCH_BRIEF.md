# Research brief for Codex: find the generator rules we are still missing

Paste this whole file into Codex. Deadline: tonight 23:59. Goal: public leaderboard ≥ 0.99; we are at **0.980496**.

## ⚠️ Machine constraints (read first)
* A long job is running on this 16 GB machine until about 15:45 (`python -m src.pipeline --mode predict`). **Do not kill Python processes, and do not load full datasets.**
* Work on **samples**. Read parquet files **lazily and by column** (`pl.scan_parquet(...).select(...)`), and keep each script under ~2 GB of RAM.
* **No external data or lookups.** Only the provided files may be used; the competition rules forbid anything else.

## Problem in one paragraph
Source 1 (S1) is a deduplicated reference list of businesses. Sources 2 and 3 (S2/S3) contain noisy copies of those businesses plus records of *other* businesses. For each S1, predict the set of its S2/S3 records (0, 1 or many).

The score is macro F0.5 per S1. Precision weighs 2×. A true singleton scores 1 only if we predict an empty list.

Every S2/S3 record belongs to **at most one** S1. Train covers US and India; test adds **France** with no labels (15% of the score).

## Data (all paths relative to `C:\Users\surya\Downloads\amazon ml\`)
* Raw data: `dataset/train/train_source{1,2,3}.tsv`, `dataset/train/train_ground_truth.tsv`, and `dataset/test/test_source{1,2,3}.tsv`.
  * Source columns: `entity_id, business_name, business_address, country`.
  * Ground-truth columns: `source1_entity_id, matched_entity_ids` (comma-separated IDs).
  * Read them with `pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)`.
* Normalized copies (faster): `work/train_s{1,2,3}_norm.parquet` and `work/test_s{1,2,3}_norm.parquet`. Extra columns:
  * `name_core`: lower-case name without legal forms or stopwords.
  * `name_full`: lower-case name with legal forms canonicalized.
  * `addr`: normalized address.

## What we already know (do NOT re-derive these)
* No leakage: file row order and ID numbers are unrelated to matches (correlations ≈ 0).
* **Already handled:** Indic transliteration, typos, legal-form expansions (and legal-form *style* — the biggest win), OCR 0→o / 1→l, accents, abbreviations, state codes, historical city aliases, truncated or padded house numbers, word reordering, prefix junk (`>>`, `##`), web-style names.
* **Remaining held-out errors** (30k S1; 2,112 missed true pairs, ~430 wrong pairs):
  * 69% of missed true pairs are **address-less records whose exact core name is shared by 2+ S1 businesses**.
  * 52% of missed pairs already have an identical name after normalization.
  * Wrong pairs are mostly look-alike businesses: same base name, one generic word swapped and/or a changed house number.
* **What moved the leaderboard:**
  * Stricter bar for *extra* matches (p ≥ 0.82).
  * Looser bar for each S1's *single best* candidate (p ≥ 0.55).
  * → Look-alikes mostly appear as **2nd/3rd candidates**, and the top candidate is usually right.
* **France** scores about 0.95 (US/India about 0.985).
  * French S1 contains separate businesses with the **same name on the same street** (1% of French S1 vs 0.01% elsewhere).
  * Rescuing French same-address pairs did not help.

## Research questions (ranked by expected value)

### Q1. How are look-alike (distractor) records generated? (train)
Take ~20k train S1.
1. Find S2/S3 records whose `name_core` equals, or nearly equals, the S1's `name_core` (same country) but that are **not** that S1's matches in the ground truth.
2. Compare them with the S1's true records. For each, report:
   * whether it belongs to another S1 or to no S1;
   * source (S2 vs S3);
   * whether it has an address;
   * house number relation to the S1 (equal / ±1 / other);
   * street equality;
   * extra or missing words;
   * legal-form token;
   * case style and punctuation.

**Looking for:** any attribute where look-alikes and true records differ systematically. For example, "a look-alike never keeps the exact S1 legal-form string", or "a look-alike always changes exactly one of {name word, house number}".

### Q2. Do an entity's true records share invariants? (train)
For S1 entities with 3+ true records, check what is *identical across all of them* but differs from the S1 or from look-alikes. Examples:
* the same house number, even when it differs from the S1's;
* the same street spelling;
* the same legal-form spelling;
* the same token order.

If the generator derives all copies from one base string, an address-less record can be tied to the cluster whose other records share its exact name variant.

**Report:** for each invariant, how often it holds within an entity vs between an entity and a same-name look-alike.

### Q3. Address-less records: which S1 owns them? (train)
For true records with an empty address whose `name_core` is shared by 2+ S1 in the same country, find any attribute that picks the right owner among the same-name S1s. Candidates:
* exact raw-name equality after lowercasing;
* legal-form spelling;
* `name_full` equality;
* source;
* S1 name-length or token patterns;
* whether the owner S1 already has other address-less records;
* count balance per source.

**Report:** the accuracy of each rule, as the share of cases where it uniquely identifies the owner.

### Q4. Records-per-entity limits (train)
Report the distribution of (#S2, #S3) true records per S1. Is there a hard cap, or a fixed pattern (e.g. never more than N from one source, always at least one S3 when there is an S2)?

A cap would let us drop the lowest-scoring extra matches beyond it.

### Q5. Singletons (train)
What fraction of true singletons have a near-exact name record (same `name_core`) in S2/S3? How do those records differ from the true records of non-singletons with the same pattern?

**Looking for:** a rule that marks an S1 as a singleton with high precision.

### Q6. France (test only, no labels)
Compare noise-operation frequencies in French S2/S3 against US/India S2/S3:
* prefix junk, casing, legal-form variants, web names;
* `#` house numbers, missing addresses;
* department vs region, number formats (`N°`, `bis`, `ter`).

List every **French-only** operation, and check whether any of them could make true French pairs look like look-alikes to a model trained on US/India.

## How to report (for each question)
1. The exact numbers, on how many S1 or records.
2. The code you ran (short).
3. A proposed **feature or rule**, and on how many of 30k held-out S1 it would change a decision.

Rules must be checked on the training labels, and must be precise enough to raise macro F0.5. A false match costs twice as much as a missed one.

We will implement and test anything promising within an hour.
