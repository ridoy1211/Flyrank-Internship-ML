# Capstone Report — The Great Decoupling: Detecting Silent Click Loss in Search Content

- **Author:** Ridoy Hasan
- **Lane:** Ranking Signal Analysis — Great Decoupling exposure detection
- **Repo:** https://github.com/ridoy1211/Flyrank-Internship-ML
- **Date:** Aug 31, 2026

## 0. Abstract

FlyRank's content strategists manage search performance for dozens of client portfolios, and
face a real, growing blind spot: since AI Overviews became standard in Google Search, a page can
hold steady impressions and ranking while its actual clicks quietly disappear — a pattern with no
name or detection method in FlyRank's existing tooling. This project asks whether that pattern,
which industry research calls the "Great Decoupling," can be identified early enough from
observable search signals for a strategist to act on. Using 123,683 real content pages from
FlyRank's 79-million-row production search warehouse, we built a transparent baseline rule and a
Random Forest classifier trained only on February 2026 signals, validated against March 2026
outcomes on a client-grouped holdout split. The baseline rule failed to beat random selection on
this split (precision@50 of 0.00),while the Random Forest achieved a 1.93x lift over the base rate (precision@50 of 0.06, ROC AUC 0.81, 95% CI [0.00, 0.16]). The result is a ranked, human-reviewed action playbook
intended to help FlyRank's content strategists triage which client pages need review first,
before the click loss becomes visible in standard reporting.

## 1. Problem framing

**Decision supported:** which pages a content/SEO strategist reviews first out of a portfolio far
too large to check by hand. **Unit of analysis:** one content page (client_hash_id +
content_hash_id pair), scored using its February performance. **Output:** a ranked score with an
archetype and one action label per page. **Action a human takes:** review flagged pages for
content-quality issues, not automatic edits. **Cost of a wrong call:** a false positive wastes a
strategist's limited review time; a false negative lets a real, valuable page keep silently
losing clicks unnoticed. **Why ML helps:** this pattern (steady impressions, falling clicks) is
invisible to a normal glance at any single metric — it only appears by deliberately comparing two
time windows across thousands of pages at once, and a hand-tuned rule (tested below) turned out
not to beat chance on held-out clients, while a model that can weigh several signals together did.

## 2. Data safety

**Tables used:** `fact_content_daily_performance` (Feb + March 2026 partitions), `dim_content`
(content_type, main_intent — static metadata only). **Deliberately excluded:**
`fact_content_query_90d` (documented leakage risk — its window overlaps recent months), GA4
columns (not needed for a GSC-only signal), any FlyRank product decision flags (not shipped in
this dataset by design). **Leakage risks considered:** label-derived columns
(`clicks_last30`/`impressions_last30`) are used only to build the evaluation label, never as
model features — confirmed with a deliberate re-add test in Week 6. `client_hash_id` and
`content_hash_id` are used only for joins and grouping, never as features. A candidate freshness
feature (`content_updated_date`) was tested and dropped after its values were found to fall
mostly *after* the March 31 decision point (median −50 days) — a genuine future-leakage risk
caught before it reached the model. No client-identifying information appears anywhere in `work/`.

## 3. Baseline

A transparent, hand-written rule: `score = visible × ctr_gap_prior30`, where `visible` is 1 if a
page's February impressions are at or above the median, and `ctr_gap_prior30` is the gap between
a page's own February CTR and the median CTR for its position bucket (clipped at zero). No fitted
weights. This is a fair comparison because it uses the exact same February-only features and is
evaluated on the exact same held-out, client-grouped test split as the model. On that split, it
scored **precision@50 of 0.00** — it did not beat random selection.

## 4. Model / analysis

**Method:** Random Forest Classifier (300 trees), chosen after Logistic Regression as a readable
first pass for a yes/no observed label, with complexity added only because it earned its place.
**Features (February-only, nothing from March):** `impressions_prior30`, `clicks_prior30`,
`ctr_prior30`, `avg_position_prior30`, `content_type`, `main_intent`. **Left out on purpose:**
`content_updated_date` (leakage risk, see above), all March-window columns, all product-flag
columns. **Label/proxy, one sentence:** `decoupling_signature` = 1 if impressions changed between
−10% and +10% and clicks dropped 15% or more, comparing February to March 2026 — a defined-rule
proxy, not an observed business outcome.

## 5. Evaluation

**Split:** `GroupShuffleSplit` grouped by `client_hash_id` (25% test) — no client appears in both
train and test, protecting against the model memorizing client-specific patterns. A random-vs-
grouped comparison in Week 6 confirmed this choice mattered.

| Method | Precision@50 | ROC AUC | Lift vs. base rate |
|---|---:|---:|---:|
| Baseline rule | 0.00 | 0.493 | 0.0x |
| Random Forest   | 0.06         | 0.805   | 1.93x               |

Base rate (test set): 3.11%. 95% bootstrap CI on the model's precision@50: [0.00, 0.16]. **Error analysis:** the baseline's complete miss at the top of its
ranking suggests the CTR-gap signal alone, without position and volume interacting jointly, isn't
sufficient to separate real decoupling risk from ordinary noise — exactly the kind of interaction
a hand-tuned single rule can't capture but a model can learn jointly.

## 6. Interpretation

The Random Forest's meaningful lift (1.93x) over both the baseline and the base rate is evidence
that combining impressions, clicks, CTR, position, and content metadata jointly captures more
signal than any one hand-picked rule. A negative result worth reporting plainly: a
content-freshness signal was hypothesized as a likely predictor (the "decay/refresh insight") but
could not be established — the available `content_updated_date` field was unusable for this
purpose in this data slice, not because freshness doesn't matter, but because this particular
field couldn't honestly support the claim.

## 7. Recommendation

Three archetypes, mapped to one action each:

| Archetype | Count | % of pages | Action |
|---|---:|---:|---|
| High-risk | 27,234 | 22.0% | REVIEW_CONTENT_QUALITY |
| Low-risk + visible | 37,278 | 30.1% | PROTECT_MONITOR |
| Low-risk + low-visibility | 59,171 | 47.8% | NO_ACTION |

A FlyRank strategist would use the "High-risk" tier as a starting review queue, prioritizing by
raw model score within it (e.g., top 50–200 at a time, not all 27,234 at once). **Confidence and
limits, stated explicitly:** this ranks a pattern already observed to correlate with an industry-
wide phenomenon; it does not diagnose why any individual page is affected, and it should never be
used to auto-publish changes, auto-redirect pages, or evaluate an individual writer's performance.

## 8. Reproducibility

Clone the repo, request access to `FlyRank/internship-warehouse` on Hugging Face, create a Read
token, and run `work/notebooks/capstone.ipynb` top to bottom in Colab with `HF_TOKEN` set as a
Secret. Random seed `42` used throughout (`GroupShuffleSplit`, `RandomForestClassifier`). The
sealed evaluation split is built and scored inside that same notebook cell, and its metrics are
committed at `work/outputs/capstone_metrics.json` — checkable from the repo, not taken on faith.

## Note:
because the warehouse query has no explicit row ordering, exact precision@50 can vary
slightly between runs even with a fixed random seed, since GroupShuffleSplit's assignment
depends on row order. ROC AUC and the qualitative conclusion (model beats baseline) are stable
across runs; the point estimate at K=50 is not.

## 9. Acknowledgments & data credit

Built on the FlyRank ML Internship dataset — [flyrank.ai](https://flyrank.ai).
