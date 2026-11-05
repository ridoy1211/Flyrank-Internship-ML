# Project brief: evaluating a page-review ranking

Ridoy Hasan · FlyRank ML Internship · 6 November 2026

[Full report](Capstone_Report.md) · [Public report notebook](notebooks/capstone.ipynb) · [Computation and access instructions](research/README.md)

## Question and decision

A content strategist can inspect only a limited number of pages. Does a learned score retrieve more pages with a later click-decline pattern than inexpensive historical rules, at a budget of 50 reviews? The project evaluates that ranking decision. It does not test whether editing the selected pages improves traffic or revenue.

## Data and design

The study uses the gated FlyRank warehouse, release v20260703. October 2025–May 2026 partitions supply historical windows; June supplies an additional outcome window. An eligible page needs 28 explicitly available prior daily records, positive prior impressions and clicks, and valid position. Outcome labels remain unknown when later coverage is incomplete. Page identifiers support joins and splits, never prediction.

An event requires at least a 15% click decline while impressions remain within ±10%, comparing equal 28-day windows. Features end three full date labels before the outcome window. Training uses exactly three mature earlier episode origins. For June, features cover May 1–28, the gap is May 29–31, and outcomes cover June 1–28.

Fixed logistic regression and random forest models use prior impressions, clicks, CTR and weighted search position. Six simple rules provide comparators. CTR is derived from the counts, so these are not four independent information sources. Evaluation separates known-client and unseen-client designs; their results are not independent samples to pool.

## Main evidence

For known clients, the difference in precision between LR and prior CTR is bounded as follows. The ranges allow all binary assignments to unknown outcomes among the fixed queues; they are not confidence intervals.

| Outcome month | Shared top-50 pages | LR minus prior CTR |
| --- | --- | --- |
| February | 41/50 | −2 to +4 percentage points |
| March | 35/50 | −12 to 0 percentage points |
| April | 44/50 | −2 to 0 percentage points |
| May | 43/50 | −4 to +2 percentage points |
| June | 49/50 | 0 percentage points |

June's two primary queues each contain five observed positives and 21 unknowns, so each absolute precision lies between 10% and 52%. Their 49 shared pages include all unknowns; their exclusive pages are observed negatives. Thus the paired difference is zero despite uncertainty in absolute precision. Added LR value over the CTR reference is not established in these comparisons.

## What the work contributes

The documented workflow combines past-only comparisons, explicit candidate eligibility, source-grain checks, reproducible aggregate reporting and missing-outcome bounds. It provides evidence against assuming that greater model complexity improves this particular review queue. The algorithms and statistical ideas are established methods; this is a retrospective applied evaluation, not a claimed algorithmic innovation or a replication of a named paper.

## Three limitations to defend

1. **Meaning of the event:** the label describes count changes, not independently verified content harm. Prior counts enter both predictors and the threshold. Measurement variation and baseline–change dependence need consideration; their exact contribution has not been identified.
2. **Observation and dependence:** June has 14.7% unknown outcomes among eligible known-client pages and 42% in the primary queues. Repeated pages, few periods and concentrated clients limit generalization. Neither a precise whole-population lift nor future equivalence follows.
3. **Retrospective evidence:** June was rerun after an expired runtime and prior source inspection. The original prediction bytes were lost. The final release cannot verify historical ingestion times; the three-day availability buffer is assumed. June is not a blind or prospective test.

## Verification and attribution

The public capstone notebook checks archived hashes and regenerates tables. Full source-to-model reproduction needs provider-approved warehouse access and the computation notebooks. Synthetic tests check constructed software cases, not the true meaning of real events. The existing smoke-test CI primarily exercises the starter pipeline.

The project lists Ridoy Hasan as author and acknowledges AI-assisted code and editorial support. An application should describe the applicant's actual decisions and checks specifically; repository authorship alone does not establish which steps were completed independently.

## What remains worth investigating

A focused follow-up question is whether the defined event distinguishes persistent deterioration from expected variability and changes in observation coverage. Existing count-reference and minimum-click diagnostics inform that question, but do not settle it. Any extension should specify its design before examining new evidence.

The [full report's related work and references](Capstone_Report.md#related-work-and-methodological-context) connect these choices to leakage, temporal evaluation, click bias, baseline–change relationships and reporting standards. Those connections contextualize this project without asserting an unverified research gap.
