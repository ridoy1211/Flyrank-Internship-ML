# June: additional retrospective validation

Use `FlyRank_June_recovery.ipynb` after restoring the historical cache. Earlier June exposure is uncertain. The protocol was committed as version 2 in commit `6cba97f`; this analysis must not be called blind, untouched or prospective.

The original June execution stopped at its global data-quality guard: 355 page groups had duplicate daily keys, all outside the May-selected candidate population. No ranking-performance results were produced. That Colab runtime subsequently expired, losing its private predictions and aggregate. The recovery therefore repeats the fixed computation and explicitly records the earlier source-quality inspection; it cannot verify the original prediction bytes or timestamps. Model code, features, eligibility, labels, dates and settings remain unchanged.

Recovery preserves the global source audit, checks that the frozen score-key union exactly equals the May-derived candidate universe, and requires zero duplicate/count/coverage failures for those candidates. It neither deduplicates source data nor removes candidates. Absent or incomplete outcomes remain unknown. The dated execution disclosure is separate from the unchanged scientific protocol.

The code reuses the **unfiltered** May 1–28 outcome aggregate as June features, trains on the March, April and May episodes mature by May 28, freezes all supported rankings before June outcome access, then aggregates only June 1–28 from the June partition. The final-month sample is excluded. The primary comparison is LR-minus-prior-CTR precision at 50 for known clients. All eight methods and three client arms are retained; other contrasts, budgets, support thresholds, count-allocation and client-deletion checks are secondary or descriptive.

## A fresh Colab runtime

Colab VMs are temporary. A reset removes private caches and may reset package versions even though the saved notebook still shows old outputs. If the verified historical cache is missing, first use `restore_history_cache.txt`. Its first run installs the exact package versions and requests a session restart when needed. After restarting, run that recovery cell again. It scans only October–May, checkpoints each month, compares seven cohort flows and twelve evaluation feature/label hashes with the committed results, and performs no model fits or June scan.

Then run the recovery notebook. Prediction and outcome checkpoints are private and checksum-verified within that runtime. They cannot survive a terminated VM unless separately retained in an approved private location. The recovery cell and June cell use the existing approved `HF_TOKEN` Colab Secret; never paste a token into code.

## Results and interpretation

The recovery completed in the approved Colab runtime: 13 software tests passed and all 14 recovery-notebook code cells completed. The aggregate archive passed local checksum verification. Results are under `results/june_recovery/20261105T181747Z/`. The global audit reproduced 355 failed groups; none overlapped the 55,957 candidate groups with June source rows. All May-selected candidates were retained, including pages with unknown outcomes.

In the primary known-client top-50 comparison, logistic regression and prior CTR shared 49 pages. Both had five observed positive outcomes and 21 unknown outcomes; the two exclusive pages were observed negatives. Their paired precision difference was therefore zero for this fixed queue. This does not establish general equivalence between the methods or reliable absolute precision.

Only the aggregate results archive may be shared. Candidate page keys, score arrays and Parquet caches remain private. The JSON files report cohort support, prevalence, coverage, observed hits, unknown labels and precision bounds. Logical missing-label bounds are not confidence intervals. Client deletion holds scores fixed and does not refit a model. The count-reference diagnostic uses future counts and cannot be used as a predictor or to select a preferred ranking.

Repository attributes preserve LF endings for JSON, SVG and text exports, so a Windows checkout does not change their byte-level checksum receipts. PNG files remain binary.

The protocol receipt preserves its original preparation state and hash. Actual execution is recorded separately in the result manifest; do not edit a frozen protocol body to change its execution status. No manuscript or website claim should be revised from a synthetic check alone.
