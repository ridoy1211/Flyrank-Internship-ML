# Research revision status

Current revision, 6 November 2026: June recovery and artifact verification are complete. The capstone notebook, report and local paper page now use the corrected February–June results. The report notebook completed all four aggregate-verification cells without querying the warehouse or refitting models. June remains retrospective: prior exposure was uncertain, source defects were inspected, and its first VM's private predictions were lost before the disclosed rerun. No new GitHub push or portal submission has been made.

The repair log below records earlier states and is retained for provenance; its “next” or “pending” statements are historical. Current results are in `work/research/results/verified_development_v1/` and `work/research/results/june_recovery/20261105T181747Z/`.

Historical status: Fix 07 paired audit, executed in connected Colab on 4 November 2026.
Aggregate arithmetic was checked; the gated warehouse was not independently rerun.

Permanent Fix07 correction: the original cell twtLqzgdKvaB was replaced, executed
successfully (run 20, about six seconds) and saved. It now uses the scoped audit-input
guard, preserves an existing matching statistical protocol receipt and records the
actual implementation hash (838331e7...dbd34). The earlier repair had added a working
recovery cell but left the original Fix07 guard unchanged; that omission is now fixed.
The current downloadable Fix07 continuation matches the corrected original cell.
Legacy failed Fix05/Fix06 cells remain historical and must be consolidated into a
clean end-to-end notebook before final validation. Next work is documented in
outputs/Next_steps_after_Fix07.md; June remains excluded pending a frozen final plan.

The original full-results checksum guard produced a false alarm after only the April
known-client RF raw checksum and available-RAM telemetry changed. All other fields,
including published RF metrics, cohorts and LR/CTR/rule hashes, matched. The scoped
integrity guard retains those meaningful checks. Six guard tests passed. Recovery
ran in the approved Colab session without a warehouse query or June access, exactly
reproduced all 12 Fix06 rows, and completed Fix07 with all replay hashes verified.

Eight base paired comparisons and 180 client-deletion cases were checked. Later
LR-minus-prior-CTR precision@50 bounds are -2 to 0 pp (April) and -4 to +2 pp (May).
Queues share 44/50 and 43/50 pages respectively. These are finite-queue logical bounds,
not confidence intervals or a population equivalence result. Added LR value is not
demonstrated. LR-minus-CTR-excess bounds remain -8 to +14 pp and -8 to +8 pp.

All 12 Fix06 origin/arm records match the delivered audit protocol and Fix05 cohort/
observation accounting. In the known-client complete-case <=200-combined-click subset,
April has 2,743 observed signatures versus 2,102.5 reference-expected (8.16% vs 6.25%);
May has 2,413 versus 2,485.1 (6.45% vs 6.64%). The direction changes by month. These
are conditional reference comparisons, not false-positive counts, significance tests,
forecasts or attribution. Missingness and count-allocation assumptions remain limitations.

Fix07 is a paired-queue/client-influence continuation on the full original known-client
frames. It replays the four unchanged LR fits and verifies original prediction hashes,
then compares LR with prior CTR and pooled CTR excess. Shared outcomes cancel; client
deletions rerank fixed scores without refitting. Seven focused tests and full synthetic
execution passed. The real paired audit completed via scoped-guard recovery. No June access was introduced.

The approved memory recovery completed with the frozen protocol hash fa08ab63...e0a8.
288 metric records and 1,152 budget records passed consistency checks. April/May primary
known-client queues have LR observed hits/unknowns 6/9 and 6/8, giving precision bounds
12–30% and 12–28%. Prior CTR has 6/10 and 5/11; CTR excess has 8/2 and 10/0.
There is no defensible primary winner. Complete-case prior CTR matches/exceeds LR
precision@50 in both later periods and has higher AUC/AP in all four known-client origins.
The main later populations have 23/24 clients, but concentration and missingness remain;
the hash-heldout populations are still dominated by two clients. Only two later periods
were evaluated, retrospectively. Do not pool arms or claim a prospective blind test.

Fix 06 is a no-query/no-fit continuation for conditional count allocation on complete
cases with at most 200 combined clicks. It retains all primary definitions and explicitly
treats its hypergeometric reference as assumption-dependent outcome analysis. Eight focused
checks, synthetic full-cell execution and an exact-reference benchmark passed. The
user-executed count-noise audit was received and checked; June remains excluded.

Earlier-stage findings retained for provenance:

- Equal 28-day windows and explicit event eligibility are implemented and user-executed.
- The current evaluation has 32,511 pages/20 clients; the test has 11,006 pages/five clients.
- Test precision@50: original rule .02, logistic regression .16, Random Forest .14.
- Test AUC: original rule .493, logistic regression .566, Random Forest .573.
- Both learned candidates had .10 validation precision@50; the simplicity tie-break selected logistic regression.
- Coverage selection is substantial; almost half of evaluated pages have only 1–4 prior clicks.
- The paired calendar definitions disagree on 2,941/32,374 pages (9.08%).

Fix 03 used the exact same Stage 2 cohort and preserved its original three result rows.
The CTR-excess rule has validation/test precision@50 of .28/.18; prior CTR test precision
is .14 with AUC/AP close to logistic regression. The learned models and excess rule each
draw their top 50 from only two of five test clients. No reliable added ML value is established.

Fix 04 confirmed 44/50 overlap between LR and prior CTR. Their unique pages contribute
two versus one positives: only one additional hit from LR. Two test clients account for
99.75% of pages and 99.82% of positives. The >=20-click subgroup retains only two clients;
its higher precision must not motivate a posthoc primary threshold change. Secondary
ten-per-client quotas produce only 36 reviews, a different policy and budget.

Fix 05 freezes all eight methods for four chronological origins (February-May 2026).
It uses numeric features only, three full gap days and exactly three mature training
episodes, known-client and descriptive hash-heldout-client arms. Primary queues retain
unknown future outcomes and report bounds; complete-case rankings are secondary.
The notebook passed local synthetic execution and its warehouse results were received.
On 4 November the user reported that the original eight-month query exceeded its
2 GB DuckDB limit before fitting/scoring. The memory recovery preserves the frozen
protocol and original receipt. It checkpoints months, uses daily bit sets, merges
one window at a time and records the revised implementation separately. Recovery
tests, full synthetic notebook execution and a 2.8M-row/128-MiB stress check passed.
The corrected warehouse execution succeeded; see the latest findings above and Fix05_results_review.md.
June remains excluded. Remaining work includes count-noise controls, dependence-aware
uncertainty, historical ingestion provenance and prospective utility.

The deployed paper has not been updated. The local manuscript still includes earlier
revision material and must be consolidated around the final validated analysis before
publication; do not present its earlier numbers as the latest results.
