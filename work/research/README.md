# Corrected retrospective research analysis

The canonical notebook is `FlyRank_clean_research.ipynb` in this directory. It replaces the execution dependency chain of Fix02–Fix07. Those earlier files remain historical evidence. June is excluded from this notebook, including the final-month sample.

Question: Does a fixed learned ranking retrieve more later click-decline signatures at a review budget of 50 than simple rules? The event is an operational count-based signature, not a diagnosis of a page needing an edit. Evaluation uses the original frozen Fix05 protocol, all eight methods, three client arms, and February–May origins. Count allocation and paired queue comparisons are explicitly retrospective diagnostics.

## Run in Colab

1. Open the notebook. Keep your existing approved `HF_TOKEN` in Colab Secrets and enable notebook access. Never paste the token into code.
2. Run all cells. If setup installs versions and requests a session restart, restart once and run all again. CPU is sufficient.
3. A verified existing Fix05 cache is reused without a warehouse call. Otherwise, the bounded extractor scans only October 2025–May 2026, one month at a time, with verified checkpoints. A changed warehouse release stops the run.
4. The results path is printed at completion: `work/research/results/development/<UTC-run-id>/`. The final cell creates an archive containing only aggregate JSONs, the figure and caption. Uncomment its two download lines if needed.

The notebook contains the module definitions, so it does not require a repository clone or previous notebook variables. The matching Python modules support `python -m work.research` from the repository root, after installing the pinned packages. Private caches require the data provider's approved access and must never be committed.

For a local Python 3.12 environment, install with `python -m pip install -r work/research/requirements.txt`. Synthetic mode requires both `FLYRANK_CLEAN_SYNTHETIC=1` and `FLYRANK_CLEAN_CACHE` pointing to an explicitly prepared, checksum-verified synthetic cache. A real cache cannot be used as a synthetic fixture.

## What to check

`run_manifest.json` must say `complete`, `june_read: false`, and warehouse mode for real findings. Synthetic mode is a code check only. Compare cohort and non-RF prediction hashes, metrics, count-reference inputs and paired bounds with the archived repaired results. Different raw Random Forest probability bytes must be investigated separately from its published metrics; they must not be substituted or edited to pass a guard.

The primary precision denominator is the actual as-of review budget, including unknown outcomes. Missing-label and paired bounds are logical bounds for fixed queues, not confidence intervals. Complete-case ranking scores are secondary. Client deletion uses fixed predictions and measures portfolio sensitivity, not uncertainty of refitting or unseen-client performance.

## Remaining work

The separate June protocol records prior exposure as uncertain. June can be an additional retrospective evaluation. It cannot establish an untouched or prospective test. Review and version its protocol before access, implement and test the separate June runner, then execute once with the same ranking rules and report all outcomes. Afterward revise the manuscript, tables, abstract and website together so that they describe the same analysis.
