# Recorded certificate benchmark

These ten records accompany the manuscript revision of 12 September 2026. Each JSON file contains the normalized circuit specification, deployed parameters, surrogate coefficients or refusal reason, numerical bounds, and measurement metadata. `summary.csv` gives the corresponding overview.

Eight instances meet the requested `1e-10` tolerance, with reported construction bounds between approximately `1.13e-16` and `5.21e-16`. The eight-qubit classifier and four-qubit fully connected ZZ example exceed the resource policy and make zero construction queries.

Measurements use 40 interval decimal digits, seed zero and the default resource policy. Construction time excludes imports and holdout comparisons. Peak RSS includes interpreter and library overhead. Each row is one worker measurement on the recorded host.

The errors concern the returned mathematical models. The ordinary floating-point holdout checks include additional evaluation and simulator rounding, so their discrepancies need not be smaller than the construction bounds.

Regenerate the results using `python scripts_revision_benchmarks.py` from the repository root. New outputs go to `benchmark_results/`, preserving these recorded measurements.
