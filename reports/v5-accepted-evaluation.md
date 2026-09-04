# Accepted v5 engineering evaluation

Status: **accepted engineering baseline; not a new final holdout**. The immutable machine-readable source is `v5-accepted-evaluation.json`; it records `newFinalConsumed: false`.

## Overall

| Event opportunities | Answered | Correct | Coverage | Window accuracy | Answered accuracy | Clean false positives |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 9,038 | 9,026 | 8,406 | 99.87% | 93.01% | 93.13% | 29 / 2,924 = 0.99% |

Correct means the final operation and exact displacement are correct and the unique 13-year window covers the current frontier truth. Whole-series events require the exact whole displacement and no local window. User-triggered whole-to-local and partial-to-missing review follows the frozen `local-review-v3` protocol.

## A/B/C/D

| Category | Opportunities | Coverage | Final accuracy |
| --- | ---: | ---: | ---: |
| A, single event | 1,246 | 100.00% | 92.94% |
| B, multiple same-family | 1,168 | 100.00% | 93.58% |
| C, clustered unit events | 2,136 | 100.00% | 92.84% |
| D, distant mixed events, adjacent events ≥30 years | 4,488 | 99.73% | 92.96% |

## File-level internal-correlation strata

Targets additionally satisfy clean-state `masterCorrelation > 0.60`. File-level `r` is used only for the fixed strata.

| File r | Opportunities | Coverage | Final accuracy | Clean FP |
| --- | ---: | ---: | ---: | ---: |
| 0.60–0.70 | 3,039 | 99.74% | 89.80% | 0.90% |
| 0.70–0.80 | 2,929 | 99.86% | 93.00% | 0.93% |
| ≥0.80 | 3,070 | 100.00% | 96.19% | 1.15% |

The 0.60–0.70 stratum remains the weakest and is explicitly not hidden by the overall average. The ≥0.80 stratum-specific Clean rate exceeds 1%, although the overall Clean gate meets 1%.

## Bark-distance strata

| Distance from bark | Opportunities | Coverage | Final accuracy |
| --- | ---: | ---: | ---: |
| 1–5 | 395 | 100.00% | 92.15% |
| 6–10 | 444 | 100.00% | 95.05% |
| 11–15 | 409 | 100.00% | 94.13% |
| 16–19 | 377 | 99.47% | 91.25% |
| 20–24 | 440 | 99.77% | 90.00% |
| 25–29 | 427 | 100.00% | 92.27% |
| 30–34 | 444 | 100.00% | 94.14% |
| 35–69 | 1,629 | 100.00% | 94.23% |
| ≥70 | 3,783 | 99.76% | 93.58% |

These values report the currently deployed signed-whole integration audit. The full JSON also contains file-clustered bootstrap intervals, category × r intersections, co612 stress tests, parity evidence, failure samples and performance measurements.
