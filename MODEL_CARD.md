# Model card: automated crossdating suggestions v5.0.0

## Intended use

Human-reviewed suggestions for missing rings, false rings, partial shifts and whole-series shifts in tree-ring RWL files with adequate shared chronology signal.

## Output contract

One operation and exact displacement. Local events have one 13-year review window; whole-series shifts have no local year window. Multi-event evaluation proceeds bark-to-pith and simulates correct user resolution after each opportunity.

## Model

LightGBM LambdaRank, 207 ordered features, 600 trees, file-grouped development/calibration isolation. Production inference is pure TypeScript over a frozen JSON dump.

## Accepted development/calibration evaluation

The frozen application-side evaluation reports 8,406 / 9,038 correct event opportunities (93.01%), 99.87% coverage and 29 / 2,924 Clean false positives (0.99%). These files participated in development/calibration; `newFinalConsumed` is false. The result is the accepted engineering baseline, not a newly claimed independent final holdout.

## Limitations

Low-correlation C/D strata and distant mixed whole/local events remain weaker. Physical bark, fracture and ring-structure inspection remains authoritative. No result should be interpreted as a fully automatic redating decision.
