# Evaluation protocol

## Eligibility and isolation

- Bin complete clean RWL files by internal inter-series correlation: `[0.60, 0.70)`, `[0.70, 0.80)`, and `≥0.80`.
- Include only target series whose clean-state master correlation is greater than `0.60` and whose reference set excludes that target.
- Split development, calibration and final by complete RWL file/content hash. A file and every series derived from it belong to one role only.
- Freeze split JSON before model fitting. Development trains; calibration selects the event gate and hyperparameters; a new final is read once after all choices are frozen.

The historical v5.0.0 engineering baseline has frozen development and calibration partitions but did not consume a new independent final. `splits/final-targets.json` records that fact explicitly. Reproduction recreates the reported baseline; it does not relabel calibration as final. A future final must use newly frozen complete files and a new model-release version.

## Scenarios

- Clean: no injected event.
- A: one event.
- B: multiple same-family events.
- C: clustered same-direction unit events with arbitrary regular or irregular gaps.
- D: mixed distant events whose adjacent event years are at least 30 years apart.

Whole shifts scan every integer displacement from -100 through +100 except zero. Partial shifts scan -100 through -2. Missing and false rings have displacements -1 and +1. Runtime inputs never include category, truth year, original zero years, bark presence or remaining-event count.

## Bark-to-pith event denominator

For a multi-event state, evaluate the newest unresolved event once. If the result is wrong or refused, count only that event, simulate the user's correct edit, rebuild all state-dependent evidence, and continue toward the pith. Case count and event-opportunity count are always reported separately.

## Correctness

A local event is correct only when:

1. final operation family is correct;
2. exact displacement is correct;
3. exactly one 13-year window is returned;
4. `windowStart <= truthYear <= windowEnd`;
5. the covered truth is the current newest unresolved event.

The predicted center or internal top-ranked year need not equal the truth. A whole event is correct only when family and exact whole displacement match; it has no local window.

## Human review transitions

The primary model emits one operation. A user may exclude a proposed whole event after inspecting bark; the same frozen state then selects the highest-scoring local family. A proposed partial event may switch to the missing locator after the user confirms there is no fracture. Exclusions are reversible and bind only to the current state. They are recorded as transitions, not as extra event opportunities. False or unrelated whole errors cannot use the partial-to-missing recovery.

## Required reports

Report coverage and final correctness overall, by A/B/C/D, file-r bin, bark-distance bin and category × r; Clean false positives; operation/displacement versus window errors; refusals; older-event prompts; review transitions; complete-file clustered bootstrap intervals; file lists and SHA-256 values. Never use an overall score to conceal a failing stratum.
