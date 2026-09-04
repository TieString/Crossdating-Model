# Frozen v5 reference closure

These files are the exact shortest source closure recovered from the final v5 research run, including the raw-to-207-field assembly stages. They are retained for auditability and to explain how the disposable candidate cache is built.

They intentionally preserve their historical CLI shapes and filenames, including references to the former research workspace. New automation should call the stable package and `scripts/reproduce_v5.py`; these files are not imported by the production desktop application.

The closure covers terminal/path evidence, operation candidates, joint 13-year windows, global binding, ranker training, evaluation and parity export. Earlier exploratory scripts and failed model variants are deliberately excluded.
