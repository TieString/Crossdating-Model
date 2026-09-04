"""Truth-blind edit/window proposals and shared runtime evidence.

Dates are edited on frozen COFECHA indices. These are alignment statistics,
not a claim that COFECHA was rerun after every hypothetical edit.
"""
from dataclasses import dataclass
import numpy as np

VIEWS = ("p50c6", "p70c6", "p70c12")
OFFSETS = (-6, -3, 0, 3, 6)
LOCAL_COLUMNS = ["hasWindow", "windowAge", "windowAgeFraction", "positiveFraction", "zeroFraction", "calendarCoverage"]
LOCAL_COLUMNS += [f"{view}{suffix}" for view in VIEWS for suffix in ("Offset", "AbsoluteOffset", "Overlap")]
LOCAL_COLUMNS += [f"local{width}{name}" for width in (5, 15, 35)
                  for name in ("GainMedian", "GainQ25", "Positive", "BeforeMedian", "AfterMedian", "NewerMedian", "ReferenceCount")]


@dataclass(frozen=True)
class Spec:
    top_k: int = 5
    stride: int = 13
    reference_count: int = 8


def propose(base, columns, identities, first, last, observed_end, spec=Spec()):
    selected = {i for i, identity in enumerate(identities) if identity[0] in ("none", "missing", "false")}
    for family in ("whole", "partial"):
        indexes = np.asarray([i for i, identity in enumerate(identities) if identity[0] == family])
        for view in VIEWS:
            scores = base[indexes, columns.index(view + "Gain")]
            selected.update(indexes[np.argsort(-scores, kind="stable")[:spec.top_k]].tolist())
    hypotheses, starts, modes = [], [], []
    if last - first < 12:
        raise ValueError("a 13-year window needs a 13-year calendar span")
    grid = set(range(first, last - 11, spec.stride)) | {last - 12}
    for identity in sorted(selected):
        if identities[identity][0] in ("none", "whole"):
            hypotheses.append(identity); starts.append(0); modes.append([0, 0, 0]); continue
        centers = [int(round(observed_end - 6 - base[identity, columns.index(view + "Distance")])) for view in VIEWS]
        proposed = grid | {max(first, min(value + offset, last - 12)) for value in centers for offset in OFFSETS}
        for start in sorted(proposed):
            hypotheses.append(identity); starts.append(start); modes.append(centers)
    return np.asarray(hypotheses, np.int16), np.asarray(starts, np.int32), np.asarray(modes, np.int32)


def moments(target, reference, years, shifts):
    low, high = min(reference), max(reference)
    dense = np.full(high - low + 1, np.nan)
    for year, value in reference.items(): dense[year - low] = value
    at = years[:, None] + shifts[None, :] - low
    y = dense[np.clip(at, 0, len(dense) - 1)]
    valid = (at >= 0) & (at < len(dense)) & np.isfinite(y)
    x = target[:, None]
    components = np.asarray([valid.astype(float), np.where(valid, x, 0), np.where(valid, y, 0),
                             np.where(valid, x*x, 0), np.where(valid, y*y, 0), np.where(valid, x*y, 0)])
    return np.concatenate([np.zeros((6, 1, len(shifts))), np.cumsum(components, axis=1)], axis=1)


def correlations(prefix, first, last, lag):
    n, sx, sy, xx, yy, xy = prefix[:, last, lag] - prefix[:, first, lag]
    denominator = np.sqrt(np.maximum(0, (xx-sx*sx/np.maximum(n, 1)) * (yy-sy*sy/np.maximum(n, 1))))
    return np.divide(xy-sx*sy/np.maximum(n, 1), denominator,
                     out=np.full_like(n, np.nan), where=(n >= 3) & (denominator > 1e-12))


def row_quantile(values, q):
    result = np.full(len(values), np.nan)
    available = np.isfinite(values).any(axis=1)
    result[available] = np.nanquantile(values[available], q, axis=1)
    return result


def build_state(sample, raw_entries, base, columns, identities, spec=Spec()):
    raw = {int(y): value for y, value in raw_entries if value is not None and value != -9999}
    years = np.asarray(sample["targetYears"], np.int32)
    order = np.argsort(years); years = years[order]
    target = np.asarray(sample["targetValues"], np.float64)[order]
    first, last = min(raw), max(raw)
    hypothesis, starts, modes = propose(base, columns, identities, first, last, int(years[-1]), spec)
    local = np.asarray([identities[i][0] not in ("none", "whole") for i in hypothesis])
    extra = np.full((len(hypothesis), len(LOCAL_COLUMNS)), np.nan)
    extra[:, 0] = local
    active = np.where(local)[0]
    start, center = starts[active], starts[active] + 6
    extra[active, 1] = last - center
    extra[active, 2] = (last - center) / max(1, last - first)
    raw_years = np.asarray(sorted(raw))
    zero_years = np.asarray(sorted(y for y, value in raw.items() if value == 0))
    extra[active, 3] = (np.searchsorted(years, start + 13) - np.searchsorted(years, start)) / 13.
    extra[active, 4] = (np.searchsorted(zero_years, start + 13) - np.searchsorted(zero_years, start)) / 13.
    extra[active, 5] = (np.searchsorted(raw_years, start + 13) - np.searchsorted(raw_years, start)) / 13.
    for view in range(3):
        difference = start - modes[active, view]
        extra[active, 6 + view * 3] = np.clip(difference / 13., -50, 50)
        extra[active, 7 + view * 3] = np.minimum(np.abs(difference) / 13., 50)
        extra[active, 8 + view * 3] = np.maximum(0, 13 - np.abs(difference)) / 13.
    master = dict(zip(sample["referenceYears"], sample["referenceValues"]))
    references = [dict(zip(ref["years"], ref["values"])) for ref in sample["individualReferences"]]
    def quality(reference):
        pairs = [(value, master[year]) for year, value in reference.items() if year in master]
        if len(pairs) < 20: return -1.
        return float(np.corrcoef(np.asarray(pairs).T)[0, 1])
    references = sorted(references, key=quality, reverse=True)[:spec.reference_count]
    shifts = np.asarray(sorted({0} | {identities[hypothesis[i]][1] for i in active}), np.int32)
    shift_indexes = np.searchsorted(shifts, [identities[hypothesis[i]][1] for i in active])
    zero = np.full(len(active), int(np.searchsorted(shifts, 0)))
    operations = [identities[hypothesis[i]][0] for i in active]
    older_end = np.asarray([np.searchsorted(years, y, side="right" if op == "missing" else "left") for y, op in zip(center, operations)])
    newer_start = np.asarray([np.searchsorted(years, y, side="left" if op == "partial" else "right") for y, op in zip(center, operations)])
    prefixes = [moments(target, ref, years, shifts) for ref in references]
    for width_index, width in enumerate((5, 15, 35)):
        older_first, newer_end = np.maximum(0, older_end-width), np.minimum(len(years), newer_start+width)
        before = np.stack([correlations(p, older_first, older_end, zero) for p in prefixes], axis=1)
        after = np.stack([correlations(p, older_first, older_end, shift_indexes) for p in prefixes], axis=1)
        newer = np.stack([correlations(p, newer_start, newer_end, zero) for p in prefixes], axis=1)
        valid = np.isfinite(before) & np.isfinite(after)
        gain = np.where(valid, after-before, np.nan)
        counts = valid.sum(axis=1)
        values = [row_quantile(gain, .5), row_quantile(gain, .25),
                  ((gain > 0) & valid).sum(axis=1) / np.maximum(1, counts),
                  row_quantile(before, .5), row_quantile(after, .5), row_quantile(newer, .5), counts]
        extra[active, 15 + width_index * 7:22 + width_index * 7] = np.asarray(values).T
    return np.concatenate([base[hypothesis], extra], axis=1).astype(np.float32), hypothesis, starts
