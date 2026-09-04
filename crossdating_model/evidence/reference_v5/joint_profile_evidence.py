"""Complete per-window path likelihood plus observed newer-side consistency."""
import importlib.util
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from joint_window_candidates import moments, correlations, row_quantile

PAIR = None
CONFIGS = (("p50c6", .5, 6.), ("p70c6", .7, 6.), ("p70c12", .7, 12.))
NAMES = [name + suffix for name, _, _ in CONFIGS for suffix in
         ("LocalGain", "LocalLogMass", "LocalPeakDelta", "LeftEdgeFraction", "RightEdgeFraction", "OlderMass", "NewerMass")]
NAMES += ["newerSuffixCorrelationMedian", "newerSuffixCorrelationQ25", "newerSuffixMseMedian", "newerSuffixPairsMedian"]


def initialize():
    global PAIR
    spec = importlib.util.spec_from_file_location("pair_profile", Path(__file__).with_name("evaluate-engine-pairhmm.py"))
    PAIR = importlib.util.module_from_spec(spec); spec.loader.exec_module(PAIR)
    PAIR.initialize(); PAIR.BASE.LAGS = np.arange(-340, 121)


def build(sample, identities, codes, starts):
    if PAIR is None: initialize()
    local = np.asarray([identities[i][0] not in ("none", "whole") for i in codes])
    output = np.full((len(codes), len(NAMES)), np.nan)
    by_beta = {}
    for view, (_, beta, change) in enumerate(CONFIGS):
        if beta not in by_beta: by_beta[beta] = PAIR.emissions(sample, beta)
        years, values, null = by_beta[beta]
        prefix = PAIR.forward(values, null, change, "max", True)
        suffix = np.r_[np.cumsum(values[::-1, 340])[::-1], 0.]
        for identity in np.unique(codes[local]):
            operation, shift = identities[identity]
            splits = np.arange(1, len(years) + (operation != "partial"))
            boundary = years[splits] if operation == "partial" else years[splits - 1]
            penalty = change * (.5 if operation == "missing" else .75 if operation == "false" else 1.)
            scores = prefix[splits-1, 340+shift] + suffix[splits] - penalty
            if operation == "false": scores = scores + null[splits-1] - values[splits-1, 340+shift]
            total = float(logsumexp(scores)); peak = float(scores.max())
            weights = np.exp(scores-peak); cumulative = np.r_[0., np.cumsum(weights)]
            for row in np.where(codes == identity)[0]:
                start = int(starts[row]); first, last = np.searchsorted(boundary, [start, start+13])
                if first == last: continue
                current = scores[first:last]
                normalizer = float(logsumexp(current))
                probability = np.exp(current-normalizer)
                positions = boundary[first:last]
                output[row, view*7:view*7+7] = [
                    (normalizer-suffix[0])/np.sqrt(len(years)), normalizer-total, float(current.max()-peak),
                    float(probability[positions < start+3].sum()), float(probability[positions >= start+10].sum()),
                    float(cumulative[first]/cumulative[-1]), float((cumulative[-1]-cumulative[last])/cumulative[-1]),
                ]
    target = dict(zip(sample["targetYears"], sample["targetValues"]))
    years = np.asarray(sorted(target)); x = np.asarray([target[y] for y in years])
    master = dict(zip(sample["referenceYears"], sample["referenceValues"]))
    references = [dict(zip(ref["years"], ref["values"])) for ref in sample["individualReferences"]]
    def quality(reference):
        pairs = [(value, master[year]) for year, value in reference.items() if year in master]
        return float(np.corrcoef(np.asarray(pairs).T)[0,1]) if len(pairs) >= 20 else -1.
    references = sorted(references, key=quality, reverse=True)[:8]
    active = np.where(local)[0]
    begin = np.searchsorted(years, starts[active] + 13)
    end, lag = np.full(len(active), len(years)), np.zeros(len(active), int)
    corr, mse, counts = [], [], []
    for reference in references:
        prefix = moments(x, reference, years, np.array([0]))
        corr.append(correlations(prefix, begin, end, lag))
        n, _, _, xx, yy, xy = prefix[:, end, 0] - prefix[:, begin, 0]
        mse.append(np.divide(xx + .49*yy - 1.4*xy, n, out=np.full_like(n, np.nan), where=n > 0))
        counts.append(n)
    corr, mse, counts = np.asarray(corr).T, np.asarray(mse).T, np.asarray(counts).T
    output[active, -4:] = np.asarray([row_quantile(corr, .5), row_quantile(corr, .25),
                                    row_quantile(mse, .5), row_quantile(counts, .5)]).T
    return output.astype(np.float32)
