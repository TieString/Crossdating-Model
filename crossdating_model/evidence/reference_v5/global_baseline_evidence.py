"""Truth-blind global G evidence; local cumulative lag remains a nuisance path.

All G, including zero, use the same older-side transition model. No bark flag,
event class, remaining count or candidate displacement value is a feature.
"""
import importlib.util
from pathlib import Path
import numpy as np

SHIFTS = np.arange(-100, 101, dtype=np.int32)
WIDTHS = (1, 3, 5, 8, 13, 21, 34, 55, 89)
VIEWS = (("p50c6", .5, 6.), ("p70c6", .7, 6.), ("p70c12", .7, 12.))
NAMES = [f"end{width}_{name}" for width in WIDTHS for name in
         ("masterLlr50", "masterLlr70", "referenceLlrMedian", "referenceLlrQ25",
          "referenceCorrelationMedian", "referenceCoverageMedian")]
NAMES += [name+"_freeOlderPathLlrMean" for name, _, _ in VIEWS]
PAIR = None


def initialize():
    global PAIR
    spec = importlib.util.spec_from_file_location("baseline_pair", Path(__file__).with_name("evaluate-engine-pairhmm.py"))
    PAIR = importlib.util.module_from_spec(spec); spec.loader.exec_module(PAIR)
    PAIR.initialize(); PAIR.BASE.LAGS = np.arange(-340, 121, dtype=np.int32)


def reference_matrix(reference, years, shifts):
    low, high = min(reference), max(reference)
    dense = np.full(high-low+1, np.nan)
    for year, value in reference.items(): dense[year-low] = value
    indexes = years[:, None]+shifts[None, :]-low
    result = dense[np.clip(indexes, 0, len(dense)-1)]
    result[(indexes<0)|(indexes>=len(dense))] = np.nan
    return result


def llr(target, reference, beta):
    return -.5*((target[:, None]-beta*reference)**2/(1-beta*beta)
                - target[:, None]**2 + np.log(1-beta*beta))


def censored_path_emissions(target, reference, beta):
    # Missing reference observations carry no likelihood evidence. They are
    # neither extrapolated measurements nor a fixed mismatch penalty.
    values = np.where(np.isfinite(reference), llr(target, reference, beta), 0.)
    return values, np.zeros(len(target))


def finite_mean(values):
    count = np.isfinite(values).sum(axis=0)
    return np.divide(np.nansum(values, axis=0), count,
                     out=np.full(values.shape[1], np.nan), where=count>0)


def finite_quantile(values, q):
    result = np.full(values.shape[1], np.nan)
    good = np.isfinite(values).any(axis=0)
    result[good] = np.nanquantile(values[:, good], q, axis=0)
    return result


def correlation(target, reference):
    valid = np.isfinite(reference)
    n = valid.sum(axis=0)
    x = np.where(valid, target[:, None], 0); y = np.nan_to_num(reference, nan=0.)
    sx, sy = x.sum(axis=0), y.sum(axis=0)
    covariance = (x*y).sum(axis=0)-sx*sy/np.maximum(n, 1)
    scale = np.sqrt(np.maximum(0., ((x*x).sum(axis=0)-sx*sx/np.maximum(n, 1))
                                  *((y*y).sum(axis=0)-sy*sy/np.maximum(n, 1))))
    return np.divide(covariance, scale, out=np.full(len(n), np.nan), where=(n>=3)&(scale>1e-12))


def free_older_path_scores(values, null, change):
    if PAIR is None: initialize()
    return (PAIR.forward(values, null, change, "max", True)[-1]-null.sum())/len(values)


def build(sample, include_path=True):
    if PAIR is None: initialize()
    order = np.argsort(sample["targetYears"])
    years = np.asarray(sample["targetYears"], dtype=np.int32)[order]
    target = np.asarray(sample["targetValues"], dtype=float)[order]
    master = dict(zip(sample["referenceYears"], sample["referenceValues"]))
    refs = [dict(zip(ref["years"], ref["values"])) for ref in sample["individualReferences"]]
    def quality(ref):
        pairs = [(value, master[y]) for y, value in ref.items() if y in master]
        return float(np.corrcoef(np.asarray(pairs).T)[0,1]) if len(pairs)>=20 else -1.
    refs = sorted(refs, key=quality, reverse=True)[:8]
    matrices = [reference_matrix(ref, years, SHIFTS) for ref in refs]
    master_values = reference_matrix(master, years, SHIFTS)
    columns = []
    for width in WIDTHS:
        x = target[-width:]; m = master_values[-width:]
        scores = np.asarray([finite_mean(llr(x, ref[-width:], .7)) for ref in matrices])
        correlations = np.asarray([correlation(x, ref[-width:]) for ref in matrices])
        coverage = np.asarray([np.isfinite(ref[-width:]).mean(axis=0) for ref in matrices])
        columns.extend([finite_mean(llr(x,m,.5)),finite_mean(llr(x,m,.7)),
                        finite_quantile(scores,.5),finite_quantile(scores,.25),
                        finite_quantile(correlations,.5),finite_quantile(coverage,.5)])
    if include_path:
        by_beta = {}
        _, path_target, path_reference = PAIR.BASE.aligned_values(sample)
        for _, beta, change in VIEWS:
            if beta not in by_beta: by_beta[beta] = censored_path_emissions(path_target,path_reference,beta)
            values, null = by_beta[beta]
            scores = free_older_path_scores(values, null, change)
            columns.append(scores[np.searchsorted(PAIR.BASE.LAGS, SHIFTS)])
    return np.asarray(columns, dtype=np.float32).T
