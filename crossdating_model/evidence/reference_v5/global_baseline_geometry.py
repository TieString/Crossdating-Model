"""Latest-plateau geometry from the same censored path, for every candidate G."""
import numpy as np
import global_baseline_evidence as base

FIELDS = ("lastJump", "latestObservations", "latestCalendarYears", "priorObservations",
          "latestLlrMean", "latestGainMean", "latestGainSum", "hasTransition", "lastPenalty")
NAMES = [name+"_"+field for name,_,_ in base.VIEWS for field in FIELDS]


def incoming(history, values, null, time, state, change):
    previous = history[time-1]
    source, best, penalty = state, previous[state], 0.
    if state >= 2:
        first = max(0,state-100)
        candidate = first+int(np.argmax(previous[first:state-1]))
        score = previous[candidate]-change
        if score > best: source,best,penalty = candidate,score,change
    if state > 0:
        score = previous[state-1]-.5*change
        if score > best: source,best,penalty = state-1,score,.5*change
    if state+1 < len(previous):
        score = previous[state+1]-.75*change+(null[time-1]-values[time-1,state+1])
        if score > best: source,best,penalty = state+1,score,.75*change
    return source,best,penalty


def geometry(values, null, years, states, change):
    if base.PAIR is None: base.initialize()
    history = base.PAIR.forward(values,null,change,"max",True)
    needs_change = history[1:] > history[:-1]+values[1:]
    times = np.arange(1,len(values))[:,None]
    last = np.maximum.accumulate(np.vstack([np.full((1,values.shape[1]),-1),
                                            np.where(needs_change,times,-1)]),axis=0)
    cumulative = np.vstack([np.zeros((1,values.shape[1])),np.cumsum(values,axis=0)])
    result = []
    for state in states:
        at = int(last[-1,state])
        if at < 0:
            result.append([0,len(values),years[-1]-years[0]+1,0,
                           cumulative[-1,state]/len(values),0,0,0,0])
            continue
        source,_,penalty = incoming(history,values,null,at,int(state),change)
        prior_at = int(last[at-1,source])
        count = len(values)-at
        matched = cumulative[-1,state]-cumulative[at,state]
        old = cumulative[-1,source]-cumulative[at,source]
        result.append([source-state,count,years[-1]-years[at]+1,
                       at-max(0,prior_at),matched/count,(matched-old)/count,matched-old,1,penalty])
    return np.asarray(result,dtype=np.float32),(history[-1,states]-null.sum())/len(values)


def build(sample):
    if base.PAIR is None: base.initialize()
    years,target,reference = base.PAIR.BASE.aligned_values(sample)
    states = np.searchsorted(base.PAIR.BASE.LAGS,base.SHIFTS)
    by_beta, parts = {}, []
    for _,beta,change in base.VIEWS:
        if beta not in by_beta: by_beta[beta] = base.censored_path_emissions(target,reference,beta)
        values,null = by_beta[beta]
        features,_ = geometry(values,null,years,states,change)
        parts.append(features)
    return np.concatenate(parts,axis=1)
