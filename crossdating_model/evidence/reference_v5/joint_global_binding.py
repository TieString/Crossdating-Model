"""Bind baseline G evidence to runtime hypotheses, never to their truth labels."""
import numpy as np


def bind_global_evidence(codes, identities, shifts, evidence):
    indexes={int(shift):i for i,shift in enumerate(shifts)}
    by_identity=np.asarray([indexes[int(shift)] if operation=="whole" else indexes[0]
                            for operation,shift in identities])
    return evidence[by_identity[codes]]
