"""Collapse internal location candidates without mixing operation identities."""
import numpy as np


def profile_operations(scores, identities, starts, identity_count):
    result = np.full(identity_count, -1e9)
    windows = np.zeros(identity_count, np.int32)
    for identity in np.unique(identities):
        members = np.where(identities == identity)[0]
        winner = int(members[np.argmax(scores[members])])
        result[identity] = scores[winner]
        windows[identity] = starts[winner]
    return result, windows
