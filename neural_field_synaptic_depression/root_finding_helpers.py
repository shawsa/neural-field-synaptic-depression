'''
Some helper functions that find all roots of a continuous function (sampled as a vector).
'''

import numpy as np
from numpy.polynomial import Polynomial

def find_root_indices(y, window=0):
    window_mask = np.logical_and(window <= np.arange(len(y)),
                                 np.arange(len(y)) < len(y)-window)
    root_mask = np.logical_and(y[1:]*y[:-1] < 0, window_mask[:-1])
    return np.arange(len(y)-1)[root_mask]

def find_roots(xs, ys, window=2):
    roots = []
    for root_index in find_root_indices(ys, window=window):
        start_index = root_index-window//2
        locs = slice(start_index, start_index+window)
        p = Polynomial.fit(xs[locs], ys[locs], window-1)
        p_prime = p.deriv()
        x = xs[root_index]
        for _ in range(5):
            x -= p(x)/p_prime(x)
        if not xs[root_index-1] < x < xs[root_index]+1:
            x = xs[root_index]
        roots.append(x)
    return roots
