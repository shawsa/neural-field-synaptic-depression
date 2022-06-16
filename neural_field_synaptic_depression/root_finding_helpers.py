'''
Some helper functions that find all roots of a continuous function (sampled as a vector).
'''

import numpy as np
from numpy.polynomial import Polynomial

def find_root_indices(y):
    root_mask = y[1:]*y[:-1] < 0
    return np.arange(len(y)-1)[root_mask]

def find_roots(xs, ys, window=2):
    roots = []
    for root_index in find_root_indices(ys):
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
