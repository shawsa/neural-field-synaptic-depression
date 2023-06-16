from functools import partial
from typing import Callable

import numpy as np
import numpy.linalg as la


class MaxIterationsReachedException(Exception):
    pass


class NewtonRootFind:

    def __init__(self,
                 F: Callable, J: Callable,
                 max_iterations=10, verbose=False):
        self.F = F
        self.J = J
        self.max_iterations = max_iterations
        self.verbose = verbose

    def roots_generator(self, vec0):
        vec = vec0.copy()
        for iteration in range(self.max_iterations+1):
            yield vec
            if self.verbose:
                print(f'Iteration: {iteration}: {vec}')
            vec = vec - la.solve(self.J(*vec), self.F(*vec))

        raise MaxIterationsReachedException()

    def cauchy_tol(self, vec0, tol=1e-12):
        seq = self.roots_generator(vec0)
        vec1 = next(seq)
        vec2 = next(seq)
        while la.norm(vec1 - vec2)/la.norm(vec1) >= tol:
            vec2, vec1 = next(seq), vec2
        return vec2

