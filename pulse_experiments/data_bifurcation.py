"""Generate data for pulse-width and pulse-speed for a varaiety of parameter combinations.

Given a set of model parameters and approximate intervals containing the speed and width
we can use the `find_c` and `find_delta` functions from the `num_assist.py` module to
to determine the solution precisely (using binary searches). Since the speed and width
are continuous with respect to the parameter inputs, we can use a given solution for a set
of paramters to generate new solutions for similar sets of parameters.

One caviat is that we expect there to be regions in parameter space without solutions.
"""
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import os
import pickle

import experiment_defaults
from num_assist import Domain, find_delta, find_c, pulse_profile, nullspace_amplitudes, v1, v2

xs_right = Domain(0, 200, 8001)
xs_left = Domain(-200, 0, 8001)

def weight_kernel(x):
    return .5*np.exp(-np.abs(x))


class Solution:
    def __init__(self, mu, alpha, gamma, theta, **_):
        self.mu = mu
        self.alpha = alpha
        self.gamma = gamma
        self.beta = 1/gamma - 1
        self.theta = theta
        self.speed = None
        self.width = None

    def __repr__(self):
        return f'Solution(mu={self.mu}, alpha={self.alpha}, gamma={self.gamma}, theta={self.theta}, speed={self.speed}, width={self.width})'

    @property
    def arr(self):
        return np.array((self.mu, self.alpha, self.gamma, self.theta))

    def dist(self, target: Solution):
        return la.norm(self.arr - target.arr)

    @property
    def dict(self):
        return {'mu': self.mu,
               'alpha': self.alpha,
               'gamma': self.gamma,
               'theta': self.theta}

    def solve(self, c_approx, Delta_approx, search_window_width=0.1, verbose=False):
        """Throws assertion error if either binary search fails."""

        speed_interval = ((1-search_window_width)*c_approx, (1+search_window_width)*c_approx)
        Delta_interval = ((1-search_window_width)*Delta_approx, (1+search_window_width)*Delta_approx)
        Delta = find_delta(*Delta_interval, *speed_interval,
                           xs_left, xs_right, verbose=verbose, weight_kernel=weight_kernel,
                           **self.dict)
        c = find_c(*speed_interval,  xs_right,
                   Delta=Delta, verbose=verbose, weight_kernel=weight_kernel,
                   **self.dict)
        self.speed = c
        self.width = Delta

class SolutionSearch:
    def __init__(self, seed: Solution):
        assert seed.speed is not None
        assert seed.width is not None
        self.solutions = [seed]

    def closest_to(self, target: Solution):
        dist, sol = min([(target.dist(sol), sol) for sol in self.solutions],
                         key=lambda tup: tup[0])
        return sol

    @staticmethod
    def sol_interp(sol1: Solution, sol2: Solution, step_size):
        step_size = min(sol1.dist(sol2), step_size)
        difference = sol2.arr - sol1.arr
        direction = difference / la.norm(difference)
        return Solution(*(sol1.arr + direction*step_size))

    def find_indermediate(self, target: Solution, step_size=10, window_width=.1):
        start = self.closest_to(target)
        target = sol_interp(start, target)
        # pick up here!!!!!!!
    



FILE_NAME = os.path.join(
        experiment_defaults.media_path,
        'bifurcation.pickle')

params = {
    'theta': 0.2,
    'beta': 5.0,
    'mu': 1.0,
    'weight_kernel': weight_kernel
}
params['gamma'] = 1/(1+params['beta'])
xs_right = Domain(0, 200, 8001)
xs_left = Domain(-200, 0, 8001)

alpha = 20.0
speed_interval = 1.0, 1.1
Delta_interval = 5, 13
c, Delta = 1.0509375967740198, 9.553535461425781
Delta = find_delta(*Delta_interval, *speed_interval,
                   xs_left, xs_right, verbose=True,
                   alpha=alpha, **params)
c = find_c(*speed_interval,  xs_right,
           Delta=Delta, verbose=True,
           alpha=alpha, **params)

alphas = [alpha]
speeds = [c]
widths = [Delta]
alpha_low = 10.0
alpha_step = 1 
while alpha > alpha_low:
    alpha -= alpha_step
    speed_interval = (c*.9, c*1.1)
    Delta_interval = (Delta*.9, Delta*1.1)
    try:
        print(f'Searching for alpha={alpha}')
        Delta = find_delta(*Delta_interval, *speed_interval,
                           xs_left, xs_right, verbose=True,
                           alpha=alpha, **params)
        c = find_c(*speed_interval,  xs_right,
                   Delta=Delta, verbose=True,
                   alpha=alpha, **params)
        alphas.append(alpha)
        speeds.append(c)
        widths.append(Delta)
    except AssertionError as e:
        print(e)


if __name__ == '__main__':
    start_params = {
        'alpha': 20.0,
        'theta': 0.2,
        'beta': 5.0,
        'mu': 1.0
    }
    start_params['gamma'] = 1/(1+start_params['beta'])

    sol = Solution(**start_params)
    sol.solve(c_approx=1.05, Delta_approx=9.5, verbose=True)
    print(sol)
    sol_search = SolutionSearch(sol)
    test = start_params.copy()
    test['alpha'] = 19.0
    sol_test = Solution(**test)
    sol_search.closest_to(sol_test)
    SolutionSearch.sol_interp(sol, sol_test, 3.4)

