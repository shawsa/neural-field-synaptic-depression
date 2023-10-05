"""Generate data for pulse-width and pulse-speed for a varaiety of parameter
combinations. Given a set of model parameters and approximate intervals
containing the speed and width we can use the `find_c` and `find_delta`
functions from the `num_assist.py` module to to determine the solution
precisely (using binary searches). Since the speed and width are continuous
with respect to the parameter inputs, we can use a given solution for a set of
paramters to generate new solutions for similar sets of parameters.

One caviat is that we expect there to be regions in parameter space without
solutions.
"""

import experiment_defaults

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import os
import pickle

from itertools import product

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

    def dist(self, target):
        return la.norm(self.arr - target.arr)

    @property
    def dict(self):
        return {'mu': self.mu,
                'alpha': self.alpha,
                'gamma': self.gamma,
                'theta': self.theta}

    def solve(self, c_approx, Delta_approx, search_window_width=0.1,
              delta_tol=1e-2, speed_tol=1e-2,
              verbose=False):
        """Throws assertion error if either binary search fails."""

        speed_interval = ((1-search_window_width)*c_approx, (1+search_window_width)*c_approx)
        Delta_interval = ((1-search_window_width)*Delta_approx, (1+search_window_width)*Delta_approx)
        Delta = find_delta(*Delta_interval, *speed_interval,
                           xs_left, xs_right, verbose=verbose, weight_kernel=weight_kernel,
                           **self.dict, tol=delta_tol)
        c = find_c(*speed_interval,  xs_right,
                   Delta=Delta, verbose=verbose, weight_kernel=weight_kernel,
                   **self.dict, tol=speed_tol)
        self.speed = c
        self.width = Delta

    def refine(self, verbose=False):
        """Find a more exact solution using the found speed and width as a starting point."""
        self.solve(c_approx = self.speed, Delta_approx=self.width,
                   search_window_width=1e-2, delta_tol=1e-5, speed_tol=1e-8,
                   verbose=verbose)

    def plot(self, ax=None, color='blue'):
        try:
            xs, Us, Qs = pulse_profile(xs_left, xs_right,
                                       c=self.speed, Delta=self.width,
                                       weight_kernel=weight_kernel, **self.dict)
        except ValueError:
            xs, Us, Qs = pulse_profile(xs_left, xs_right,
                                       c=self.speed, Delta=self.width,
                                       weight_kernel=weight_kernel, **self.dict,
                                       vanish_tol=None)
        if ax is None:
            plt.figure()
            plt.plot(xs, Us, '-', color=color)
            plt.plot(xs, Qs, '--', color=color)
            plt.ylim(-.1, 1.1)
        else:
            ax.plot(xs, Us, '-', color=color)
            ax.plot(xs, Qs, '--', color=color)
            ax.set_ylim(-.1, 1.1)




class SolutionSearch:
    def __init__(self, seed: Solution):
        self.solutions = []
        self.add(seed)

    def add(self, sol: Solution):
        assert sol.speed is not None
        assert sol.width is not None
        self.solutions.append(sol)

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

    def _find_indermediate(self, start: Solution, target: Solution,
                           step_size, window_width,
                           verbose=False):
        """Tries to find a new solution in the direction of the target that
        is at most `step_size` away (in Euclidean distance). Can throw
        assertion errors if the binary search fails.
        """
        target = SolutionSearch.sol_interp(start, target, step_size)
        if verbose:
            print(f'Attempting solve: {target}')
        target.solve(start.speed, start.width,
                     search_window_width=window_width,
                     verbose=verbose)
        self.add(target)
        return target

    def seek(self, target: Solution,
             min_step_size=0.01, window_width=0.1,
             verbose=False):
        """Attempts to traverse the parameter space, finding intermediate
        solutions, until it arives at the target solution."""
        sol = self.closest_to(target)
        step_size = sol.dist(target)
        while sol.dist(target) > 1e-5:
            try:
                if verbose:
                    print(f'Attempting a step: {sol}')
                sol = self._find_indermediate(sol, target, step_size,
                                              window_width, verbose=verbose)
            except AssertionError:
                step_size /= 2
                if step_size < min_step_size:
                    raise ValueError
                if verbose:
                    print(f'Attempt, failed, refining step-size: {step_size}')
                continue
        return sol


if __name__ == '__main__':
    FILE_NAME = os.path.join(
            experiment_defaults.data_path,
            'bifurcation.pickle')

    if False:
        """Load the previously saved search. This is included to help
        explore using the REPL.
        """
        with open(FILE_NAME, 'rb') as f:
            sol_search = pickle.load(f)

    start_params = {
        'alpha': 20.0,
        'theta': 0.2,
        'beta': 5.0,
        'mu': 1.0
    }
    start_params['gamma'] = 1/(1+start_params['beta'])
    sol = Solution(**start_params)
    print('Solving seed solution.')
    sol.solve(c_approx=1.05, Delta_approx=9.5, verbose=True)
    print(sol)

    sol_search = SolutionSearch(sol)
    alphas = list(range(20, 7, -1))
    gammas = [0.13, 0.14, 0.15, 0.17, 0.19]
    gammas.sort(key=lambda x: abs(x - sol.gamma))
    for gamma in gammas:
        print(f'Seeking target: gamma={gamma}')
        my_sol = Solution(**{**start_params,
                             'gamma': gamma})
        print(f'Seeking {my_sol}')
        try:
            sol_search.seek(my_sol, min_step_size=1e-5,
                            verbose=True)
            print('Target found')
        except ValueError:
            print('Aborting seek.')

    gammas.sort()

    for gamma, alpha in product(gammas, alphas):
        print(f'Seeking target: alpha={alpha}, gamma={gamma}')
        my_sol = Solution(**{**start_params,
                             'alpha': alpha,
                             'gamma': gamma})
        print(f'Seeking {my_sol}')
        try:
            sol_search.seek(my_sol, min_step_size=1e-4,
                            verbose=True)
            print('Target found')
        except ValueError:
            print('Aborting seek.')
            continue

    print(f'Solutions found: {len(sol_search.solutions)}')
    plt.plot(*zip(*((sol.alpha, sol.gamma) for sol in sol_search.solutions)), '.')

    with open(FILE_NAME, 'wb') as f:
        pickle.dump(sol_search, f)
