import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import pickle
import sympy as sym

import experiment_defaults
from helper_symbolics import generate_Newton_args_for_speed_width
from newton import MaxIterationsReachedException, NewtonRootFind

from tqdm import tqdm
from itertools import product

import warnings

DATA_FILE = os.path.join(
    experiment_defaults.data_path,
    'smooth_bifurcation.pickle')


def convert_beta(alpha, beta):
    """Converts to old params. The helper_symbolics files has not been
    updated to the new params via alpha*beta -> beta. We must convert
    back to use this module correctly.
    """
    return beta / alpha


class Arguments:
    def __init__(self,
                 mu: float,
                 alpha: float,
                 theta: float,
                 beta: float):
        self.mu = mu
        self.alpha = alpha
        self.theta = theta
        self.beta = beta

    @property
    def gamma(self):
        return 1/(1+self.beta)

    @property
    def dict(self):
        return {
            "mu": self.mu,
            "alpha": self.alpha,
            "theta": self.theta,
            "beta": self.beta,
        }

    @property
    def dict_old_beta(self):
        return {**self.dict, "beta": self.beta/self.alpha}

    def __repr__(self):
        return f"Arguments: {self.dict}"

    @property
    def vec(self):
        return np.array([
            self.mu,
            self.alpha,
            self.theta,
            self.beta
        ])

    def dist(self, target):
        return la.norm(self.vec - target.vec)

    def __eq__(self, target):
        return np.all(self.vec == target.vec)

class Solution:
    def __init__(self, *, args: Arguments, speed_guess, width_guess, max_iterations=10):
        self.args = args
        self.max_iterations = max_iterations
        self.solved = False
        self.tol = None
        self.speed = speed_guess
        self.width = width_guess

    @property
    def mu(self):
        return self.args.mu

    @property
    def alpha(self):
        return self.args.alpha

    @property
    def theta(self):
        return self.args.theta

    @property
    def beta(self):
        return self.args.beta

    @property
    def gamma(self):
        return self.args.gamma

    @property
    def dict(self):
        if self.solved:
            return {
                "speed": self.speed,
                "width": self.width,
                **self.args.dict,
            }
        return {
            **self.args.dict,
            "solved": self.solved
        }

    def __repr__(self):
        return f"Solution: {self.dict}"

    @property
    def sol_vec(self):
        return np.array([
            self.speed,
            self.width
        ])

    def solve(self, tol=1e-12, verbose=False):
        assert tol > 0
        if self.solved and self.tol is not None and self.tol >= tol:
            # already solved, so do nothing.
            if verbose:
                print('Already solved.')
            return self.speed, self.width
        F, J = generate_Newton_args_for_speed_width(self.args.dict_old_beta)
        newton = NewtonRootFind(F, J,
                                max_iterations=self.max_iterations,
                                verbose=verbose)
        self.speed, self.width = newton.cauchy_tol(self.sol_vec, tol=tol)
        self.tol = tol
        self.solved = True
        return self.speed, self.width


class MaxRefinementReachedException(Exception):
    pass


class SolutionSearch:

    def __init__(self, seed: Solution):
        assert seed.solved
        self.solutions = [seed]

    def __repr__(self):
        return f'Solution Search: solutions={len(self.solutions)}'

    def __iter__(self):
        yield from self.solutions

    def closest_to(self, target: Arguments):
        sol = self.solutions[0]
        dist = sol.args.dist(target)
        for sol1 in self.solutions:
            dist1 = sol1.args.dist(target)
            if dist1 < dist:
                dist, sol = dist1, sol1
        return sol

    def seek(self,
             target: Arguments,
             start: Arguments = None,
             tol=1e-12,
             step_size=float('inf'),
             max_refinements=4,
             search_verbose=False,
             solver_verbose=False):
        if search_verbose:
            print(f'Seeking {target}')
        refinements = 0
        if start is None:
            sol = self.closest_to(target)
        else:
            sol = self.closest_to(start)
        direction = target.vec - sol.args.vec
        if la.norm(direction) < 1e-15:
            return sol
        direction = direction/la.norm(direction)
        while True:
            if refinements >= max_refinements:
                raise MaxRefinementReachedException()
            if step_size > sol.args.dist(target):
                step_size = sol.args.dist(target)

            new_sol = Solution(args=Arguments(*(sol.args.vec + step_size*direction)),
                               speed_guess=sol.speed,
                               width_guess=sol.width)
            try:
                if search_verbose:
                    print(f'Solving: {new_sol.args}')
                new_sol.solve(verbose=solver_verbose)
                if any(map(np.isnan, [new_sol.speed, new_sol.width])):
                    raise MaxIterationsReachedException
            except MaxIterationsReachedException:
                step_size /= 2
                if search_verbose:
                    print(f'Failed... refining step size to {step_size}')
                refinements += 1
                continue
            if search_verbose:
                print(f'Solution found: {new_sol}')
            sol = new_sol
            self.solutions.append(sol)
            if sol.args == target:
                return sol


if __name__ == "__main__":
    if True:
        """For REPL loading of extant data."""
        with open(DATA_FILE , 'rb') as f:
            alphas, gammas, stable_search, unstable_search = pickle.load(f)

    default_params = { "mu": 1.0,
            "alpha": 20.0,
            "theta": 0.2,
            "beta": 5.0
    }

    alphas = np.arange(20.0, 10.0, -0.05)
    gammas = list(np.arange(0, 0.2, 0.005)) + [0.2, 1/6]
    # gammas.sort(key = lambda x: abs(x-1/6))
    gammas.sort()
    gammas = gammas[::-1]

    stable_seed = Solution(args=Arguments(**default_params),
                           speed_guess=1.2,
                           width_guess=12)
    print('Solving for stable seed')
    stable_seed.solve(verbose=True)

    # stable search
    if not 'stable_search' in locals():
        stable_search = SolutionSearch(stable_seed)
    failed_stable_seeks = []


    refinements = 2
    step_size = 0.05

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # for gamma in tqdm(gammas, position=0, leave=True):
        for gamma in tqdm([0.2], position=0, leave=True):
            for alpha in tqdm(alphas, position=1, leave=False):
                beta = round(1/gamma-1, 10)
                target = Arguments(**{**default_params, 'alpha': alpha, 'beta': beta})
                try:
                    stable_search.seek(target, step_size=step_size, max_refinements=refinements)
                except (MaxRefinementReachedException, la.LinAlgError):
                    failed_stable_seeks.append(target)
                    break

    # unstable search
    unstable_seed = Solution(args=Arguments(mu=1.0, alpha=20.0, theta=0.2, beta=5.0),
                             speed_guess=0.227,
                             width_guess=1.76)
    print('Solving for unstable seed')
    unstable_seed.solve(verbose=True)
    if not 'unstable_search' in locals():
        unstable_search = SolutionSearch(unstable_seed)
    failed_unstable_seeks = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for alpha in tqdm(alphas, position=0):
            for gamma in tqdm(gammas, position=1, leave=False):
                beta = round(1/gamma-1, 10)
                target = Arguments(**{**default_params, 'alpha': alpha, 'beta': beta})
                try:
                    unstable_search.seek(target, step_size=step_size, max_refinements=refinements)
                except la.LinAlgError:
                    failed_unstable_seeks.append(target)
                    continue
                except MaxRefinementReachedException:
                    failed_unstable_seeks.append(target)
                    # break
                    continue

    # for gamma in tqdm(np.linspace(0.13, 0.14, 21), position=0):
    #     for alpha in tqdm(np.linspace(18.0, 17, 11), position=1, leave=False):
    #         beta = round(1/gamma-1, 10)
    #         target = Arguments(**{**default_params, 'alpha': alpha, 'beta': beta})
    #             try:
    #                 unstable_search.seek(target, step_size=.1, max_refinements=refinements)
    #             except la.LinAlgError:
    #                 failed_unstable_seeks.append(target)
    #                 continue
    #             except MaxRefinementReachedException:
    #                 failed_unstable_seeks.append(target)


    # plotting
    plt.figure('Stable branch alpha')
    plt.plot([sol.alpha for sol in stable_search],
             [sol.speed for sol in stable_search],
             'b.')
    plt.xlabel(r'$\tau_q$')
    plt.ylabel('speed')
    plt.show()

    # plotting
    plt.figure('Stable branch gamma')
    plt.plot([sol.gamma for sol in stable_search],
             [sol.speed for sol in stable_search],
             'b.')
    plt.xlabel(r'$\gamma$')
    plt.ylabel('speed')
    plt.show()


    # plotting
    plt.figure('unstable branch')
    plt.plot([sol.alpha for sol in unstable_search],
             [sol.speed for sol in unstable_search],
             'b.')
    plt.xlabel(r'$\tau_q$')
    plt.ylabel('speed')
    plt.show()

    plt.figure('parameter space stable')
    plt.plot(*zip(*[(sol.alpha, sol.gamma) for sol in stable_search]), 'b.')

    plt.figure('parameter space unstable')
    plt.plot(*zip(*[(sol.alpha, sol.gamma) for sol in unstable_search]), 'g.')
    plt.plot(*zip(*[(sol.alpha, sol.gamma) for sol in failed_unstable_seeks]), 'mx')

    with open(DATA_FILE, 'wb') as f:
        pickle.dump((alphas, gammas, stable_search, unstable_search), f)

    if False:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for gamma in tqdm(plot_gammas, position=0, leave=True):
                for alpha in tqdm(alphas, position=1):
                    step_size = 1e-2
                    refinements = 4
                    beta = round(1/gamma-1, 10)
                    target = Arguments(**{**default_params, 'alpha': alpha, 'beta': beta})
                    try:
                        sol = unstable_search.seek(target, step_size=step_size, max_refinements=refinements)
                        plt.plot(sol.alpha, sol.gamma, 'kv')
                    except la.LinAlgError:
                        continue
                    except MaxRefinementReachedException:
                        continue

