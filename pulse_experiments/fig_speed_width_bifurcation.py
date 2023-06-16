import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import sympy as sym

import experiment_defaults
from helper_symbolics import generate_Newton_args_for_speed_width
from newton import MaxIterationsReachedException, NewtonRootFind


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

    def seek(self, target: Arguments, tol=1e-12,
             step_size=float('inf'), max_refinements=4,
             verbose=False):
        print(f'Seeking {target}')
        refinements = 0
        sol = self.closest_to(target)
        direction = target.vec - sol.args.vec
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
                print(f'Solving: {new_sol.args}')
                new_sol.solve(verbose=verbose)
                if any(map(np.isnan, [new_sol.speed, new_sol.width])):
                    raise MaxIterationsReachedException
            except MaxIterationsReachedException:
                step_size /= 2
                print(f'Failed... refining step size to {step_size}')
                refinements += 1
                continue
            print(f'Solution found: {new_sol}')
            sol = new_sol
            self.solutions.append(sol)
            if sol.args == target:
                return sol


if __name__ == "__main__":
    params = {
            "mu": 1.0,
            "alpha": 20.0,
            "theta": 0.2,
            "beta": 5.0
    }
    # stable search
    sol = Solution(args=Arguments(mu=1.0, alpha=20.0, theta=0.2, beta=5.0),
                   speed_guess=1.2,
                   width_guess=12)
    target = Arguments(mu=1.0, alpha=2.0, theta=0.2, beta=5.0)
    sol.solve(verbose=True)
    stable_search = SolutionSearch(sol)
    stable_search.seek(target, step_size=1.0, max_refinements=7)

    # unstable search
    sol = Solution(args=Arguments(mu=1.0, alpha=20.0, theta=0.2, beta=5.0),
                   speed_guess=0.227,
                   width_guess=1.76)
    target = Arguments(mu=1.0, alpha=2.0, theta=0.2, beta=5.0)
    sol.solve(verbose=True)
    unstable_search = SolutionSearch(sol)
    unstable_search.seek(target, step_size=1.0, max_refinements=7)

    # plotting
    plt.figure()
    plt.plot([sol.alpha for sol in stable_search],
             [sol.speed for sol in stable_search],
             'b-')
    plt.plot([sol.alpha for sol in unstable_search],
             [sol.speed for sol in unstable_search],
             'b--')
    plt.xlabel(r'$\tau_q$')
    plt.ylabel('speed')
    plt.show()
