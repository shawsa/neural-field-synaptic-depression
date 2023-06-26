import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from itertools import cycle

from data_speed_width_bifurcation import SolutionSearch, Solution, Arguments

import experiment_defaults

FILE = os.path.join(
    experiment_defaults.data_path,
    'smooth_bifurcation.pickle')


with open(FILE , 'rb') as f:
    alphas, gammas, stable_search, unstable_search = pickle.load(f)

alphas.sort()
gammas.sort()

plt.figure('Numerical Data')
plt.plot(*zip(*((sol.alpha, sol.gamma) for sol in unstable_search)), 'mx')
plt.plot(*zip(*((sol.alpha, sol.gamma) for sol in stable_search)), 'g.')

color_norm = matplotlib.colors.Normalize(vmin=0.1, vmax=0.2)
cmap = matplotlib.cm.get_cmap('winter')
colors = [cmap(color_norm(gamma)) for gamma in gammas]

plt.figure('Pulse speed by alpha')
# plot numerical results
tol = 1e-12

stable_search.solutions.sort(key=lambda sol: sol.alpha)
unstable_search.solutions.sort(key=lambda sol: sol.alpha)
for gamma, color in zip(gammas, cycle(colors)):
    plt.plot(*zip(*[(sol.alpha, sol.speed)
                     for sol in stable_search
                     if abs(sol.gamma-gamma)<tol]),
             linestyle='-',
             color=color)
    plt.plot(*zip(*[(sol.alpha, sol.speed)
                     for sol in unstable_search
                     if abs(sol.gamma-gamma)<tol]),
             linestyle=':',
             color=color)
plt.xlabel(r'$\tau_q$')
plt.ylabel('speed')
plt.show()

plt.figure('Speed by gamma')
stable_search.solutions.sort(key=lambda sol: sol.gamma)
unstable_search.solutions.sort(key=lambda sol: sol.gamma)
for alpha, color in zip(alphas[::10], cycle(colors)):
    plt.plot(*zip(*[(sol.gamma, sol.speed)
                     for sol in stable_search
                     if abs(sol.alpha-alpha)<tol]),
             linestyle='-',
             color=color)
    plt.plot(*zip(*[(sol.gamma, sol.speed)
                     for sol in unstable_search
                     if abs(sol.alpha-alpha)<tol]),
             linestyle=':',
             color=color)
plt.xlabel(r'$gamma$')
plt.ylabel('speed')
plt.show()
