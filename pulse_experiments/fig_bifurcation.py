"""Generate figures for pulse-width and pulse-speed for a varaiety of parameter
combinations. Use data generated from `data_bifurcation.py`.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

import experiment_defaults

from data_bifurcation import Solution, SolutionSearch

from itertools import product

DATA_FILE_NAME = os.path.join(
        experiment_defaults.data_path,
        'bifurcation.pickle')

with open(DATA_FILE_NAME, 'rb') as f:
    sol_search = pickle.load(f)

alphas = list(range(20, 7, -1))
gammas = [0.13, 0.14, 0.15, 0.17, 0.19]

plt.figure("Solutions found")
seed = sol_search.solutions[0]
plt.plot(seed.alpha, seed.gamma, 'm*', markersize=20)
plt.plot(*(zip(*product(alphas, gammas))), 'b*')
plt.plot(*zip(*((sol.alpha, sol.gamma) for sol in sol_search.solutions)), 'g.')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$\gamma$')
plt.title('Paramter combinations with found solutions.')

IMAGE_FILE_NAME = os.path.join(
        experiment_defaults.media_path,
        'bifurcation_params')
for ext in ['.eps', '.png']:
    plt.savefig(IMAGE_FILE_NAME + ext)


plt.figure('Speed by alpha')
for gamma in gammas:
    pairs = sorted((sol.alpha, sol.speed) for sol in sol_search.solutions
                   if sol.gamma == gamma)
    if len(pairs) > 0:
        plt.plot(*zip(*pairs), '.-', label=f'$\\gamma={gamma:.2f}$')

plt.xlabel(r'$\alpha$')
plt.ylabel(r'$c$')
plt.title('Speed by alpha')
plt.legend()
IMAGE_FILE_NAME = os.path.join(
        experiment_defaults.media_path,
        'bifurcation_alpha_speed')
for ext in ['.eps', '.png']:
    plt.savefig(IMAGE_FILE_NAME + ext)

plt.figure('Width by alpha')
for gamma in gammas:
    pairs = sorted((sol.alpha, sol.width) for sol in sol_search.solutions
                   if sol.gamma == gamma)
    if len(pairs) > 0:
        plt.plot(*zip(*pairs), '.-', label=f'$\\gamma={gamma:.2f}$')

plt.xlabel(r'$\alpha$')
plt.ylabel(r'$\Delta$')
plt.title('Width by alpha')
plt.legend()

IMAGE_FILE_NAME = os.path.join(
        experiment_defaults.media_path,
        'bifurcation_alpha_width')
for ext in ['.eps', '.png']:
    plt.savefig(IMAGE_FILE_NAME + ext)

plt.figure('Speed by gamma')
for alpha in alphas:
    pairs = sorted((sol.gamma, sol.speed) for sol in sol_search.solutions
                   if sol.alpha == alpha)
    if len(pairs) > 0:
        plt.plot(*zip(*pairs), '.-', label=f'$\\alpha={alpha}$')

plt.xlabel(r'$\gamma$')
plt.ylabel(r'$c$')
plt.title('Speed by gamma')
plt.legend()

IMAGE_FILE_NAME = os.path.join(
        experiment_defaults.media_path,
        'bifurcation_gamma_speed')
for ext in ['.eps', '.png']:
    plt.savefig(IMAGE_FILE_NAME + ext)

plt.figure('Width by gamma')
for alpha in alphas:
    pairs = sorted((sol.gamma, sol.width) for sol in sol_search.solutions
                   if sol.alpha == alpha)
    if len(pairs) > 0:
        plt.plot(*zip(*pairs), '.-', label=f'$\\alpha={alpha}$')

plt.xlabel(r'$\gamma$')
plt.ylabel(r'$\Delta$')
plt.title('Width by gamma')
plt.legend()

IMAGE_FILE_NAME = os.path.join(
        experiment_defaults.media_path,
        'bifurcation_gamma_width')
for ext in ['.eps', '.png']:
    plt.savefig(IMAGE_FILE_NAME + ext)
