import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import os
import pickle
import warnings

from tqdm import tqdm
from itertools import cycle

from data_speed_width_bifurcation import SolutionSearch, Solution, Arguments, MaxRefinementReachedException

import experiment_defaults

DATA_FILE = os.path.join(
    experiment_defaults.data_path,
    'smooth_bifurcation.pickle')

plt.rc('font', size=15)


with open(DATA_FILE, 'rb') as f:
    alphas, gammas, stable_search, unstable_search = pickle.load(f)

alphas.sort()
gammas.sort()

plt.figure()
# for gamma in plot_gammas:
#     plt.plot(alphas, gamma+0*alphas, 'k-')
plt.title('Numerical Data')
plt.plot(*zip(*((sol.alpha, sol.gamma) for sol in unstable_search)), 'mx')
plt.plot(*zip(*((sol.alpha, sol.gamma) for sol in stable_search)), 'g.')
plt.show()

# color info and defaults
plot_gammas = np.arange(0.2, 0.125, -0.005)
plot_alphas = np.arange(20.0, 11.0, -.5)
cmap = matplotlib.cm.get_cmap('viridis')
gamma_color_norm = matplotlib.colors.Normalize(vmin=0.1, vmax=0.2)
gamma_colors = [cmap(gamma_color_norm(gamma)) for gamma in plot_gammas]
alpha_color_norm = matplotlib.colors.Normalize(vmin=11, vmax=20)
alpha_colors = [cmap(alpha_color_norm(alpha)) for alpha in plot_alphas]

boundary_color = 'gold'
style_legend_color = 'grey'
tol = 1e-8

# speed by alpha
SPEED_BY_ALPHA_FILE_NAME = 'bifurcation_alpha_speed'
plt.figure()
stable_search.solutions.sort(key=lambda sol: sol.alpha)
unstable_search.solutions.sort(key=lambda sol: sol.alpha)
plt.plot([], [], linestyle='-', color=style_legend_color, label='Stable')
plt.plot([], [], linestyle=':', color=style_legend_color, label='Unstable')
for gamma, color in zip(plot_gammas, cycle(gamma_colors)):
    if gamma == 0.2:
        fill_alpha, fill_speed = zip(*[(sol.alpha, sol.speed)
                                       for sol in stable_search
                                       if abs(sol.gamma-gamma) < tol])
        plt.fill_between(fill_alpha, fill_speed, [100]*len(fill_alpha), color='gray')
        fill_alpha, fill_speed2 = zip(*[(sol.alpha, sol.speed)
                                        for sol in unstable_search
                                        if abs(sol.gamma-gamma) < tol])
        plt.fill_between(fill_alpha, [-100]*len(fill_alpha), fill_speed2, color='gray')
        plt.text(10.5, 0.15, 'Front Regime', color='white')
    plt.plot(*zip(*[(sol.alpha, sol.speed)
                    for sol in stable_search
                    if abs(sol.gamma-gamma) < tol]),
             linestyle='-',
             color=color)
    plt.plot(*zip(*[(sol.alpha, sol.speed)
                    for sol in unstable_search
                    if abs(sol.gamma-gamma) < tol]),
             linestyle=':',
             color=color)
plt.xlabel(r'$\tau_q$')
plt.ylabel('speed')
plt.xlim(min(sol. alpha for sol in stable_search), 20)
plt.ylim(0.1, 1.2)
plt.title('Pulse speed vs $\\tau_q$')
plt.legend(loc='upper left')
plt.colorbar(matplotlib.cm.ScalarMappable(norm=gamma_color_norm, cmap=cmap),
             label='$\\gamma$')
for ext in ['png', 'eps', 'pdf']:
    plt.savefig(os.path.join(experiment_defaults.media_path,
                             SPEED_BY_ALPHA_FILE_NAME) + '.' + ext)
plt.show()

# width by alpha
WIDTH_BY_ALPHA_FILE_NAME = 'bifurcation_alpha_width'
plt.figure()
stable_search.solutions.sort(key=lambda sol: sol.alpha)
unstable_search.solutions.sort(key=lambda sol: sol.alpha)
plt.plot([], [], linestyle='-', color=style_legend_color, label='Stable')
plt.plot([], [], linestyle=':', color=style_legend_color, label='Unstable')
for gamma, color in zip(plot_gammas, cycle(gamma_colors)):
    if gamma == 0.2:
        fill_alpha, fill_speed = zip(*[(sol.alpha, sol.width)
                                       for sol in stable_search
                                       if abs(sol.gamma-gamma) < tol])
        plt.fill_between(fill_alpha, fill_speed, [100]*len(fill_alpha), color='gray')
        fill_alpha, fill_speed2 = zip(*[(sol.alpha, sol.width)
                                        for sol in unstable_search
                                        if abs(sol.gamma-gamma) < tol])
        plt.fill_between(fill_alpha, [-100]*len(fill_alpha), fill_speed2, color='gray')
        plt.text(10.5, 1.25, 'Front Regime', color='white')
    plt.plot(*zip(*[(sol.alpha, sol.width)
                    for sol in stable_search
                    if abs(sol.gamma-gamma) < tol]),
             linestyle='-',
             color=color)
    plt.plot(*zip(*[(sol.alpha, sol.width)
                    for sol in unstable_search
                    if abs(sol.gamma-gamma) < tol]),
             linestyle=':',
             color=color)
plt.xlabel(r'$\tau_q$')
plt.ylabel('width')
plt.xlim(min(sol. alpha for sol in stable_search), 20)
plt.ylim(0, 14)
plt.title('Pulse width vs $\\tau_q$')
plt.legend()
plt.colorbar(matplotlib.cm.ScalarMappable(norm=gamma_color_norm, cmap=cmap),
             label='$\\gamma$')
for ext in ['png', 'eps', 'pdf']:
    plt.savefig(os.path.join(experiment_defaults.media_path,
                             WIDTH_BY_ALPHA_FILE_NAME) + '.' + ext)
plt.show()

# speed by gamma
SPEED_BY_GAMMA_FILE_NAME = 'bifurcation_gamma_speed'
plt.figure()
stable_search.solutions.sort(key=lambda sol: sol.gamma)
unstable_search.solutions.sort(key=lambda sol: sol.gamma)
plt.plot([], [], linestyle='-', color=style_legend_color, label='Stable')
plt.plot([], [], linestyle=':', color=style_legend_color, label='Unstable')
for alpha, color in zip(plot_alphas, cycle(alpha_colors)):
    plt.plot(*zip(*[(sol.gamma, sol.speed)
                    for sol in stable_search
                    if abs(sol.alpha-alpha) < tol]),
             linestyle='-',
             color=color)
    plt.plot(*zip(*[(sol.gamma, sol.speed)
                    for sol in unstable_search
                    if abs(sol.alpha-alpha) < tol]),
             linestyle=':',
             color=color)
plt.fill_between([.2, .4], [-10]*2, [10]*2, color='gray')
plt.text(.205, .4, 'Front Regime', color='white', rotation=90)
plt.xlabel(r'$\gamma$')
plt.ylabel('speed')
plt.xlim(.12, .22)
plt.ylim(.1, 1.2)
plt.title('Pulse speed vs $\\gamma$')
plt.legend(loc='upper left')
plt.colorbar(matplotlib.cm.ScalarMappable(norm=alpha_color_norm, cmap=cmap),
             label='$\\tau_q$')
fsor ext in ['png', 'eps', 'pdf']:
    plt.savefig(os.path.join(experiment_defaults.media_path,
                             SPEED_BY_GAMMA_FILE_NAME) + '.' + ext)
plt.show()


# width by gamma
WIDTH_BY_GAMMA_FILE_NAME = 'bifurcation_gamma_width'
plt.figure()
stable_search.solutions.sort(key=lambda sol: sol.gamma)
unstable_search.solutions.sort(key=lambda sol: sol.gamma)
plt.plot([], [], linestyle='-', color=style_legend_color, label='Stable')
plt.plot([], [], linestyle=':', color=style_legend_color, label='Unstable')
for alpha, color in zip(plot_alphas, cycle(alpha_colors)):
    plt.plot(*zip(*[(sol.gamma, sol.width)
                    for sol in stable_search
                    if abs(sol.alpha-alpha) < tol]),
             linestyle='-',
             color=color)
    plt.plot(*zip(*[(sol.gamma, sol.width)
                    for sol in unstable_search
                    if abs(sol.alpha-alpha) < tol]),
             linestyle=':',
             color=color)
plt.fill_between([.2, .4], [-100]*2, [100]*2, color='gray')
plt.text(.205, 5, 'Front Regime', color='white', rotation=90)
plt.xlabel(r'$\gamma$')
plt.ylabel('width')
plt.xlim(.12, .22)
plt.ylim(1, 15)
plt.title('Pulse width vs $\\gamma$')
plt.legend(loc='upper left')
plt.colorbar(matplotlib.cm.ScalarMappable(norm=alpha_color_norm, cmap=cmap),
             label='$\\tau_q$')
for ext in ['png', 'eps', 'pdf']:
    plt.savefig(os.path.join(experiment_defaults.media_path,
                             WIDTH_BY_GAMMA_FILE_NAME) + '.' + ext)
plt.show()

# search code blocks
default_params = {"mu": 1.0, "alpha": 20.0, "theta": 0.2, "beta": 5.0}

if False:
    step_size = 1e-2
    refinements = 4
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for gamma in tqdm(plot_gammas, position=0, leave=True):
            alpha_min = min(sol.alpha for sol in stable_search
                            if abs(sol.gamma - gamma) < 1e-10) *.97
            alpha_min = alphas[np.argmin(np.abs([alpha-alpha_min for alpha in alphas]))]
            for alpha in tqdm(np.arange(alpha_min, alpha_min*.97, -0.01), position=1, leave=True):
                print(f'\nalpha: {alpha}, gamma: {gamma}')
                beta = round(1/gamma-1, 10)
                target = Arguments(**{**default_params, 'alpha': alpha, 'beta': beta})
                try:
                    sol = stable_search.seek(target, step_size=step_size,
                                             max_refinements=refinements,
                                             search_verbose=True)
                except la.LinAlgError:
                    print('\nFailed')
                    break
                except MaxRefinementReachedException:
                    print('\nFailed')
                    break
                finally:
                    plt.plot(sol.alpha, sol.gamma, 'k.')
                    plt.pause(1e-4)

if False:
    for searcher, plot_style in [(stable_search, 'bv'),
                                 (unstable_search, 'c^')]:
        # searcher = unstable_search
        # plot_style = 'bv'
        step_size = 1e-2
        refinements = 10
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # for gamma in tqdm([.135, .145], position=0, leave=True):
            for gamma in tqdm(plot_gammas, position=0, leave=True):
                alpha_min = min(sol.alpha for sol in searcher
                                if abs(sol.gamma - gamma) < 1e-10)
                alpha_min = alphas[np.argmin(np.abs([alpha-alpha_min for alpha in alphas]))]
                print(gamma, alpha_min, '\n'*3)
                for alpha in tqdm(np.arange(alpha_min, alpha_min-4e-2, -1e-2), position=1):
                # for alpha in tqdm(np.arange(18.5, 18.4, -.01), position=1):
                    beta = round(1/gamma-1, 10)
                    target = Arguments(**{**default_params, 'alpha': alpha, 'beta': beta})
                    try:
                        sol = searcher.seek(target, step_size=step_size,
                                            max_refinements=refinements,
                                            search_verbose=False)
                    except la.LinAlgError:
                        break
                    except MaxRefinementReachedException:
                        break
                    finally:
                        plt.plot(searcher.solutions[-1].alpha,
                                 searcher.solutions[-1].gamma,
                                 plot_style)
                        plt.pause(1e-4)

if False:
    with open(DATA_FILE, 'wb') as f:
        pickle.dump((alphas, gammas, stable_search, unstable_search), f)

