#!/usr/bin/python3
'''
The wave response to the inpulse I(x,t) = epsilon * delta(t - 1)
simulated and compared to asymptotic approximation.
'''

import experiment_defaults

from functools import partial
from itertools import product
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os.path
from tqdm import tqdm

def get_min_gamma(*, mu, alpha, theta):
    A = (2*theta*mu)**2
    B = -4*theta*mu*alpha*(1 + 2*theta)
    C = 8*theta*mu*alpha + (2*theta - 1)**2*alpha**2
    return [2*A/(-B + sign*np.sqrt(B**2 - 4*A*C))
            for sign in [1, -1]]


def get_speed(*, mu, alpha, theta, gamma, desc=None):
    A = 2*theta*mu*alpha
    B = (2*theta-1)*alpha + 2*theta*mu/gamma
    C = 2*theta/gamma - 1
    if desc is None:
        desc = B**2 - 4*A*C
    return (-B + np.sqrt(desc))/(2*A)

def get_speed2(*, mu, alpha, theta, gamma, desc=None):
    A = 2*theta*mu*alpha
    B = (2*theta-1)*alpha + 2*theta*mu/gamma
    C = 2*theta/gamma - 1
    if desc is None:
        desc = B**2 - 4*A*C
    return (-B - np.sqrt(desc))/(2*A)


def regressive_speed(*, mu, theta, gamma):
    return (1 - theta/(gamma - theta))/2/mu

def alpha_crit(gamma, mu, theta):
    return 2*mu*theta*(-2*gamma + 2*theta + 2*np.sqrt(gamma**2 - 2*gamma*theta - gamma + 2*theta) + 1)/(gamma*(4*theta**2 - 4*theta + 1))


FILE_NAME = 'speed_by_tau_q'
mu = 1.0
theta = 0.1

regressive_linspace = np.linspace(0.1, 0.2, 5)
gammas = list(regressive_linspace) + \
    [.3, .5, 1]
color_norm = matplotlib.colors.Normalize(vmin=0.1, vmax=0.2)
cmap = matplotlib.cm.get_cmap('winter')

colors = [cmap(color_norm(gamma)) for gamma in regressive_linspace] + \
         ['g', 'gold', 'red']

plt.figure(figsize=(5, 3))

gamma_linspace = np.linspace(2*theta, theta, 201)
alpha_crit_arr = alpha_crit(gamma_linspace, mu, theta)
speed_crit_arr = get_speed(
        mu=mu,
        alpha=alpha_crit_arr,
        theta=theta,
        gamma=gamma_linspace,
        desc=0)


alphas = np.linspace(0, 20, 2001)

for gamma, color in zip(gammas, colors):
    speeds = get_speed(alpha=alphas, mu=mu, theta=theta, gamma=gamma)
    speeds[speeds < 0] = np.nan
    speeds2 = get_speed2(alpha=alphas, mu=mu, theta=theta, gamma=gamma)
    speeds2[speeds2 < 0] = np.nan
    if gamma == theta:
        mask = np.logical_not(np.isnan(speeds))
        plt.fill_between(alphas[mask],
                         speeds2[mask],
                         speeds[mask],
                         color='grey')# ,label='$\\gamma < \\theta$')
    if gamma == theta or gamma >= 2*theta:
        plt.plot(alphas, speeds, '-', color=color, label=f'$\\gamma={gamma}$')
        plt.plot(alphas, speeds2, ':', color=color)
    else:
        plt.plot(alphas, speeds, '-', color=color)
        plt.plot(alphas, speeds2, ':', color=color)
    if gamma != theta and gamma < 2*theta:
        plt.plot(alphas,
                 len(alphas)*[regressive_speed(mu=mu, theta=theta, gamma=gamma)],
                 '--', color=color)

plt.plot(alpha_crit_arr[0], 0, 'k*')
plt.plot(alpha_crit_arr, speed_crit_arr, 'k-')# , label='bifurcation')
plt.text(10, 1, 'Pulse Regime', color='w')
plt.text(1, 3, 'stable')
plt.text(2, .05, 'unstable')
plt.text(1, -1, 'regressive')
# plt.title('$c$ vs. $\\tau_q$')
# plt.plot([], [], 'k-', label='Stable')
# plt.plot([], [], 'k:', label='Unstable')
# plt.plot([], [], 'k--', label='Regressive')
plt.title('Front Bifurcation')
plt.xlabel('synaptic efficacy timescale ($\\tau_q$)')
plt.ylabel('speed')
plt.xlim(0, 20)
plt.ylim(-2, 4.5)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.tight_layout()

for extension in ['.png', '.eps']:
    plt.savefig(os.path.join(experiment_defaults.media_path,
                             FILE_NAME + extension), dpi=300)
plt.show()
