#!/usr/bin/python3
'''
The wave response to the inpulse I(x,t) = epsilon * delta(t - 1)
simulated and compared to asymptotic approximation.
'''

import experiment_defaults

from functools import partial
from itertools import product
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


def main():

    file_name = 'speed_by_gamma'

    mu = 1.0
    alphas = [1.1, 2, 5, 10]
    colors = ['b', 'g', 'm', 'c']
    theta = 0.1

    plt.figure(figsize=(5, 3))

    gammas = np.linspace(0, 1, 20001)

    mins_curve_gammas = []
    mins_curve_speeds = []
    for alpha in np.logspace(-1, 1.2, 20001):
        gamma_mins = get_min_gamma(mu=mu, alpha=alpha, theta=theta)
        for gamma in gamma_mins:
            speed = get_speed(alpha=alpha, mu=mu, theta=theta, gamma=gamma)
            mins_curve_gammas.append(gamma)
            mins_curve_speeds.append(speed)

    mins_curve_gammas = np.array(mins_curve_gammas)
    mins_curve_speeds = np.array(mins_curve_speeds)

    nan_mask = np.logical_and(np.logical_not(np.isnan(mins_curve_gammas)),
                              np.logical_not(np.isnan(mins_curve_speeds)))

    mins_curve_speeds[mins_curve_speeds < 0] = np.nan

    # plt.plot(mins_curve_gammas, mins_curve_speeds, 'k.', markersize=1)
    plt.plot(mins_curve_gammas[nan_mask],
             mins_curve_speeds[nan_mask],
             'k-', label='bifurcation')

    for alpha, color in zip(alphas, colors):
        gamma_mins = get_min_gamma(mu=mu, alpha=alpha, theta=theta)
        speeds = get_speed(alpha=alpha, mu=mu, theta=theta, gamma=gammas)
        speeds[speeds < 0] = np.nan
        speeds2 = get_speed2(alpha=alpha, mu=mu, theta=theta, gamma=gammas)
        speeds2[speeds2 < 0] = np.nan
        plt.plot(gammas, speeds, '-', color=color, label=f'$\\tau_q={alpha}$')
        plt.plot(gammas, speeds2, ':', color=color)
        for gamma in gamma_mins:
            desc = ((2*theta-1)*alpha + 2*theta*mu/gamma)**2 \
                     - 8*theta*mu*alpha * (2*theta/gamma - 1)
            assert np.all(np.abs(desc) < 1e-10)
            speed = get_speed(alpha=alpha, mu=mu, theta=theta,
                              gamma=gamma, desc=0)

    regressive_gammas = np.linspace(theta, 2*theta, 201)
    plt.plot(regressive_gammas,
             regressive_speed(mu=mu, theta=theta, gamma=regressive_gammas),
             'r-',
             label='regressive')

    # plt.plot([theta]*2, [-5, 5], 'k:', label='$\\theta$')
    plt.plot([2*theta]*2, [-5, 5], 'k:', label='$\\gamma = 2\\theta$')
    plt.fill_between([-1, theta], 2*[5], 2*[-5], color='grey', label='$\\gamma < \\theta$')
    plt.plot(2*theta, 0, 'k*', markersize=10)

    plt.legend(loc='lower right')
    plt.title('$c$ vs. $\\gamma$')
    plt.xlabel('$\\gamma$')
    plt.ylabel('$c$')
    plt.xlim(.04, 1)
    plt.ylim(-4.1, 4.1)

    plt.tight_layout()

    for extension in ['.png', '.eps']:
        plt.savefig(os.path.join(experiment_defaults.media_path,
                                 file_name + extension), dpi=300)
    plt.show()


if __name__ == '__main__':
    main()
