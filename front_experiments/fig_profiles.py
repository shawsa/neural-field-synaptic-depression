#!/usr/bin/python3

import experiment_defaults
import os.path
import matplotlib.pyplot as plt
import numpy as np
from neural_field import Parameters
from plotting_styles import Q_style, U_style, solution_styles, threshold_style

from front_profile_helpers import (Q_progressive, Q_regressive, U_progressvie,
                                   U_regressive, get_speed_lower,
                                   get_speed_regressive, get_speed_upper)


file_name = 'front_profiles'
fig_sizes = (5, 3)

theta = 0.1

xs = np.linspace(-50, 50, 201)

params = Parameters(mu=1.0, alpha=20.0, gamma=0.15)

fig = plt.figure(figsize=fig_sizes)

plt.plot([], [], 'k-', label='$U$')
plt.plot([], [], 'k--', label='$Q$')
plt.title('')

c_neg = get_speed_regressive(mu=params.mu,
                             alpha=params.alpha,
                             theta=theta,
                             gamma=params.gamma)
c_hi = get_speed_upper(mu=params.mu,
                       alpha=params.alpha,
                       theta=theta,
                       gamma=params.gamma)
c_lo = get_speed_lower(mu=params.mu,
                       alpha=params.alpha,
                       theta=theta,
                       gamma=params.gamma)

plt.plot(xs, U_regressive(xs,
                          mu=params.mu,
                          alpha=params.alpha,
                          gamma=params.gamma,
                          theta=theta,
                          c=c_neg), 'r-', label=f'$c={c_neg:.2f}$')
plt.plot(xs, Q_regressive(xs,
                          mu=params.mu,
                          alpha=params.alpha,
                          gamma=params.gamma,
                          theta=theta,
                          c=c_neg), 'r--')
plt.plot(xs, U_progressvie(xs,
                           mu=params.mu,
                           alpha=params.alpha,
                           gamma=params.gamma,
                           theta=theta,
                           c=c_lo), 'g-', label=f'$c={c_lo:.2f}$')
plt.plot(xs, Q_progressive(xs,
                           mu=params.mu,
                           alpha=params.alpha,
                           gamma=params.gamma,
                           theta=theta,
                           c=c_lo), 'g--')
plt.plot(xs, U_progressvie(xs,
                           mu=params.mu,
                           alpha=params.alpha,
                           gamma=params.gamma,
                           theta=theta,
                           c=c_hi), 'b-', label=f'$c={c_hi:.2f}$')
plt.plot(xs, Q_progressive(xs,
                           mu=params.mu,
                           alpha=params.alpha,
                           gamma=params.gamma,
                           theta=theta,
                           c=c_hi), 'b--')
plt.legend() 
plt.xlabel('$\\xi$')
plt.tight_layout()
for extension in ['.png', '.eps']:
    plt.savefig(os.path.join(experiment_defaults.media_path,
                             file_name + extension))

# progressive stable
fig = plt.figure(figsize=fig_sizes)
plt.plot(xs, U_progressvie(xs,
                           mu=params.mu,
                           alpha=params.alpha,
                           gamma=params.gamma,
                           theta=theta,
                           c=c_hi), 'b-', label=f'$c={c_hi:.2f}$')
plt.plot(xs, Q_progressive(xs,
                           mu=params.mu,
                           alpha=params.alpha,
                           gamma=params.gamma,
                           theta=theta,
                           c=c_hi), 'b--')

plt.text(10, 0.5, 'Fast Stable')
plt.legend()
plt.tight_layout()
plt.xlabel('$\\xi$')
for extension in ['.png', '.eps']:
    plt.savefig(os.path.join(experiment_defaults.media_path,
                             file_name + '_progressive' + extension))
plt.close()

# regressive
fig = plt.figure(figsize=fig_sizes)
plt.plot(xs, U_regressive(xs,
                          mu=params.mu,
                          alpha=params.alpha,
                          gamma=params.gamma,
                          theta=theta,
                          c=c_neg), 'r-', label=f'$c={c_neg:.2f}$')
plt.plot(xs, Q_regressive(xs,
                          mu=params.mu,
                          alpha=params.alpha,
                          gamma=params.gamma,
                          theta=theta,
                          c=c_neg), 'r--')

plt.text(10, 0.5, 'Regressive Stable')
plt.legend()
plt.tight_layout()
plt.xlabel('$\\xi$')
for extension in ['.png', '.eps']:
    plt.savefig(os.path.join(experiment_defaults.media_path,
                             file_name + '_regressive' + extension))
plt.close()

# progressive unstable
fig = plt.figure(figsize=fig_sizes)
plt.plot(xs, U_progressvie(xs,
                           mu=params.mu,
                           alpha=params.alpha,
                           gamma=params.gamma,
                           theta=theta,
                           c=c_lo), 'g-', label=f'$c={c_lo:.2f}$')
plt.plot(xs, Q_progressive(xs,
                           mu=params.mu,
                           alpha=params.alpha,
                           gamma=params.gamma,
                           theta=theta,
                           c=c_lo), 'g--')
plt.text(10, .5, 'Slow Unstable')
plt.legend()
plt.tight_layout()
plt.xlabel('$\\xi$')
for extension in ['.png', '.eps']:
    plt.savefig(os.path.join(experiment_defaults.media_path,
                             file_name + '_unstable' + extension))
plt.close()
