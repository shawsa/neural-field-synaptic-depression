#!/usr/bin/python3
'''
The wave response to the inpulse I(x,t) = epsilon * delta(t - 1)
simulated and compared to asymptotic approximation.
'''

import experiment_defaults

from adaptive_front import U_numeric, Q_numeric, get_speed, response
from functools import partial
import matplotlib.pyplot as plt
from neural_field import NeuralField, Parameters, heaviside_firing_rate, exponential_weight_kernel
import numpy as np
import os.path
from root_finding_helpers import find_roots
from space_domain import SpaceDomain
from time_domain import TimeDomain_Start_Stop_MaxSpacing
from time_integrator import EulerDelta
from time_integrator_tqdm import TqdmWrapper
from tqdm import tqdm


def main():

    FILE_NAME = 'wave_response_small'

    params = Parameters(mu=1.0, alpha=20.0, gamma=.2)
    theta = 0.1

    space = SpaceDomain(-100, 200, 10**4)
    time = TimeDomain_Start_Stop_MaxSpacing(0, 18, 1e-3/5)

    initial_offset = 0
    u0 = np.empty((2, space.num_points))
    u0[0] = U_numeric(space.array+initial_offset, theta=theta, **params.dict)
    u0[1] = Q_numeric(space.array+initial_offset, theta=theta, **params.dict)

    model = NeuralField(space=space,
                        firing_rate=partial(heaviside_firing_rate,
                                            theta=theta),
                        weight_kernel=exponential_weight_kernel,
                        params=params)

    delta_time = 1
    epsilon = 0.01
    pulse_profile = np.ones_like(u0)
    pulse_profile[1] *= 0

    asymptotic = response(epsilon, **params.dict, theta=theta)
    speed = get_speed(theta=theta, **params.dict)

    solver = TqdmWrapper(EulerDelta(delta_time, epsilon*pulse_profile))


    us = solver.solve(u0, model.rhs, time)

    print('finding fronts')
    fronts = [find_roots(space.array, u-theta, window=7)[-1] for u, q in tqdm(us)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(time.array, fronts, 'k-', label='measured')
    axes[0].plot(time.array,
                 speed*time.array-initial_offset + np.heaviside(time.array-delta_time, 0)*asymptotic,
                 'b--', label='predicted')
    axes[0].legend()

    axes[1].plot([], [], 'k-', label='measured')
    axes[1].plot(time.array, fronts - speed*time.array, 'k,')
    axes[1].plot(time.array,
                 np.heaviside(time.array-delta_time, 0)*asymptotic,
                 'b--', label='predicted')
    axes[1].legend()
    for extension in ['.png', '.eps']:
        plt.savefig(os.path.join(experiment_defaults.media_path,
                                 FILE_NAME + extension))
    plt.show()

if __name__ == '__main__':
    main()
