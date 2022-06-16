#!/usr/bin/python3
'''
A driver for the neural field simulator. Consider this a manual test of
most of the functionality.
'''

import experiment_defaults

from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import os.path
from tqdm import tqdm
from adaptive_front import U_numeric, Q_numeric, get_speed, response
from neural_field import (NeuralField,
                          ParametersBeta,
                          heaviside_firing_rate,
                          exponential_weight_kernel)
from space_domain import SpaceDomain
from time_domain import TimeDomain_Start_Stop_MaxSpacing
from time_integrator import EulerDelta
from time_integrator_tqdm import TqdmWrapper

from root_finding_helpers import find_roots


def main():

    file_name = 'fig_spatially_homogeneous_limit'

    params = ParametersBeta(mu=1.0, alpha=20.0, beta=0.2)
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
    pulse_profile = np.ones_like(u0)
    pulse_profile[1] *= 0

    speed = get_speed(theta=theta, **params.dict)
    base_response = speed * time.array[-1] - initial_offset

    response_slope = response(1, **params.dict, theta=theta)

    epsilons = np.linspace(-0.09, 0.09, 11)
    responses = []

    x_root_index_start = 200

    for eps_index, epsilon in enumerate(epsilons):
        print(f'{eps_index+1}/{len(epsilons)}')
        solver = TqdmWrapper(EulerDelta(delta_time, epsilon*pulse_profile))
        u, q = solver.t_final(u0, model.rhs, time)
        front = find_roots(space.array[x_root_index_start:], u[x_root_index_start:]-theta, window=7)[-1]
        responses.append(front - base_response)

    plt.figure(figsize=(7, 4))
    plt.plot(epsilons, responses, 'go', label='measured')
    plt.plot(epsilons, epsilons*response_slope, 'k-', label='asyptotic')

    plt.xlabel('$\\epsilon$')
    plt.ylabel('$\\nu_\\infty$')
    plt.title('Response to $I(x,t) = \\epsilon \\delta(t)$')

    for extension in ['.png', '.eps']:
        plt.savefig(os.path.join(experiment_defaults.media_path,
                                 file_name + extension))

if __name__ == '__main__':
    main()
