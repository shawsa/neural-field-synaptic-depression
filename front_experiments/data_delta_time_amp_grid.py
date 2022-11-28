#!/usr/bin/python3
'''
'''

import experiment_defaults

import numpy as np
import os.path
from adaptive_front import U_numeric, Q_numeric, get_speed
from functools import partial
from itertools import product
from neural_field import (NeuralField,
                          Parameters,
                          heaviside_firing_rate,
                          exponential_weight_kernel)
import pickle
from plotting_helpers import make_animation
from root_finding_helpers import find_roots
from space_domain import BufferedSpaceDomain
from time_domain import TimeDomain, TimeDomain_Start_Stop_MaxSpacing
from time_integrator import Euler
from tqdm import tqdm


def main():

    file_name = os.path.join(experiment_defaults.data_path,
                             'delta_time_amp_grid.pickle')

    N = 21
    eps_u_array = np.linspace(-0.08, 0.08, N)
    eps_q_array = np.linspace(-0.08, 0.08, N)

    params = Parameters(mu=1.0, alpha=20.0, gamma=0.2)
    theta = 0.1

    speed = get_speed(mu=params.mu,
                      alpha=params.alpha,
                      gamma=params.gamma,
                      theta=theta)

    space = BufferedSpaceDomain(-100, 200, 10**4, .2)
    time = TimeDomain_Start_Stop_MaxSpacing(0, 30, 1e-3/5)
    # space = SpaceDomain(-100, 200, 10**3)
    # time = TimeDomain_Start_Stop_MaxSpacing(0, 18, 1e-2)
    solver = Euler()

    u0 = np.empty((2, space.num_points))
    model = NeuralField(space=space,
                        firing_rate=partial(heaviside_firing_rate,
                                            theta=theta),
                        weight_kernel=exponential_weight_kernel,
                        params=params)

    u0[0] = U_numeric(space.array, theta=theta, **params.dict)
    u0[1] = Q_numeric(space.array, theta=theta, **params.dict)

    eps_u_mat, eps_q_mat = np.meshgrid(eps_u_array, eps_q_array)
    response_mat = np.zeros_like(eps_u_mat)

 
    loop_len = len(eps_u_array)*len(eps_q_array)
    for (u_index, eps_u), (q_index, eps_q) in tqdm(product(enumerate(eps_u_array), enumerate(eps_q_array)), total=loop_len):
        u0[0] = U_numeric(space.array, **params.dict, theta=theta) + eps_u
        u0[1] = Q_numeric(space.array, **params.dict, theta=theta) + eps_q

        us = solver.t_final(u0, model.rhs, time)
        fronts = find_roots(space.inner, us[0][space.inner_slice]-theta, window=3)
        if len(fronts) != 1:
            response_mat[q_index, u_index] = np.nan
            print(eps_u, eps_q, np.nan)
        else:
            response_mat[q_index, u_index] = fronts[0] - speed*time.array[-1]

    with open(file_name, 'wb') as f:
        pickle.dump((eps_u_mat, eps_q_mat, response_mat), f)


if __name__ == '__main__':
    main()
