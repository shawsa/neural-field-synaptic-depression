#!/usr/bin/python3
'''
A driver for the neural field simulator. Consider this a manual test of
most of the functionality.
'''

import experiment_defaults

import numpy as np
import os.path
from adaptive_front import U_numeric, Q_numeric
from functools import partial
from neural_field import (NeuralField,
                          ParametersBeta,
                          heaviside_firing_rate,
                          exponential_weight_kernel)
from plotting_helpers import make_animation
from space_domain import SpaceDomain
from time_domain import TimeDomain, TimeDomain_Start_Stop_MaxSpacing
from time_integrator import Euler, EulerDelta
from time_integrator_tqdm import TqdmWrapper



def main():

    file_name = os.path.join(experiment_defaults.media_path,
                             'delta_time_u.mp4')

    params = ParametersBeta(mu=1.0, alpha=20.0, beta=0.2)
    theta = 0.1

    # space = SpaceDomain(-100, 200, 10**4)
    # time = TimeDomain_Start_Stop_MaxSpacing(0, 18, 1e-3/5)
    space = SpaceDomain(-100, 200, 10**3)
    time = TimeDomain_Start_Stop_MaxSpacing(0, 18, 1e-2)

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
    epsilon = 0.09
    pulse_profile = np.ones_like(u0)
    pulse_profile[1] *= 0

    solver = TqdmWrapper(EulerDelta(delta_time, epsilon*pulse_profile))

    print('solving purturbed case')
    us = solver.solve(u0, model.rhs, time)

    # unperturbed
    print('solving unpurturbed case')
    unperturbed_solver = TqdmWrapper(Euler())
    us_unperturbed = unperturbed_solver.solve(u0, model.rhs, time)


    print('animating...')

    make_animation(file_name,
                   time.array,
                   space.array,
                   [us, us_unperturbed],
                   us_labels=('perturbed', 'unperturbed'),
                   theta=theta,
                   x_window=(-15, 80),
                   title=f'$I_u(x,t) = {epsilon}\\delta(t-{delta_time})$',
                   frames=100,
                   fps=24,
                   animation_interval=400)


if __name__ == '__main__':
    main()
