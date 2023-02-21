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
from neural_field import NeuralField, Parameters, heaviside_firing_rate, exponential_weight_kernel
from plotting_helpers import make_animation
from space_domain import SpaceDomain
from time_domain import TimeDomain, TimeDomain_Start_Stop_MaxSpacing
from time_integrator import Euler, EulerDelta
from time_integrator_tqdm import TqdmWrapper


def main():

    file_name = os.path.join(experiment_defaults.media_path,
                             'regressive.mp4')

    params1 = Parameters(mu=1.0, alpha=10, gamma=0.25)
    params2 = Parameters(mu=1.0, alpha=10, gamma=0.20)
    params3 = Parameters(mu=1.0, alpha=10, gamma=0.15)
    theta = 0.1

    space = SpaceDomain(-100, 400, 10**4)
    time = TimeDomain_Start_Stop_MaxSpacing(0, 100, 1e-2)

    u0 = np.empty((2, space.num_points))
    u0[0] = np.heaviside(50-space.array, 0)*.2
    u0[1] = np.ones_like(space.array)*.2

    model1 = NeuralField(space=space,
                         firing_rate=partial(heaviside_firing_rate,
                                             theta=theta),
                         weight_kernel=exponential_weight_kernel,
                         params=params1)

    model2 = NeuralField(space=space,
                         firing_rate=partial(heaviside_firing_rate,
                                             theta=theta),
                         weight_kernel=exponential_weight_kernel,
                         params=params2)

    model3 = NeuralField(space=space,
                         firing_rate=partial(heaviside_firing_rate,
                                             theta=theta),
                         weight_kernel=exponential_weight_kernel,
                         params=params3)

    print('solving model 1')
    solver = TqdmWrapper(Euler())
    us1 = solver.solve(u0, model1.rhs, time)

    print('solving model 2')
    solver = TqdmWrapper(Euler())
    us2 = solver.solve(u0, model2.rhs, time)

    print('solving model 3')
    solver = TqdmWrapper(Euler())
    us3 = solver.solve(u0, model3.rhs, time)

    print('animating...')

    # file_name = None
    make_animation(file_name,
                   time.array,
                   space.array,
                   [us1, us2, us3],
                   us_labels=[f'$\\gamma = {params.gamma}$'
                              for params in [params1, params2, params3]],
                   theta=theta,
                   x_window=(-15, 160),
                   y_window=(-.1, 1.3),
                   frames=100,
                   fps=12,
                   animation_interval=400)


if __name__ == '__main__':
    main()
