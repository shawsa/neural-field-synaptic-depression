#!/usr/bin/python3
'''
Test traveling front solution for the laterally inhibited case
(wizzard hat weight function).
'''

import experiment_defaults

import numpy as np
import os.path
# from laterally_inhibited_front import U_numeric, Q_numeric
from functools import partial
from neural_field import (NeuralField,
                          ParametersBeta,
                          heaviside_firing_rate,
                          wizzard_hat)
from plotting_helpers import make_animation
from space_domain import SpaceDomain
from time_domain import TimeDomain, TimeDomain_Start_Stop_MaxSpacing
from time_integrator import Euler
from time_integrator_tqdm import TqdmWrapper



def main():

    file_name = os.path.join(experiment_defaults.media_path,
                             'profile.mp4')

    params = ParametersBeta(mu=1.0, alpha=20.0, beta=0.2)
    theta = 0.1

    # space = SpaceDomain(-100, 200, 10**4)
    # time = TimeDomain_Start_Stop_MaxSpacing(0, 18, 1e-3/5)
    space = SpaceDomain(-100, 200, 10**3)
    time = TimeDomain_Start_Stop_MaxSpacing(0, 18, 1e-2)

    u0 = np.empty((2, space.num_points))
    u0[0] = np.exp(-.01*space.array**2)
    u0[1] = 1
    # u0[0] = U_numeric(space.array+initial_offset, theta=theta, **params.dict)
    # u0[1] = Q_numeric(space.array+initial_offset, theta=theta, **params.dict)

    model = NeuralField(space=space,
                        firing_rate=partial(heaviside_firing_rate,
                                            theta=theta),
                        weight_kernel=wizzard_hat,
                        params=params)

    solver = TqdmWrapper(Euler())
    us = solver.solve(u0, model.rhs, time)

    print('an2imating...')

    make_animation(file_name,
                   time.array,
                   space.array,
                   [us],
                   us_labels=('perturbed', ),
                   theta=theta,
                   x_window=(-15, 80),
                   frames=100,
                   fps=24,
                   animation_interval=400)


if __name__ == '__main__':
    main()
