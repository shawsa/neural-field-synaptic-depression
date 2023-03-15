#!/usr/bin/python3
'''
A driver for the neural field simulator. Consider this a manual test of
most of the functionality.
'''

import experiment_defaults

import matplotlib.pyplot as plt
import numpy as np
import os.path
from adaptive_front import U_numeric, Q_numeric
from functools import partial
from neural_field import NeuralField, Parameters, heaviside_firing_rate, exponential_weight_kernel
from plotting_helpers import make_animation
from root_finding_helpers import find_roots
from space_domain import SpaceDomain, BufferedSpaceDomain
from time_domain import TimeDomain, TimeDomain_Start_Stop_MaxSpacing
from time_integrator import Euler, EulerDelta
from time_integrator_tqdm import TqdmWrapper

FILE_NAME = os.path.join(experiment_defaults.media_path,
                         'simple_stim')

params = Parameters(mu=1.0, alpha=20.0, gamma=.2)
theta = 0.1

# space = SpaceDomain(-100, 200, 10**4)
# time = TimeDomain_Start_Stop_MaxSpacing(0, 18, 1e-3/5)
space = BufferedSpaceDomain(-100, 200, 10**3, .1)
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

def target_speed(t):
    if t < 5:
        return 7
    elif 5 <= t < 10:
        return 2
    else:
        return 20

def level_foo(alpha, mu, gamma, c, theta, **_):
    return theta - (gamma + c*alpha*gamma)/2/(1+c*alpha*gamma)/(1+c*mu)

def rhs(t, u):
    stim = level_foo(c=target_speed(t), theta=theta, **params.dict)
    profile = np.zeros_like(u)
    profile[0] += stim
    return model.rhs(t, u) + profile

solver = TqdmWrapper(Euler())
fig, axes = plt.subplots(2, 1, figsize=(10, 10))
u_line, = axes[0].plot(space.array, u0[0], 'b-')
speed_line, = axes[1].plot(time.array, list(map(target_speed, time.array)), 'k-', label='target')
fronts = []
for (u, q), t in zip(solver.solution_generator(u0, rhs, time), time.array):
    u_line.set_ydata(u)
    front = find_roots(space.inner, u[space.inner_slice]-theta, window=3)[-1]
    fronts.append(front)
    if len(fronts) > 1:
        speed = (fronts[-1] - fronts[-2])/time.spacing
        axes[1].plot(t, speed, 'b.')
    plt.pause(time.spacing)
