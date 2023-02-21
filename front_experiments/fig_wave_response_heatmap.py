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
from neural_field import NeuralField, Parameters, heaviside_firing_rate, exponential_weight_kernel
from space_domain import SpaceDomain
from time_domain import TimeDomain_Start_Stop_MaxSpacing
from time_integrator import EulerDelta
from time_integrator_tqdm import TqdmWrapper

from root_finding_helpers import find_roots

FILE_NAME = 'front_wave_response'

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
epsilon = 0.09
pulse_profile = np.ones_like(u0)
pulse_profile[1] *= 0

solver = TqdmWrapper(EulerDelta(delta_time, epsilon*pulse_profile))

print('solving purturbed case')
us = solver.solve(u0, model.rhs, time)

print('finding front locations')
fronts = fronts = [find_roots(space.array, u-theta, window=7)[-1]
                   for u, q in tqdm(us)]

speed = get_speed(theta=theta, **params.dict)

x_window = (-20, 70)
x_window_indices = [np.argmin(np.abs(x - space.array)) for x in x_window]

x_pixels = 500
x_stride = (x_window_indices[1] - x_window_indices[0])//x_pixels
x_slice = slice(x_window_indices[0], x_window_indices[1], x_stride)

t_stop_index = int(0.8*len(time.array))
y_pixels = 500
y_stride = t_stop_index//y_pixels
y_slice = slice(0, t_stop_index, y_stride)

sol_array = np.array([u[0][x_slice] for u in us[y_slice]]).T

plt.figure(figsize=(7, 4))

plt.pcolormesh(time.array[y_slice],
               space.array[x_slice],
               sol_array,
               cmap='Greys')
plt.ylabel('$x$')
plt.xlabel('$t$')

plt.plot(time.array[y_slice],
         fronts[y_slice],
         'b-', label='measured', linewidth=3)
plt.plot(time.array[y_slice],
         (speed * time.array[y_slice]-initial_offset +
          np.heaviside(time.array[y_slice]-delta_time, 0) *
          response(epsilon, theta=theta, **params.dict)),
         'g--', linewidth=3, label='predicted')

plt.colorbar(label='$u$')
plt.legend(loc='lower right')
plt.title(f'$I(x,t) = {epsilon} \\delta(t - {delta_time})$')
plt.tight_layout()
for extension in ['.png', '.eps']:
    plt.savefig(os.path.join(experiment_defaults.media_path,
                             FILE_NAME + extension))
