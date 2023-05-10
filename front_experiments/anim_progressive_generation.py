#!/usr/bin/python3
'''
A driver for the neural field simulator. Consider this a manual test of
most of the functionality.
'''

import experiment_defaults

import imageio
import matplotlib.pyplot as plt
import numpy as np
import os.path

from functools import partial
from neural_field import NeuralField, Parameters, heaviside_firing_rate, exponential_weight_kernel
from space_domain import SpaceDomain
from time_domain import TimeDomain, TimeDomain_Start_Stop_MaxSpacing
from time_integrator import Euler
from time_integrator_tqdm import TqdmWrapper


FILE_NAME = os.path.join(experiment_defaults.media_path,
                         'progressive_front_gen.gif')

params = Parameters(mu=1.0, alpha=10, gamma=0.25)
theta = 0.1

space = SpaceDomain(-100, 400, 10**4)
time = TimeDomain_Start_Stop_MaxSpacing(0, 50, 1e-2)

u0 = np.empty((2, space.num_points))
u0[0] = np.heaviside(50-space.array, 0)*.2
u0[1] = np.ones_like(space.array)*.2

model = NeuralField(space=space,
                    firing_rate=partial(heaviside_firing_rate,
                                        theta=theta),
                    weight_kernel=exponential_weight_kernel,
                    params=params)

solver = TqdmWrapper(Euler())


plt.rc('font', size=15)
plt.rc('lines', linewidth=5)

fig, ax = plt.subplots(1, 1, figsize=(10, 4))
line_u, = ax.plot(space.array, u0[0], 'g-')

ax.set_xlim(-5, 200)
ax.set_ylim(-.1, 1.1)
ax.set_xlabel('$x$')
ax.set_ylabel('Activity')
plt.tight_layout()
FRAME_SKIP = 40
with imageio.get_writer(FILE_NAME, mode='I') as writer:
    for index, (u, q) in enumerate(solver.solution_generator(u0, model.rhs, time)):
        if index % FRAME_SKIP != 0:
            continue
        line_u.set_ydata(u)

        plt.savefig(FILE_NAME + '.png')
        image = imageio.imread(FILE_NAME + '.png')
        writer.append_data(image)
        plt.pause(0.001)

os.remove(FILE_NAME + '.png')
plt.close()
