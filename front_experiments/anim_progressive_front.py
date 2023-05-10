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

from adaptive_front import U_numeric, Q_numeric
from functools import partial
from neural_field import NeuralField, Parameters, heaviside_firing_rate, exponential_weight_kernel
from plotting_helpers import make_animation
from space_domain import SpaceDomain
from time_domain import TimeDomain, TimeDomain_Start_Stop_MaxSpacing
from time_integrator import Euler, EulerDelta
from time_integrator_tqdm import TqdmWrapper


FILE_NAME = os.path.join(experiment_defaults.media_path,
                         'progressive_front.gif')

params = Parameters(mu=1.0, alpha=10, gamma=0.25)
theta = 0.1

space = SpaceDomain(-100, 400, 10**4)
time = TimeDomain_Start_Stop_MaxSpacing(0, 20, 1e-2)

u0 = np.empty((2, space.num_points))
u0[0] = U_numeric(space.array, **params.dict, theta=theta)
u0[1] = Q_numeric(space.array, **params.dict, theta=theta)

model = NeuralField(space=space,
                    firing_rate=partial(heaviside_firing_rate,
                                        theta=theta),
                    weight_kernel=exponential_weight_kernel,
                    params=params)


solver = TqdmWrapper(Euler())

plt.rc('font', size=15)
plt.rc('lines', linewidth=5)
fig, axes = plt.subplots(1, 1, figsize=(5, 4))
if not hasattr(axes, 'len'):
    axes = [axes]
u_line, = axes[0].plot(space.array, u0[0], 'b-', label='$u$')
q_line, = axes[0].plot(space.array, u0[1], 'b--', label='$q$')
axes[0].plot(space.array, theta + 0*space.array, 'k:', zorder=-10, label='$\\theta$')
# axes[0].plot(space.array, params.gamma + 0*space.array, 'g:', zorder=-10, label='$\\gamma$')
axes[0].set_xlim(-20, 100)
axes[0].set_xlabel('$x$')
axes[0].legend(loc='upper left')
plt.tight_layout()
frame_skip = 20
with imageio.get_writer(FILE_NAME, mode='I') as writer:
    for index, (t, (u, q)) in enumerate(
            zip(time.array,
                solver.solution_generator(u0, model.rhs, time))):
        if index % frame_skip != 0:
            continue
        u_line.set_ydata(u)
        q_line.set_ydata(q)

        plt.savefig(FILE_NAME + '.png')
        image = imageio.imread(FILE_NAME + '.png')
        writer.append_data(image)
        plt.pause(0.001)

os.remove(FILE_NAME + '.png')
