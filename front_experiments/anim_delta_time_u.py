#!/usr/bin/python3
"""
A driver for the neural field simulator. Consider this a manual test of
most of the functionality.
"""

import experiment_defaults

import imageio
import matplotlib.pyplot as plt
import numpy as np
import os.path

from adaptive_front import U_numeric, Q_numeric
from functools import partial

from neural_field_synaptic_depression.neural_field import (
    NeuralField,
    Parameters,
    heaviside_firing_rate,
    exponential_weight_kernel,
)
from neural_field_synaptic_depression.space_domain import SpaceDomain
from neural_field_synaptic_depression.time_domain import (
    TimeDomain,
    TimeDomain_Start_Stop_MaxSpacing,
)
from neural_field_synaptic_depression.time_integrator import Euler, EulerDelta
from neural_field_synaptic_depression.time_integrator_tqdm import TqdmWrapper


FILE_NAME = os.path.join(experiment_defaults.media_path, "front_delta_time.gif")

params = Parameters(mu=1.0, alpha=20.0, gamma=0.2)
theta = 0.1

# space = SpaceDomain(-100, 200, 10**4)
# time = TimeDomain_Start_Stop_MaxSpacing(0, 18, 1e-3/5)
space = SpaceDomain(-100, 200, 10**3)
time = TimeDomain_Start_Stop_MaxSpacing(0, 18, 1e-2)

initial_offset = 0
initial = np.empty((2, space.num_points))
initial[0] = U_numeric(space.array + initial_offset, theta=theta, **params.dict)
initial[1] = Q_numeric(space.array + initial_offset, theta=theta, **params.dict)

model = NeuralField(
    space=space,
    firing_rate=partial(heaviside_firing_rate, theta=theta),
    weight_kernel=exponential_weight_kernel,
    params=params,
)

delta_time = 5
epsilon = 0.09
pulse_profile = np.ones_like(initial)
pulse_profile[1] *= 0

solver0 = TqdmWrapper(Euler())
solver1 = TqdmWrapper(EulerDelta(delta_time, epsilon * pulse_profile))

plt.rc("font", size=15)
plt.rc("lines", linewidth=5)

fig, ax = plt.subplots(1, 1, figsize=(5, 4))
(line_u0,) = ax.plot(space.array, initial[0], "g-")
(line_q0,) = ax.plot(space.array, initial[1], "g--")
(line_u1,) = ax.plot(space.array, initial[0], "b-")
(line_q1,) = ax.plot(space.array, initial[1], "b--")

ax.set_xlim(-20, 100)
ax.set_xlabel("$x$")
plt.tight_layout()
FRAME_SKIP = 20
with imageio.get_writer(FILE_NAME, mode="I") as writer:
    for index, (t, (u0, q0), (u1, q1)) in enumerate(
        zip(
            time.array,
            solver0.solution_generator(initial, model.rhs, time),
            solver1.solution_generator(initial, model.rhs, time),
        )
    ):
        if index % FRAME_SKIP != 0 and (t < 4.5 or t > 7.5):
            continue
        line_u0.set_ydata(u0)
        line_q0.set_ydata(q0)
        line_u1.set_ydata(u1)
        line_q1.set_ydata(q1)

        plt.savefig(FILE_NAME + ".png")
        image = imageio.imread(FILE_NAME + ".png")
        writer.append_data(image)
        plt.pause(0.001)

os.remove(FILE_NAME + ".png")
