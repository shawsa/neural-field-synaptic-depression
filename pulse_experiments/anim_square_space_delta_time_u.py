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

from functools import partial

from neural_field_synaptic_depression.neural_field import (
    NeuralField,
    ParametersBeta,
    heaviside_firing_rate,
    exponential_weight_kernel,
)
from neural_field_synaptic_depression.root_finding_helpers import find_roots
from neural_field_synaptic_depression.space_domain import (
    SpaceDomain,
    BufferedSpaceDomain,
)
from neural_field_synaptic_depression.time_domain import (
    TimeDomain,
    TimeDomain_Start_Stop_MaxSpacing,
)
from neural_field_synaptic_depression.time_integrator import Euler, EulerDelta
from neural_field_synaptic_depression.time_integrator_tqdm import TqdmWrapper

from num_assist import (
    Domain,
    find_delta,
    find_c,
    pulse_profile,
    nullspace_amplitudes,
    v1,
    v2,
    local_interp,
)

from helper_symbolics import get_traveling_pulse, get_numerical_parameters

FILE_NAME = os.path.join(
    experiment_defaults.media_path, "square_space_delta_time_u.gif"
)

params = ParametersBeta(
    **{
        "alpha": 20.0,
        "beta": 5.0,
        "mu": 1.0,
    }
)
params_dict = {
    **params.dict,
    "gamma": params.gamma,
    "theta": 0.2,
    "weight_kernel": exponential_weight_kernel,
}
theta = params_dict["theta"]
xs_right = Domain(0, 200, 8001)
xs_left = Domain(-200, 0, 8001)

"""Finding the speed and pulse width can be slow. Saving them for a given
parameter set helps for rappid testing."""
USE_SAVED_VALUES = True
if USE_SAVED_VALUES:
    c, Delta = 1.0509375967740198, 9.553535461425781
    print(f"c={c}\nDelta={Delta}")
else:
    Delta_interval = (7, 20)
    speed_interval = (1, 10)
    Delta = find_delta(
        *Delta_interval, *speed_interval, xs_left, xs_right, verbose=True, **params
    )
    c = find_c(*speed_interval, xs_right, Delta=Delta, verbose=True, **params)

params_dict["c"] = c
params_dict["Delta"] = Delta

xis, Us, Qs = pulse_profile(xs_right=xs_right, xs_left=xs_left, **params_dict)

space = BufferedSpaceDomain(-100, 200, 10**4, 0.1)
time = TimeDomain_Start_Stop_MaxSpacing(0, 20, 2e-2)

initial_offset = 0
initial = np.empty((2, space.num_points))
initial[0] = np.array([local_interp(x, xis, Us) for x in space.array])
initial[1] = np.array([local_interp(x, xis, Qs) for x in space.array])


stim_center = 8
stim_width = 1
delta_time = 5
epsilon = 0.2
stim_profile = np.zeros_like(initial)
stim_profile[0] = np.heaviside(
    space.array - (stim_center - stim_width / 2), 0.5
) * np.heaviside(-space.array + (stim_center + stim_width / 2), 0.5)

model = NeuralField(
    space=space,
    firing_rate=partial(heaviside_firing_rate, theta=theta),
    weight_kernel=exponential_weight_kernel,
    params=params,
)

solver0 = TqdmWrapper(EulerDelta(delta_time, epsilon * stim_profile))
solver1 = TqdmWrapper(Euler())

plt.rc("font", size=15)
plt.rc("lines", linewidth=5)

fig, ax = plt.subplots(1, 1, figsize=(5, 4))
(line_u0,) = ax.plot(space.array, initial[0], "g-")
(line_q0,) = ax.plot(space.array, initial[1], "g--")
(line_u1,) = ax.plot(space.array, initial[0], "b-")
(line_q1,) = ax.plot(space.array, initial[1], "b--")

ax.set_xlim(-20, 40)
ax.set_xlabel("$x$")
plt.tight_layout()
FRAME_SKIP = 10
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
