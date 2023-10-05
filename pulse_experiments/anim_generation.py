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
from itertools import islice
from more_itertools import windowed
from scipy.stats import linregress

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

from plotting_helpers.plotting_helpers import make_animation

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

FILE_NAME = os.path.join(experiment_defaults.media_path, "pulse_generation.gif")

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

space = BufferedSpaceDomain(-100, 100, 10**4, 0.1)
time = TimeDomain_Start_Stop_MaxSpacing(0, 40, 1e-3)

u0 = np.ones((2, space.num_points))
u0[0] = 0.3 * np.exp(-space.array**2)

model = NeuralField(
    space=space,
    firing_rate=partial(heaviside_firing_rate, theta=params_dict["theta"]),
    weight_kernel=exponential_weight_kernel,
    params=params,
)

solver = TqdmWrapper(Euler())

fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.set_ylim(-0.1, 1.1)
(u_line,) = ax.plot(space.array, u0[0], "b-", linewidth=5)
# q_line, = ax.plot(space.array, u0[1], 'b--', label='synaptic efficacy')
steps_per_frame = 350
window_width = 10
with imageio.get_writer(FILE_NAME, mode="I") as writer:
    for index, (t, (u, q)) in enumerate(
        zip(time.array, solver.solution_generator(u0, model.rhs, time))
    ):
        if index % steps_per_frame != 0:
            continue
        u_line.set_ydata(u)
        # q_line.set_ydata(q)
        plt.savefig(FILE_NAME + ".png")
        image = imageio.imread(FILE_NAME + ".png")
        writer.append_data(image)
        plt.pause(0.001)

os.remove(FILE_NAME + ".png")
plt.close()
