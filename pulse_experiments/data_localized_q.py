#!/usr/bin/python3
"""Spatially locallized stimulation of traveling pulses."""

import experiment_defaults

import matplotlib.pyplot as plt
import numpy as np
import os.path
import pickle

from functools import partial
from tqdm import tqdm

from neural_field_synaptic_depression.neural_field import (
    NeuralField,
    ParametersBeta,
    heaviside_firing_rate,
    exponential_weight_kernel,
)
from neural_field_synaptic_depression.space_domain import BufferedSpaceDomain
from neural_field_synaptic_depression.time_domain import (
    TimeDomain_Start_Stop_MaxSpacing,
)
from neural_field_synaptic_depression.time_integrator import Euler
from neural_field_synaptic_depression.time_integrator_tqdm import TqdmWrapper
from neural_field_synaptic_depression.root_finding_helpers import find_roots

from helper_symbolics import get_speed_and_width
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


FILE_NAME = "spatially_localized_q.pickle"

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
space = BufferedSpaceDomain(-100, 200, 10**4, 0.2)
time = TimeDomain_Start_Stop_MaxSpacing(0, 20, 1e-3 / 5)

# pulse solution
xs_left = Domain(space.array[0], 0, len(space.array))
xs_right = Domain(0, space.array[-1], len(space.array))
USE_SAVED_VALUES = True
if USE_SAVED_VALUES:
    c, Delta = 1.0457801654119976, 9.497941970825195
    print(f"c={c}\nDelta={Delta}")
else:
    Delta_interval = (7, 10)
    speed_interval = (1, 2)
    Delta = find_delta(
        *Delta_interval, *speed_interval, xs_left, xs_right, verbose=True, **params_dict
    )
    c = find_c(*speed_interval, xs_right, Delta=Delta, verbose=True, **params_dict)

xis, Us, Qs = pulse_profile(
    xs_right=xs_right, xs_left=xs_left, c=c, Delta=Delta, **params_dict
)

u0 = np.empty((2, space.num_points))
u0[0] = np.array([local_interp(x, xis, Us) for x in space.array])
u0[1] = np.array([local_interp(x, xis, Qs) for x in space.array])

model = NeuralField(
    space=space,
    firing_rate=partial(heaviside_firing_rate, theta=params_dict["theta"]),
    weight_kernel=params_dict["weight_kernel"],
    params=params,
)

stim_width = 5
stim_amp = 0.01
base_response = c * time.array[-1]
solver = TqdmWrapper(Euler())
u, q = solver.t_final(u0, model.rhs, time)
base_loc = find_roots(
    space.inner, u[space.inner_slice] - params_dict["theta"], window=3
)[-1]

locations = np.linspace(-10, 10, 101)
responses = []
for loc in tqdm(locations):
    pulse_profile = np.zeros_like(u0)
    pulse_profile[1] += (
        stim_amp
        / params.alpha
        * (
            np.heaviside(space.array - loc + stim_width / 2, 0)
            - np.heaviside(space.array - loc - stim_width / 2, 0)
        )
    )
    solver = TqdmWrapper(Euler())
    u, q = solver.t_final(u0 + pulse_profile, model.rhs, time)
    front = find_roots(
        space.inner, u[space.inner_slice] - params_dict["theta"], window=3
    )[-1]
    responses.append(front - base_loc)

plt.plot(locations, responses, "k.")

with open(os.path.join(experiment_defaults.data_path, FILE_NAME), "wb") as f:
    pickle.dump((params, params_dict, stim_width, stim_amp, locations, responses), f)
