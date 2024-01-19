#!/usr/bin/python3
"""
A driver for the neural field simulator. Consider this a manual test of
most of the functionality.
"""

import experiment_defaults

import matplotlib.pyplot as plt
import numpy as np
import os.path
import pickle

from functools import partial
from itertools import product
from tqdm import tqdm
from adaptive_front import U_numeric, Q_numeric, get_speed
from neural_field_synaptic_depression.neural_field import (
    NeuralField,
    Parameters,
    heaviside_firing_rate,
    exponential_weight_kernel,
)
from neural_field_synaptic_depression.space_domain import BufferedSpaceDomain
from neural_field_synaptic_depression.time_domain import (
    TimeDomain_Start_Stop_MaxSpacing,
)
from neural_field_synaptic_depression.time_integrator import EulerDelta
from neural_field_synaptic_depression.time_integrator_tqdm import TqdmWrapper
from neural_field_synaptic_depression.root_finding_helpers import find_roots


FILE_NAME = "figure3_error.pickle"

mu = 1.0
theta = 0.1
alphas = np.linspace(5.0, 20.0, 5)
gammas = np.arange(0.11, 0.31, 0.1)
delta_epsilon = 0.01

space = BufferedSpaceDomain(-100, 200, 10**4, 0.2)
time = TimeDomain_Start_Stop_MaxSpacing(0, 30, 1e-3 / 5)
initial_offset = 0

delta_time = 1
u0 = np.empty((2, space.num_points))
pulse_profile = np.ones_like(u0)
pulse_profile[1] *= 0

data_list = []
for alpha, gamma, epsilon in tqdm(
    list(product(alphas, gammas, [-delta_epsilon, delta_epsilon]))
):
    params = Parameters(mu=mu, alpha=alpha, gamma=gamma)
    u0[0] = U_numeric(space.array + initial_offset, theta=theta, **params.dict)
    u0[1] = Q_numeric(space.array + initial_offset, theta=theta, **params.dict)
    model = NeuralField(
        space=space,
        firing_rate=partial(heaviside_firing_rate, theta=theta),
        weight_kernel=exponential_weight_kernel,
        params=params,
    )

    speed = get_speed(theta=theta, **params.dict)
    base_response = speed * time.array[-1] - initial_offset
    solver = TqdmWrapper(EulerDelta(delta_time, epsilon * pulse_profile))
    u, q = solver.t_final(u0, model.rhs, time)
    front = find_roots(space.inner, u[space.inner_slice] - theta, window=3)
    response = front - base_response
    data_list.append((params, theta, epsilon, speed, response))

with open(os.path.join(experiment_defaults.data_path, FILE_NAME), "wb") as f:
    pickle.dump(data_list, f)
