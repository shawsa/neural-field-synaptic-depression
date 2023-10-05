#!/usr/bin/python3

import experiment_defaults

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


FILE_NAME = os.path.join(experiment_defaults.media_path, "tracking_test.mp4")

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
xs_right = Domain(0, 400, 10001)
xs_left = Domain(-400, 0, 10001)

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

xis, Us, Qs = pulse_profile(
    xs_right=xs_right, xs_left=xs_left, vanish_tol=1e-2, **params_dict
)

space = BufferedSpaceDomain(-100, 400, 10**4, 0.1)
time = TimeDomain_Start_Stop_MaxSpacing(0, 80, 1e-2)

u0 = np.empty((2, space.num_points))
u0[0] = np.array([local_interp(x, xis, Us) for x in space.array])
u0[1] = np.array([local_interp(x, xis, Qs) for x in space.array])

model = NeuralField(
    space=space,
    firing_rate=partial(heaviside_firing_rate, theta=params_dict["theta"]),
    weight_kernel=exponential_weight_kernel,
    params=params,
)

solver = TqdmWrapper(Euler())


def square(t, t0, tf):
    return np.heaviside(t - t0, 0) * np.heaviside(tf - t, 1)


def target(t):
    return (
        3 * t * square(t, 0, 10)
        + (-10 + 4 * t) * square(t, 10, 20)
        + (50 + t) * square(t, 20, np.inf)
    )


stim_mag = 0.03


def stim_func(t):
    return stim_mag * np.heaviside(target(t) - space.array, 0)


def rhs(t, u):
    stim = np.zeros_like(u0)
    stim[0] = stim_func(t)
    return model.rhs(t, u) + stim


fig, axes = plt.subplots(2, 1, figsize=(10, 10))
(u_line,) = axes[0].plot(space.array, u0[0], "b-", label="activity")
(q_line,) = axes[0].plot(space.array, u0[1], "b--", label="synaptic efficacy")
(stim_line,) = axes[0].plot(space.array, stim_func(0), "m:", label="stimulus")
(front_line,) = axes[0].plot([0], [params_dict["theta"]], "k.")
(target_line,) = axes[0].plot(target(0), params_dict["theta"], "ro", label="Target")
axes[0].set_xlim(-20, 200)
axes[0].set_xlabel("$x$")
axes[0].legend(loc="upper left")
fronts = []
for index, (t, (u, q)) in enumerate(
    zip(time.array, solver.solution_generator(u0, rhs, time))
):
    u_line.set_ydata(u)
    q_line.set_ydata(q)
    fronts.append(
        find_roots(space.inner, u[space.inner_slice] - params_dict["theta"], window=3)[
            -1
        ]
    )
    front_line.set_xdata([fronts[-1]])
    stim_line.set_ydata(stim_func(t))
    target_line.set_xdata(target(t))
    if index % 100 == 0:
        plt.pause(0.001)

plt.figure("Entrainment?")
plt.plot(time.array, fronts, "b-")
plt.plot(time.array, [target(t) for t in time.array], "r-")
