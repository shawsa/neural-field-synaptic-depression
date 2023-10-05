"""Generate a figure similar to Fig 2. in Kilpatrik Ermentrout 2012.
Use a delta for the stimulus variable, and use a bi-exponential
traveling pulse.
"""

import experiment_defaults

import matplotlib.pyplot as plt
import numpy as np
import os

from functools import partial

from neural_field_synaptic_depression.neural_field import (
    NeuralField,
    ParametersBeta,
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


FIG_FILE_NAME = os.path.join(
    experiment_defaults.media_path, "bi-exponential_response_delta_time.png"
)


def weight_kernel(x):
    return 0.5 * np.exp(-np.abs(x))


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
    "weight_kernel": weight_kernel,
}
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

xis, Us, Qs = pulse_profile(xs_right, xs_left, **params_dict)

space = SpaceDomain(-100, 200, 10**3)
time = TimeDomain_Start_Stop_MaxSpacing(0, 18, 1e-2)

initial_offset = 0
u0 = np.empty((2, space.num_points))
u0[0] = np.array([local_interp(x, xis, Us) for x in space.array])
u0[1] = np.array([local_interp(x, xis, Qs) for x in space.array])

model = NeuralField(
    space=space,
    firing_rate=partial(heaviside_firing_rate, theta=params["theta"]),
    weight_kernel=exponential_weight_kernel,
    params=params,
)

stim_center = -4
stim_width = 5
delta_time = 1
epsilon = 0.2
stim_profile = np.zeros_like(u0)
stim_profile[0] = np.heaviside(
    space.array - (stim_center - stim_width / 2), 0.5
) * np.heaviside(-space.array + (stim_center + stim_width / 2), 0.5)

solver = TqdmWrapper(EulerDelta(delta_time, epsilon * stim_profile))

print("solving purturbed case")
us = solver.solve(u0, model.rhs, time)

# unperturbed
print("solving unpurturbed case")
unperturbed_solver = TqdmWrapper(Euler())
us_unperturbed = unperturbed_solver.solve(u0, model.rhs, time)


print("animating...")

# make_animation(file_name,
#            time.array,
#            space.array,
#            [us, us_unperturbed],
#            us_labels=('perturbed', 'unperturbed'),
#            theta=theta,
#            x_window=(-15, 20),
#            frames=100,
#            fps=12,
#            animation_interval=400)
