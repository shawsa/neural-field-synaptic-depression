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
    local_diff,
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
space = BufferedSpaceDomain(-100, 200, 2 * 10**4, 0.2)
time = TimeDomain_Start_Stop_MaxSpacing(0, 50, 1e-3)

# pulse solution
xs_left = Domain(space.array[0], 0, len(space.array))
xs_right = Domain(0, space.array[-1], len(space.array))
USE_SAVED_VALUES = True
if USE_SAVED_VALUES:
    c, Delta = 1.0383422424611126, 9.424068408086896
    print(f"c={c}\nDelta={Delta}")
else:
    Delta_interval = (7, 10)
    speed_interval = (1, 2)
    print(f"Delta interval: {Delta_interval}")
    Delta = find_delta(
        *Delta_interval,
        *speed_interval,
        xs_left,
        xs_right,
        verbose=True,
        tol=1e-8,
        **params_dict,
    )
    c = find_c(
        *speed_interval, xs_right, Delta=Delta, verbose=True, tol=1e-12, **params_dict
    )
    print(f"c={c}\nDelta={Delta}")

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


solver = TqdmWrapper(Euler())

# fig, ax = plt.subplots(num='Const stim')
# ax.set_xlim(-15, 50)
# u_base_line, = ax.plot(space.array, u0[0], 'g-')
# q_base_line, = ax.plot(space.array, u0[1], 'g--')
# u_line, = ax.plot(space.array, u0[0], 'b-')
# q_line, = ax.plot(space.array, u0[1], 'b--')
# for perturbed, base in zip(solver.solution_generator(u0, rhs, time),
#                            solver.solution_generator(u0, model.rhs, time)):
#
#     u_line.set_ydata(perturbed[0])
#     q_line.set_ydata(perturbed[1])
#     u_base_line.set_ydata(base[0])
#     q_base_line.set_ydata(base[1])
#     plt.pause(0.01)

base_fronts = []
print("Unperturbed case.")
for u, q in solver.solution_generator(u0, model.rhs, time):
    front = find_roots(
        space.inner, u[space.inner_slice] - params_dict["theta"], window=3
    )[-1]
    base_fronts.append(front)
base_fronts = np.array(base_fronts)

eps_list = [0.001, 0.002, 0.004]
front_list = []
for eps in eps_list:
    print(f"eps={eps}")
    const_stim = np.zeros_like(u0)
    const_stim[0] += eps
    fronts = []
    for u, q in solver.solution_generator(
        u0, lambda u, t: model.rhs(u, t) + const_stim, time
    ):
        front = find_roots(
            space.inner, u[space.inner_slice] - params_dict["theta"], window=3
        )[-1]
        fronts.append(front)

    front_list.append(np.array(fronts))

plt.figure()
for eps, fronts in zip(eps_list, front_list):
    plt.plot(time.array, fronts - base_fronts, ".", label=f"{eps}")
plt.legend()

plt.plot(time.array, time.array * c, "r-")


dU = np.array([local_diff(z, space.array, Us, width=2) for z in space.array])
dQ = np.array([local_diff(z, space.array, Qs, width=2) for z in space.array])
A0, AmD = nullspace_amplitudes(space.array, Us, Qs, c, Delta, **params_dict)
v1_arr = v1(space.array, A0=A0, AmD=AmD, c=c, Delta=Delta, **params.dict)
v2_arr = v2(space.array, A0=A0, AmD=AmD, c=c, Delta=Delta, **params_dict)

quad_dom = Domain(space.array[0], space.array[-1], len(space.array))
num = quad_dom.quad(v1_arr)
denom = quad_dom.quad(params.mu * dU * v1_arr + params.alpha * dQ * v2_arr)
slope = -num / denom * eps
plt.plot(time.array, slope * time.array, "k:")
