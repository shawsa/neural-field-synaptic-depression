import experiment_defaults

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import os.path
import pickle

from pqdm.processes import pqdm
from tqdm import tqdm

from collections import deque
from functools import partial
from itertools import product

from neural_field import (
    NeuralField,
    ParametersBeta,
    heaviside_firing_rate,
    exponential_weight_kernel,
)

from helper_symbolics import (
    expr_dict,
    find_symbol_by_string,
    free_symbols_in,
    get_traveling_pulse,
    get_adjoint_nullspace,
    get_numerical_parameters,
    recursive_reduce,
    symbolic_dictionary,
)

from num_assist import Domain  # for quadrature

from root_finding_helpers import find_roots

from time_domain import TimeRay, TimeDomain_Start_Stop_MaxSpacing

from apparent_motion_utils import (
    ShiftingDomain,
    ShiftingEuler,
    ApparentMotionStimulus,
)

DATA_PATH = os.path.join(experiment_defaults.data_path, "figure6.pickle")

NUM = 41

stim_mags = np.linspace(0, 0.2, NUM)[1:]
stim_speed_deltas = np.linspace(0, 1.5, NUM)[1:]
stim_width = 10
stim_start = -0.05

on_off_pairs = [(0.1, 0.5), (0.5, 0.5), (0.5, 0.1)]

print(f"number of sims: {len(stim_mags)*len(stim_speed_deltas)*len(on_off_pairs)}")

params = ParametersBeta(
    **{
        "alpha": 20.0,
        "beta": 5.0,
        "mu": 1.0,
    }
)
theta = 0.2
c, Delta = 1.0509375967740198, 9.553535461425781

# calculate slope factor
symbol_params = symbolic_dictionary(c=c, Delta=Delta, theta=theta, **params.dict)
symbol_params = get_numerical_parameters(symbol_params, validate=False)
U, Q, U_prime, Q_prime = get_traveling_pulse(symbol_params, validate=False)
v1, v2 = get_adjoint_nullspace(symbol_params, validate=False)
# find D from asymptotic approximation
dom = Domain(-30, 30, 2001)
D = -dom.quad(
    params.mu * U_prime(dom.array) * v1(dom.array)
    + params.alpha * Q_prime(dom.array) * v2(dom.array)
)


# initialize sim
space = ShiftingDomain(-25, 30, 6_001)
model = NeuralField(
    space=space,
    firing_rate=partial(heaviside_firing_rate, theta=theta),
    weight_kernel=exponential_weight_kernel,
    params=params,
)

solver = ShiftingEuler(shift_tol=1e-3, shift_fraction=4 / 5, space=space)

u0 = np.empty((2, space.num_points))
u0[0] = U(space.array)
u0[1] = Q(space.array)


if False:
    my_time = TimeDomain_Start_Stop_MaxSpacing(0, 300, 1e-3)
    space.reset()
    for u, q in tqdm(
        solver.solution_generator(u0, model.rhs, my_time), total=len(my_time.array)
    ):
        pass
    front = find_roots(space.array, u - theta, window=3)[-1]
    sim_speed = front / my_time.array[-1]
else:
    sim_speed = 1.0294047249587208
print(f"{sim_speed=}")


# stim = ApparentMotionStimulus(t_on, speed=sim_speed, mag=0)
# slope = 1 / (stim.t_on / stim.period * c * params.mu / D)
slope_factor = c * params.mu / D

MAX_SIM_TIME = 10_000

if "results" not in locals():
    results = dict()

my_tqdm = tqdm(list(product(stim_mags, stim_speed_deltas, on_off_pairs)))
for mag, delta, (t_on, t_off) in my_tqdm:
    my_key = (mag, delta, t_on, t_off)
    if my_key in results.keys():
        continue
    stim = ApparentMotionStimulus(
        t_on=t_on,
        t_off=t_off,
        speed=sim_speed + delta,
        start=stim_start,
        width=stim_width,
        mag=mag,
    )
    max_time_step = 1e-2
    time_step = stim.period / np.ceil(stim.period / max_time_step)
    # time = TimeRay(0, time_step)
    time = TimeDomain_Start_Stop_MaxSpacing(0, MAX_SIM_TIME, time_step)

    def rhs(t, u):
        return model.rhs(t, u) + stim(space.array, t)

    space.reset()

    next_period_time = stim.next_off(0)
    lag_deq = deque([float("inf")] * 5, maxlen=5)
    stop_tol = 1e-3
    success = None
    for index, (t, (u, q)) in enumerate(
        zip(time, solver.solution_generator(u0, rhs, time))
    ):
        front = find_roots(space.array, u - theta, window=3)[-1]
        lag = front - stim.front(t)
        if not abs(t - next_period_time) < time.spacing / 2:
            continue
        # stimulus is turning off
        desc = f"({mag:.3f}, {delta:.3f}) l={lag:.3g}|ch={(lag-lag_deq[-1]):.3g}"

        my_tqdm.set_description(desc=desc.ljust(35, " "))
        next_period_time = stim.next_off(t + time.spacing)
        if (
            all(abs(lag2 - lag) < stop_tol for lag2 in lag_deq)
            or abs(sum(lag_deq) / len(lag_deq) - lag) < stop_tol
        ):
            results[my_key] = (True, index, t)
            break
        elif -lag > stim.width * 1.05:
            results[my_key] = (False, index, t)
            break
        lag_deq.append(lag)


if False:
    with open(DATA_PATH, "rb") as f:
        results = pickle.load(f)
if False:
    with open(DATA_PATH, "wb") as f:
        pickle.dump(results, f)

fig, axes = plt.subplots(
    1, len(on_off_pairs), figsize=(20, 7), sharex=True, sharey=True
)

axes_dict = {}
for ax, (t_on, t_off) in zip(axes, on_off_pairs):
    axes_dict[(t_on, t_off)] = ax
    ax.set_title(f"on/off ratio = {t_on/t_off:.2f}")
    plot_list = [0, stim_mags[-1]]
    ax.plot(
        plot_list,
        [x * slope_factor * t_on / (t_on + t_off) for x in plot_list],
        "b-",
    )
    ax.set_xlim(0, stim_mags[-1])
    ax.set_ylim(0, stim_speed_deltas[-1])

for (mag, delta, t_on, t_off), (success, index, t, u, q) in results.items():
    ax = axes_dict[(t_on, t_off)]
    if success:
        ax.plot(mag, delta, "g+")
    else:
        ax.plot(mag, delta, "mx")
