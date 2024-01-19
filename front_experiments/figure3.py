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
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

from neural_field_synaptic_depression.neural_field import (
    NeuralField,
    Parameters,
    heaviside_firing_rate,
    exponential_weight_kernel,
)
from neural_field_synaptic_depression.space_domain import SpaceDomain
from neural_field_synaptic_depression.time_domain import (
    TimeDomain_Start_Stop_MaxSpacing,
)
from neural_field_synaptic_depression.time_integrator import EulerDelta
from neural_field_synaptic_depression.time_integrator_tqdm import TqdmWrapper
from neural_field_synaptic_depression.root_finding_helpers import find_roots

from adaptive_front import (
    U_numeric,
    Q_numeric,
    get_speed,
    response,
)

plt.rcParams.update(
    {
        "font.size": 12,
        "text.usetex": True,
    }
)

FIG_FILE_NAME = os.path.join(experiment_defaults.media_path, "figure3")
DATA_FILE_NAME = os.path.join(
    experiment_defaults.data_path, "spatially_homogeneous_limit.pickle"
)

ERROR_FILE_NAME = os.path.join(
    experiment_defaults.data_path, "figure3_error.pickle"
)


# heatmap panel
params = Parameters(mu=1.0, alpha=20.0, gamma=0.2)
theta = 0.1

space = SpaceDomain(-100, 200, 10**4)
time = TimeDomain_Start_Stop_MaxSpacing(0, 18, 1e-3 / 5)

initial_offset = 0
u0 = np.empty((2, space.num_points))
u0[0] = U_numeric(space.array + initial_offset, theta=theta, **params.dict)
u0[1] = Q_numeric(space.array + initial_offset, theta=theta, **params.dict)

model = NeuralField(
    space=space,
    firing_rate=partial(heaviside_firing_rate, theta=theta),
    weight_kernel=exponential_weight_kernel,
    params=params,
)

delta_time = 1
epsilon = 0.09
pulse_profile = np.ones_like(u0)
pulse_profile[1] *= 0

solver = TqdmWrapper(EulerDelta(delta_time, epsilon * pulse_profile))

print("solving purturbed case")
us = solver.solve(u0, model.rhs, time)

print("finding front locations")
fronts = fronts = [
    find_roots(space.array, u - theta, window=7)[-1] for u, q in tqdm(us)
]

speed = get_speed(theta=theta, **params.dict)

x_window = (-20, 70)
x_window_indices = [np.argmin(np.abs(x - space.array)) for x in x_window]

x_pixels = 500
x_stride = (x_window_indices[1] - x_window_indices[0]) // x_pixels
x_slice = slice(x_window_indices[0], x_window_indices[1], x_stride)

t_stop_index = int(0.8 * len(time.array))
y_pixels = 500
y_stride = t_stop_index // y_pixels
y_slice = slice(0, t_stop_index, y_stride)

sol_array = np.array([u[0][x_slice] for u in us[y_slice]]).T

# figure

figsize = (7, 6)
grid = GridSpec(2, 2)
fig = plt.figure(figsize=figsize)

# simulation panel
ax_heatmap = fig.add_subplot(grid[0, 0])

heatmap = ax_heatmap.pcolormesh(
    time.array[y_slice], space.array[x_slice], sol_array, cmap="Greys"
)

ax_heatmap.plot(
    time.array[y_slice], fronts[y_slice], "b-", label="measured", linewidth=3
)
ax_heatmap.plot(
    time.array[y_slice],
    (
        speed * time.array[y_slice]
        - initial_offset
        + np.heaviside(time.array[y_slice] - delta_time, 0)
        * response(epsilon, theta=theta, **params.dict)
    ),
    "g--",
    linewidth=3,
    label="predicted",
)

plt.colorbar(heatmap, ax=ax_heatmap, label="$u$")
ax_heatmap.set_ylabel("$x$")
ax_heatmap.set_xlabel("$t$")
ax_heatmap.set_title(f"$I(x,t) = {epsilon} \\delta(t - {delta_time})$")
ax_heatmap.legend(loc="lower right")

# wave response panel

with open(DATA_FILE_NAME, "rb") as f:
    params, theta, epsilons, responses = pickle.load(f)

response_slope = response(1, **params.dict, theta=theta)

ax_response = fig.add_subplot(grid[0, 1])

ax_response.plot(epsilons, epsilons * response_slope, "k-", label="Theory")
ax_response.plot(epsilons, responses, "go", label="Simulation")

ax_response.set_xlabel("$\\varepsilon$")
ax_response.set_ylabel("$\\nu_\\infty$")
ax_response.set_xticks([-0.08, 0, 0.08])
ax_response.set_yticks([-4, 0, 4, 8])
ax_response.set_title("Front response to global stimulus")
ax_response.legend(loc="upper left")


# error panel

with open(ERROR_FILE_NAME, "rb") as f:
    data_list = pickle.load(f)

alphas = []
gammas = []
epsilons = []
data_dict = {}
for params, theta, epsilon, speed, measured_response in data_list:
    alphas.append(params.alpha)
    gammas.append(params.gamma)
    epsilons.append(epsilon)
    data_dict[(params.alpha, params.gamma, epsilon)] = measured_response[-1]

alphas = sorted(list(set(alphas)))
gammas = sorted(list(set(gammas)))
epsilons = sorted(list(set(epsilons)))

error_dict = {}
e1, e2 = epsilons
for alpha, gamma in product(alphas, gammas):
    r1 = data_dict[(alpha, gamma, e1)]
    r2 = data_dict[(alpha, gamma, e2)]
    approx = (r2 - r1)/(e2 - e1)
    response_slope = response(1, mu=1.0, alpha=alpha, beta=1/gamma-1, theta=theta)
    error_dict[(alpha, gamma)] = abs((response_slope - approx)/approx)

alpha_mat, gamma_mat = np.meshgrid(alphas, gammas)
err_mat = np.zeros_like(alpha_mat)
for index, (alpha, gamma) in enumerate(zip(alpha_mat.ravel(), gamma_mat.ravel())):
    err_mat.ravel()[index] = error_dict[(alpha, gamma)]

ax_error = fig.add_subplot(grid[1, 1])
heatmap = ax_error.pcolormesh(
    alpha_mat, gamma_mat, err_mat, cmap="viridis"
)
plt.colorbar(heatmap, ax=ax_error, label="error")
ax_error.set_xlabel("$\\tau_q$")
ax_error.set_ylabel("$\\gamma$")
ax_error.set_title("Relative Error")

# theory plot

slope_mat = response(1, alpha=alpha_mat, beta=1/gamma_mat-1, theta=theta, mu=1.0)
ax_theory = fig.add_subplot(grid[1, 0])
heatmap = ax_theory.pcolormesh(
    alpha_mat, gamma_mat, slope_mat, cmap="viridis"
)
plt.colorbar(heatmap, ax=ax_theory, label="slope")
ax_theory.set_xlabel("$\\tau_q$")
ax_theory.set_ylabel("$\\gamma$")
ax_theory.set_title("Wave response slope")


# subplot labels
subplot_label_font = {
    "size": "x-large",
    "weight": "bold",
}
for ax, label in [
        (ax_heatmap, "A"),
        (ax_response, "B"),
        (ax_theory, "C"),
        (ax_error, "D"),
]:
    ax.text(
        -0.15,
        1.04,
        label,
        fontdict=subplot_label_font,
        transform=ax.transAxes,
    )

plt.tight_layout()
for extension in [".png", ".pdf"]:
    plt.savefig(os.path.join(experiment_defaults.media_path, FIG_FILE_NAME + extension))
