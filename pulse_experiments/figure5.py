#!/usr/bin/python3
"""
A driver for the neural field simulator. Consider this a manual test of
most of the functionality.
"""

import experiment_defaults

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os.path
import pickle
from functools import partial
from itertools import islice
from matplotlib.gridspec import GridSpec

from neural_field import (
    NeuralField,
    ParametersBeta,
    heaviside_firing_rate,
    exponential_weight_kernel,
)
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
from plotting_helpers import make_animation
from root_finding_helpers import find_roots
from space_domain import SpaceDomain, BufferedSpaceDomain
from time_domain import TimeDomain, TimeDomain_Start_Stop_MaxSpacing
from time_integrator import Euler, EulerDelta
from time_integrator_tqdm import TqdmWrapper

from scipy.stats import linregress
from more_itertools import windowed

DATA_FILE = os.path.join(experiment_defaults.data_path, "entrainment_square.pickle")

FIG_FILE = os.path.join(experiment_defaults.media_path, "figure5")

plt.rcParams.update(
    {
        "font.size": 12,
        "text.usetex": True,
    }
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
time = TimeDomain_Start_Stop_MaxSpacing(0, 40, 1e-3)

initial_offset = 0
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

stim_start = -20
stim_magnitude = 0.1
stim_width = 10
fronts_dict = {}

for pannel, stim_speed in [("success", 3.3), ("failure", 3.5)]:

    def stim_func(t):
        # return stim_magnitude*np.exp(-np.abs(space.array - stim_start - stim_speed*t)**2)
        def square(xs, center, width):
            return np.heaviside(xs - center + width / 2, 0.5) * np.heaviside(
                center - xs + width / 2, 0.5
            )

        return stim_magnitude * square(
            space.array, stim_start + stim_speed * t, stim_width
        )

    def rhs(t, u):
        stim = np.zeros_like(u0)
        stim[0] = stim_func(t)
        return model.rhs(t, u) + stim

    fronts = []
    for t, (u, q) in zip(time.array, solver.solution_generator(u0, rhs, time)):
        fronts.append(
            find_roots(
                space.inner, u[space.inner_slice] - params_dict["theta"], window=3
            )[-1]
        )
    fronts_dict[pannel] = (stim_speed, fronts)


figsize = (7, 2.5)
fig = plt.figure(figsize=figsize)
grid = GridSpec(1, 3)

ax_success = fig.add_subplot(grid[0, 0])
ax_failure = fig.add_subplot(grid[0, 1], sharey=ax_success)
ax_contour = fig.add_subplot(grid[0, 2])

for ax, pannel in [(ax_success, "success"), (ax_failure, "failure")]:
    stim_speed, fronts = fronts_dict[pannel]

    ax.plot(time.array[1:], fronts[1:], "b-", label="stimulated front")
    ax.plot(time.array, c * time.array, "b--", label="unstimulated front")
    ax.plot(time.array, stim_speed * time.array + stim_start, "m-", label="stim center")
    # ax.legend()
    ax.set_title(f"Entrainment {pannel}")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$x$")

ax_success.text(-1, 15, "Pulse")
ax_success.text(4, -20, "Stimulus")

# contour plot
with open(DATA_FILE, "rb") as f:
    (
        stim_magnitudes,
        stim_speeds,
        results,
        stim_width,
        slope,
        params,
        params_dict,
    ) = pickle.load(f)

mag_mat, speed_mat = np.meshgrid(stim_magnitudes, stim_speeds)

res_mat = np.zeros_like(mag_mat, dtype=bool)
lag_mat = np.zeros_like(mag_mat, dtype=float)
lag_mat[:] = np.nan
for index, (mag, speed) in enumerate(zip(mag_mat.flat, speed_mat.flat)):
    for sol in results:
        if sol["stim_magnitude"] == mag and sol["stim_speed"] == speed:
            np.ravel(res_mat)[index] = sol["entrained"]
            if sol["entrained"]:
                np.ravel(lag_mat)[index] = -sol["relative_stim_position"]
            break

color_mesh = ax_contour.pcolormesh(
    mag_mat,
    speed_mat,
    lag_mat,
    # lag_mat[:-1, :-1],
    # shading="flat",
    shading="nearest",
    vmin=0,
    vmax=10,
    cmap=matplotlib.colormaps["plasma"]
)

plt.colorbar(color_mesh, ax=ax_contour, fraction=0.05, ticks=[0, 10])
ax_contour.text(1.15, .4, "Lag", rotation=90, transform=ax_contour.transAxes)

# contour_set = plt.contour(mag_mat, speed_mat, res_mat, [0.5], colors=['green'])
# ax_contour.plot(
#     [mag for mag, entrained in zip(mag_mat.flat, res_mat.flat) if entrained],
#     [speed for speed, entrained in zip(speed_mat.flat, res_mat.flat) if entrained],
#     "+",
#     color="green",
#     label="Entrained",
#     markersize=4.0,
#     alpha=0.5
# )
# ax_contour.plot(
#     [mag for mag, entrained in zip(mag_mat.flat, res_mat.flat) if not entrained],
#     [speed for speed, entrained in zip(speed_mat.flat, res_mat.flat) if not entrained],
#     "x",
#     color="magenta",
#     label="Failed to Entrain",
#     markersize=4.0,
#     alpha=0.5,
# )
ax_contour.plot(
    stim_magnitudes, params_dict["c"] + stim_magnitudes * slope, "k-", label="Theory"
)
ax_contour.set_ylim(1.0, 2.6)
# ax_contour.text(0.03, 1.1, "Success", rotation=45, fontsize="x-large")
ax_contour.text(0.0, 1.7, "Failure", rotation=45)
# ax_contour.text(0.0, 2.3, "Entrainment")
# ax_contour.text(0.05, 1.5, "Non-Entrainment")
ax_contour.set_xlabel("Stimulus Magnitude")
ax_contour.set_ylabel("Stimulus Speed")
ax_contour.set_title("Critical Speed")

grid.tight_layout(fig, w_pad=0.5, h_pad=-0.5)

subplot_label_x = -0.15
subplot_label_y = 1.05
subplot_label_font = {
    "size": "x-large",
    "weight": "bold",
}

for label, ax in [("A", ax_success), ("B", ax_failure), ("C", ax_contour)]:
    ax.text(
        subplot_label_x,
        subplot_label_y,
        label,
        fontdict=subplot_label_font,
        transform=ax.transAxes,
    )

for ext in [".png", ".eps", ".pdf"]:
    plt.savefig(FIG_FILE + ext)
