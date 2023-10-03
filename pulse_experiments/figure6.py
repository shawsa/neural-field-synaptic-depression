import experiment_defaults

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os.path
import pickle

from apparent_motion_utils import ApparentMotionStimulus
from neural_field import ParametersBeta
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

plt.rcParams.update(
    {
        "font.size": 12,
        "text.usetex": True,
    }
)

DATA_FILE = os.path.join(experiment_defaults.data_path, "figure6.pickle")
FIG_FILE = os.path.join(experiment_defaults.media_path, "figure6")

# model constants

params = ParametersBeta(
    **{
        "alpha": 20.0,
        "beta": 5.0,
        "mu": 1.0,
    }
)
theta = 0.2
c, Delta = 1.0509375967740198, 9.553535461425781
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
slope_factor = c * params.mu / D

figsize = (7, 3)
grid = GridSpec(1, 3)
fig = plt.figure(figsize=figsize)

# panel C - contour plots
ax_contour = fig.add_subplot(grid[0, 2])

NUM = 41
stim_mags = np.linspace(0, 0.2, NUM)[1:]
stim_speed_deltas = np.linspace(0, 1.5, NUM)[1:]
on_off_pairs = [(0.1, 0.5), (0.5, 0.5), (0.5, 0.1)]
mag_mat, speed_mat = np.meshgrid(stim_mags, stim_speed_deltas)

with open(DATA_FILE, "rb") as f:
    results = pickle.load(f)

ax_contour.plot(mag_mat.ravel(), speed_mat.ravel(), "k,", alpha=0.25)
ax_contour.set_xlim(0, stim_mags[-1])
ax_contour.set_ylim(0, stim_speed_deltas[-1])
for (t_on, t_off), color in zip(on_off_pairs, ["blue", "green", "black"]):
    Z = np.zeros_like(mag_mat)
    for Z_index, (mag, delta) in enumerate(zip(mag_mat.ravel(), speed_mat.ravel())):
        if results[(mag, delta, t_on, t_off)][0]:
            Z.ravel()[Z_index] = 1
    ax_contour.contour(mag_mat, speed_mat, Z, levels=[0.5], colors=[color])
    plot_list = [0, stim_mags[-1]]
    ax_contour.plot(
        plot_list,
        [x * slope_factor * t_on / (t_on + t_off) for x in plot_list],
        color=color,
        linestyle="--",
        label=f"ratio={t_on/t_off:.1f}",
    )

ax_contour.text(.1, .1, "Success")
ax_contour.text(.005, .7, "Failure", rotation=90)
ax_contour.set_title("Entrainment \nby on/off ratio")
# ax_contour.legend(loc="upper right")
ax_contour.set_xlabel(r"$\varepsilon$")
ax_contour.set_ylabel(r"$\Delta_c$")

# panel A - entrainment success
ax_success = fig.add_subplot(grid[0, 0])

ax_success.set_title("Entrainment \nSuccess")

# panel B - entrainment success
ax_failure = fig.add_subplot(grid[0, 1])
ax_failure.set_title("Entrainment \nFailure")


# panel labels
subplot_label_font = {
    "size": "x-large",
    "weight": "bold",
}
for ax, label in [(ax_success, "A"), (ax_failure, "B"), (ax_contour, "C")]:
    ax.text(
        -0.15,
        1.04,
        label,
        fontdict=subplot_label_font,
        transform=ax.transAxes,
    )

plt.tight_layout()
for ext in [".png", ".pdf"]:
    plt.savefig(FIG_FILE + ext)
