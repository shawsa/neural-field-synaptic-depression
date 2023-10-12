import experiment_defaults

import numpy as np
import matplotlib.pyplot as plt
import os.path
import pickle

from functools import partial
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

from apparent_motion_utils import (
    ShiftingDomain,
    ShiftingEuler,
    ApparentMotionStimulus,
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
from neural_field_synaptic_depression.neural_field import (
    NeuralField,
    ParametersBeta,
    heaviside_firing_rate,
    exponential_weight_kernel,
)
from neural_field_synaptic_depression.root_finding_helpers import find_roots
from neural_field_synaptic_depression.time_domain import (
    TimeRay,
    TimeDomain_Start_Stop_MaxSpacing,
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
HI_RES_FIG_FILE = os.path.join(experiment_defaults.media_path, "fig6_hi_res.png")

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

ax_contour.text(0.1, 0.1, "Success")
ax_contour.text(0.005, 0.7, "Failure", rotation=90)
ax_contour.set_title("Entrainment \nby $T_{on}/T$ ratio")
# ax_contour.legend(loc="upper right")
ax_contour.set_xlabel(r"$\varepsilon$")
ax_contour.set_ylabel(r"$\Delta_c$")

# panel A - entrainment success
space = ShiftingDomain(-40, 40, 6_001)
model = NeuralField(
    space=space,
    firing_rate=partial(heaviside_firing_rate, theta=theta),
    weight_kernel=exponential_weight_kernel,
    params=params,
)

solver = ShiftingEuler(shift_tol=1e-4, shift_fraction=4 / 5, space=space)

u0 = np.empty((2, space.num_points))
u0[0] = U(space.array)
u0[1] = Q(space.array)
stim = ApparentMotionStimulus(
    **{
        "t_on": 0.5,
        "t_off": 0.5,
        "speed": c + 0.5,
        "mag": 0.2,
        "width": 1,
        "start": -0.05,
    }
)
max_time_step = 1e-2
time_step = stim.period / np.ceil(stim.period / max_time_step)
time = TimeDomain_Start_Stop_MaxSpacing(0, 12, time_step)
# time = TimeDomain_Start_Stop_MaxSpacing(0, 8, time_step)


def rhs(t, u):
    return model.rhs(t, u) + stim(space.array, t)


fronts = []
space.reset()
for t, (u, q) in tqdm(
    zip(time, solver.solution_generator(u0, rhs, time)), total=len(time.array)
):
    front = find_roots(space.array, u - theta, window=3)[-1]
    fronts.append(front)


ax_success = fig.add_subplot(grid[0, 0])
time_on = stim.next_on(0)
while time_on < time.array[-1]:
    stim_front = stim.front(time_on)
    ax_success.fill_between(
        [time_on, time_on + stim.t_on],
        [stim_front] * 2,
        [stim_front - stim.width] * 2,
        color="magenta",
    )
    time_on = stim.next_on(time_on)
ax_success.plot(time.array, fronts, "b-")
ax_success.set_xlabel("$t$")
ax_success.set_ylabel("$x$")
ax_success.set_title("Entrainment \nSuccess")
label_x_loc = 5
ax_success.plot(label_x_loc, 2, "b.")
ax_success.text(label_x_loc + 0.5, 1.8, "pulse")
ax_success.fill_between(
    [label_x_loc - 0.2, label_x_loc + 0.2], [0.6] * 2, [0.8] * 2, color="magenta"
)
ax_success.text(label_x_loc + 0.5, 0.3, "stim")

# panel B - entrainment failure
stim = ApparentMotionStimulus(
    **{
        "t_on": 0.5,
        "t_off": 0.5,
        "speed": c + 0.5,
        "mag": 0.12,
        "width": 1,
        "start": -0.05,
    }
)
time = TimeDomain_Start_Stop_MaxSpacing(0, 12, time_step)


def rhs(t, u):
    return model.rhs(t, u) + stim(space.array, t)


fronts = []
space.reset()
for t, (u, q) in tqdm(
    zip(time, solver.solution_generator(u0, rhs, time)), total=len(time.array)
):
    front = find_roots(space.array, u - theta, window=3)[-1]
    fronts.append(front)


ax_failure = fig.add_subplot(grid[0, 1])
time_on = stim.next_on(0)
while time_on < time.array[-1]:
    stim_front = stim.front(time_on)
    ax_failure.fill_between(
        [time_on, time_on + stim.t_on],
        [stim_front] * 2,
        [stim_front - stim.width] * 2,
        color="magenta",
    )
    time_on = stim.next_on(time_on)
ax_failure.plot(time.array, fronts, "b-")
ax_failure.set_title("Entrainment \nFailure")
ax_failure.set_xlabel("$t$")
ax_failure.set_ylabel("$x$")


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
plt.savefig(HI_RES_FIG_FILE, dpi=300)
for ext in [".png", ".pdf"]:
    plt.savefig(FIG_FILE + ext)
