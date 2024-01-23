import experiment_defaults

import imageio
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec

import numpy as np
import os.path

from functools import partial
from itertools import islice
from more_itertools import windowed
from scipy.stats import linregress

from neural_field_synaptic_depression.neural_field_2d import (
    SpaceDomain2D,
    NeuralField2D,
    Parameters,
)
from neural_field_synaptic_depression.time_domain import (
    TimeDomain,
    TimeDomain_Start_Stop_MaxSpacing,
)
from neural_field_synaptic_depression.time_integrator import Euler
from neural_field_synaptic_depression.time_integrator_tqdm import TqdmWrapper

from bump_init import params, firing_rate, weight_kernel, get_initial_condition


def gaussian(x, y):
    return np.exp(-space.dist(x, y) ** 2) / (2 * np.pi)


IMAGE_FILE = os.path.join(experiment_defaults.media_path, "figure7")

# parameter specification

space = SpaceDomain2D(
    -15,
    50,
    200,
    -30,
    30,
    200,
)
time0 = TimeDomain_Start_Stop_MaxSpacing(-10e-2, 0, 1e-2)
time1 = TimeDomain_Start_Stop_MaxSpacing(0, 300, 1e-2)
time2 = TimeDomain_Start_Stop_MaxSpacing(300, 600, 1e-2)

bump_speed = 0.0705
stim_speed_delta = 0.01
stim_mag = None
stim_angle = np.pi/12

t_intercept = 300

steps_per_frame = 500
# end parameter specification

# unperturbed intercept point
stim_speed = bump_speed + stim_speed_delta
x_intercept_ratio = 3.525/50
stim_intercept = np.array([t_intercept * x_intercept_ratio, 0])

stim_velocity = np.array(
    [
        np.cos(stim_angle) * stim_speed,
        np.sin(stim_angle) * stim_speed,
    ]
)


def stim_path(t: float):
    return stim_intercept + (t - t_intercept) * stim_velocity


u0 = get_initial_condition(space)
model = NeuralField2D(
    space=space,
    firing_rate=firing_rate,
    weight_kernel=weight_kernel,
    params=params,
)


def forcing(t):
    return stim_mag * gaussian(*stim_path(t))


def rhs(t, v):
    ret = model.rhs(t, v)
    ret[0] += forcing(t)
    return ret


def get_peak(arr):
    index = np.argmax(arr.ravel())
    return space.X.ravel()[index], space.Y.ravel()[index]


# successful entrainment
stim_mag = .3
solver = TqdmWrapper(Euler())

u_e0 = solver.t_final(u0, rhs, time0)
u_e1 = solver.t_final(u_e0, rhs, time1)
u_e2 = solver.t_final(u_e1, rhs, time2)

# failed entrainment
stim_mag = .2
solver = TqdmWrapper(Euler())

u_f0 = solver.t_final(u0, rhs, time0)
u_f1 = solver.t_final(u_f0, rhs, time1)
u_f2 = solver.t_final(u_f1, rhs, time2)

# Make Figure

cmap = "seismic"
norm = Normalize(vmin=-2, vmax=2)
efficacy_variance = 0.5
efficacy_norm = Normalize(vmin=(1 - efficacy_variance), vmax=(1 + efficacy_variance))

figsize = (7, 2.5)
fig = plt.figure(figsize=figsize)
grid = GridSpec(2, 3)

time_x, time_y = -5, 5

# entrainment panel 1
ax_e1 = fig.add_subplot(grid[0, 0])
ax_e1.pcolormesh(space.X, space.Y, u_e0[0], cmap=cmap, norm=norm)
ax_e1.plot(*stim_path(time1.start), "g*")
ax_e1.text(time_x, time_y, f"time={time1.start}")

# entrainment panel 2
ax_e2 = fig.add_subplot(grid[0, 1])
ax_e2.pcolormesh(space.X, space.Y, u_e1[0], cmap=cmap, norm=norm)
ax_e2.plot(*stim_path(time2.start), "g*")
ax_e2.text(time_x, time_y, f"time={time2.start}")

# entrainment panel 3
ax_e3 = fig.add_subplot(grid[0, 2])
ax_e3.pcolormesh(space.X, space.Y, u_e2[0], cmap=cmap, norm=norm)
ax_e3.plot(*stim_path(time2.array[-1]), "g*")
ax_e3.text(time_x, time_y, f"time={time2.array[-1]}")

# failure panel 1
ax_f1 = fig.add_subplot(grid[1, 0])
ax_f1.pcolormesh(space.X, space.Y, u_f0[0], cmap=cmap, norm=norm)
ax_f1.plot(*stim_path(time1.start), "g*")
ax_f1.text(time_x, time_y, f"time={time1.start}")

# failure panel 2
ax_f2 = fig.add_subplot(grid[1, 1])
ax_f2.pcolormesh(space.X, space.Y, u_f1[0], cmap=cmap, norm=norm)
ax_f2.plot(*stim_path(time2.start), "g*")
ax_f2.text(time_x, time_y, f"time={time2.start}")

# failure panel 3
ax_f3 = fig.add_subplot(grid[1, 2])
ax_f3.pcolormesh(space.X, space.Y, u_f2[0], cmap=cmap, norm=norm)
ax_f3.plot(*stim_path(time2.array[-1]), "g*")
ax_f3.text(time_x, time_y, f"time={time2.array[-1]}")

for ax in (ax_e1, ax_e2, ax_e3, ax_f1, ax_f2, ax_f3):
    ax.plot([-1000, 1000], [0, 0], "k:")
    ax.plot(*list(zip(*map(stim_path, [-1000, 1000]))), "g:")
    ax.set_xlim(-10, 50)
    ax.set_ylim(-10, 15)

for ax in (ax_e1, ax_e2, ax_e3):
    ax.set_xticks([])

for ax in (ax_f1, ax_f2, ax_f3):
    ax.set_xlabel("$x$")

for ax in (ax_e2, ax_e3, ax_f2, ax_f3):
    ax.set_yticks([])

for ax in (ax_e1, ax_f1):
    ax.set_ylabel("$y$")

subplot_label_font = {
    "size": "x-large",
    "weight": "bold",
}
for ax, label in [
        (ax_e1, "A"),
        (ax_e2, "B"),
        (ax_e3, "C"),
        (ax_f1, "D"),
        (ax_f2, "E"),
        (ax_f3, "F"),
]:
    ax.text(
        0.15,
        1.04,
        label,
        fontdict=subplot_label_font,
        transform=ax.transAxes,
    )

grid.tight_layout(fig)

for ext in [".pdf", ".png"]:
    plt.savefig(IMAGE_FILE + ext)
