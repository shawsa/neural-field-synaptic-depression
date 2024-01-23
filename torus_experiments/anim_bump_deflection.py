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


FILE_NAME = os.path.join(experiment_defaults.media_path, "bump_deflection.gif")

# parameter specification

space = SpaceDomain2D(
    -15,
    50,
    200,
    -30,
    30,
    200,
)
time = TimeDomain_Start_Stop_MaxSpacing(0, 600, 1e-2)

bump_speed = 0.0705
stim_speed_delta = 0.01
stim_mag = .3
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


solver = TqdmWrapper(Euler())

cmap = "seismic"
norm = Normalize(vmin=-2, vmax=2)
efficacy_variance = 0.5
efficacy_norm = Normalize(vmin=(1 - efficacy_variance), vmax=(1 + efficacy_variance))

figsize = (12, 5)
fig = plt.figure(figsize=figsize)
grid = GridSpec(1, 2)
ax_activity = fig.add_subplot(grid[0, 0])
ax_efficacy = fig.add_subplot(grid[0, 1])

for ax in (ax_activity, ax_efficacy):
    ax.plot([-1000, 1000], [0, 0], "k:")
    ax.plot(*list(zip(*map(stim_path, [-1000, 1000]))), "g:")
    ax.set_xlim(np.min(space.X), np.max(space.X))
    ax.set_ylim(np.min(space.Y), np.max(space.Y))

activity_color = ax_activity.pcolormesh(space.X, space.Y, u0[0], cmap=cmap, norm=norm)
plt.colorbar(activity_color)
ax_activity.set_title("Activity")

efficacy_color = ax_efficacy.pcolormesh(
    space.X, space.Y, u0[1], cmap=cmap, norm=efficacy_norm
)
plt.colorbar(efficacy_color)
ax_efficacy.set_title("Efficacy")

(peak,) = ax_activity.plot([0], [0], "k.")
(stim_loc,) = ax_activity.plot(*stim_path(0), "g*")


with imageio.get_writer(FILE_NAME, mode="I") as writer:
    for step, (t, u) in enumerate(zip(time, solver.solution_generator(u0, rhs, time))):
        if not step % steps_per_frame == 0:
            continue
        activity_color.set_array(u[0].ravel())
        efficacy_color.set_array(u[1].ravel())
        peak.set_data(*get_peak(u[0]))
        stim_loc.set_data(*stim_path(t))
        plt.savefig(FILE_NAME + ".png")
        image = imageio.imread(FILE_NAME + ".png")
        writer.append_data(image)
        plt.pause(1e-3)

os.remove(FILE_NAME + ".png")
plt.close()
