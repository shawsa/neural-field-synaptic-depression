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

FILE_NAME = os.path.join(experiment_defaults.media_path, "bump.gif")

param = Parameters(mu=1.0, alpha=20.0, gamma=0.5)
linspace_params = (-15, 15, 201)
time = TimeDomain_Start_Stop_MaxSpacing(0, 300, 1e-2)

space = SpaceDomain2D(*linspace_params, *linspace_params)

u0 = np.ones((2, *space.X.shape))
u0[0] = space.dist(0, 0) < 1
u0[1] += (space.dist(0, 0) < 1) * space.X/2


def firing_rate(u, theta=0.2):
    # return u > 0.2
    return 1 / (1 + np.exp(-20 * (u - theta)))


def weight_kernel(r):
    # return np.exp(-r) * (3 - r) / (2 * np.pi)
    # return np.exp(-r) * (2 - r)
    return np.exp(-r) * (1 - r) * 2


def gaussian(x, y):
    return np.exp(-space.dist(x, y) ** 2) / (2 * np.pi)


model = NeuralField2D(
    space=space,
    firing_rate=firing_rate,
    weight_kernel=weight_kernel,
    params=params,
)

forcing_radius = 2
speed = 0.1
magnitude = 1
speed_factor = speed / forcing_radius


def forcing(t):
    # x_center = forcing_radius * np.sin(t*speed_factor)
    # y_center = forcing_radius/2 * (1 + np.cos(t*speed_factor))
    # return magnitude*gaussian(x_center, y_center)
    return 0


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

activity_color = ax_activity.pcolormesh(space.X, space.Y, u0[0], cmap=cmap, norm=norm)
plt.colorbar(activity_color)
ax_activity.set_title("Activity")

efficacy_color = ax_efficacy.pcolormesh(
    space.X, space.Y, u0[1], cmap=cmap, norm=efficacy_norm
)
plt.colorbar(efficacy_color)
ax_efficacy.set_title("Efficacy")

peak, = ax_activity.plot(*get_peak(u0[0]), "k.")

steps_per_frame = 100

with imageio.get_writer(FILE_NAME, mode="I") as writer:
    for u in islice(solver.solution_generator(u0, rhs, time), None, None, steps_per_frame):
        activity_color.set_array(u[0].ravel())
        efficacy_color.set_array(u[1].ravel())
        peak.set_data(*get_peak(u[0]))
        plt.savefig(FILE_NAME + ".png")
        image = imageio.imread(FILE_NAME + ".png")
        writer.append_data(image)
        plt.pause(1e-3)

os.remove(FILE_NAME + ".png")
plt.close()
