import experiment_defaults

import imageio
import matplotlib.pyplot as plt
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

FILE_NAME = os.path.join(experiment_defaults.media_path, "generation.gif")

params = Parameters(mu=1.0, alpha=50.0, gamma=.1)
linspace_params = (-100, 100, 401)
time = TimeDomain_Start_Stop_MaxSpacing(0, 30, 1e-2)

space = SpaceDomain2D(*linspace_params, *linspace_params)

u0 = np.ones((2, *space.X.shape))
u0[0] = space.dist(0, 0) < 10


def firing_rate(u, theta=0.2):
    # return u > 0.2
    return 1 / (1 + np.exp(-20 * (u - theta)))


def weight_kernel(r):
    return np.exp(-r) / (2 * np.pi)


model = NeuralField2D(
    space=space,
    firing_rate=firing_rate,
    weight_kernel=weight_kernel,
    params=params,
)

solver = TqdmWrapper(Euler())

plt.figure("Wave")
color_mesh = plt.pcolormesh(space.X, space.Y, u0[0], cmap="gnuplot")
plt.colorbar(color_mesh)

with imageio.get_writer(FILE_NAME, mode="I") as writer:
    for u in islice(solver.solution_generator(u0, model.rhs, time), None, None, 10):
        color_mesh.set_array(u[0].ravel())
        plt.savefig(FILE_NAME + ".png")
        image = imageio.imread(FILE_NAME + ".png")
        writer.append_data(image)
        plt.pause(1e-3)

os.remove(FILE_NAME + ".png")
plt.close()
