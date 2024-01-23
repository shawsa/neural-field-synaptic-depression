import experiment_defaults

from functools import reduce
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
import numpy as np
import os.path
import pickle

from neural_field_synaptic_depression.neural_field_2d import SpaceDomain2D

from scipy.interpolate import RectBivariateSpline

FILE_NAME = os.path.join(experiment_defaults.data_path, "bump_init.pickle")

with open(FILE_NAME, "rb") as f:
    params, X, Y, u0_data = pickle.load(f)

u_spline = RectBivariateSpline(X[0], Y[:, 0], u0_data[0].T)
q_spline = RectBivariateSpline(X[0], Y[:, 0], u0_data[1].T)


def firing_rate(u, theta=0.2):
    return 1 / (1 + np.exp(-20 * (u - theta)))


def weight_kernel(r):
    return np.exp(-r) * (1 - r) * 2


def get_initial_condition(space: SpaceDomain2D):
    u0 = np.zeros((2, *space.X.shape))
    mask = reduce(np.logical_and,
                  [
                      space.X >= np.min(X),
                      space.X <= np.max(X),
                      space.Y >= np.min(Y),
                      space.Y <= np.max(Y),
                  ])
    u0[0][mask] = u_spline(space.X[mask], space.Y[mask], grid=False)
    u0[1][mask] = q_spline(space.X[mask], space.Y[mask], grid=False)
    u0[0][~mask] = 0
    u0[1][~mask] = 1
    return u0


if __name__ == "__main__":
    # Test to make sure the data and interpolation function properly
    space = SpaceDomain2D(-40, 30, 1200, -25, 15, 120)
    u0 = get_initial_condition(space)

    cmap = "seismic"
    norm = Normalize(vmin=-2, vmax=2)
    efficacy_variance = 0.5
    efficacy_norm = Normalize(
        vmin=(1 - efficacy_variance), vmax=(1 + efficacy_variance)
    )
    figsize = (12, 5)
    fig = plt.figure(figsize=figsize)
    grid = GridSpec(1, 2)
    ax_activity = fig.add_subplot(grid[0, 0])
    ax_efficacy = fig.add_subplot(grid[0, 1])

    activity_color = ax_activity.pcolormesh(
        space.X, space.Y, u0[0], cmap=cmap, norm=norm
    )
    plt.colorbar(activity_color)
    ax_activity.set_title("Activity")

    efficacy_color = ax_efficacy.pcolormesh(
        space.X, space.Y, u0[1], cmap=cmap, norm=efficacy_norm
    )
    plt.colorbar(efficacy_color)
    ax_efficacy.set_title("Efficacy")
