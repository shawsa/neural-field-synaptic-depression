"""Generate figures for pulse-width and pulse-speed for a varaiety of parameter
combinations. Use data generated from `data_bifurcation.py`.
"""

import experiment_defaults

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from itertools import cycle
from matplotlib.gridspec import GridSpec

from data_speed_width_bifurcation import (
    SolutionSearch,
    Solution,
    Arguments,
    MaxRefinementReachedException,
)

# for pulse profile
from neural_field_synaptic_depression.neural_field import ParametersBeta
from helper_symbolics import (
    get_traveling_pulse,
    get_numerical_parameters,
    symbolic_dictionary,
)

# Bifurcation data
DATA_FILE = os.path.join(experiment_defaults.data_path, "smooth_bifurcation.pickle")
IMAGE_FILE = os.path.join(experiment_defaults.media_path, "figure2")
HI_RES_IMAGE_FILE = os.path.join(experiment_defaults.media_path, "hi_res_figure2.png")
with open(DATA_FILE, "rb") as f:
    alphas, gammas, stable_search, unstable_search = pickle.load(f)
alphas.sort()
gammas.sort()

plt.rcParams.update(
    {
        "font.size": 12,
        "text.usetex": True,
    }
)

# figsize = (15, 7)
figsize = (7, 4)

subplot_label_x = -0.05
subplot_label_y = 1.05
subplot_label_font = {
    "size": "x-large",
    "weight": "bold",
}

fig = plt.figure(figsize=figsize)
grid = GridSpec(12, 24)

col1_slice = slice(0, 7)
col2_slice = slice(9, 16)
col3_slice = slice(17, 24)
col2_color_slice = slice(10, 15)
col3_color_slice = slice(18, 23)

row1_slice = slice(1, 6)
row2_slice = slice(7, 12)
color_slice = slice(0, 1)
speed_slice = slice(2, 6)
width_slice = slice(7, 12)


ax_prof = fig.add_subplot(grid[row1_slice, col1_slice])
ax_prof_unstable = fig.add_subplot(grid[row2_slice, col1_slice])

ax_gamma_colorbar = fig.add_subplot(grid[color_slice, col2_color_slice])
ax_speed_alpha = fig.add_subplot(grid[speed_slice, col2_slice])
ax_width_alpha = fig.add_subplot(grid[width_slice, col2_slice], sharex=ax_speed_alpha)

ax_alpha_colorbar = fig.add_subplot(grid[color_slice, col3_color_slice])
ax_speed_gamma = fig.add_subplot(grid[speed_slice, col3_slice], sharey=ax_speed_alpha)
ax_width_gamma = fig.add_subplot(
    grid[width_slice, col3_slice], sharex=ax_speed_gamma, sharey=ax_width_alpha
)

for ax in [ax_speed_gamma, ax_width_gamma]:
    plt.setp(ax.get_yticklabels(), visible=False)

for ax in [ax_speed_gamma, ax_speed_alpha]:
    plt.setp(ax.get_xticklabels(), visible=False)

# profiles

param_dict = {
    "alpha": 20.0,
    "beta": 5.0,
    "mu": 1.0,
}
theta = 0.2
params = ParametersBeta(**param_dict)
# c, Delta =  1.0509375967740198, 9.553535461425781
for ax, sol_obj in [
    (
        ax_prof,
        stable_search.closest_to(Arguments(**param_dict, theta=theta)),
    ),
    (
        ax_prof_unstable,
        unstable_search.closest_to(Arguments(**param_dict, theta=theta)),
    ),
]:
    c = sol_obj.speed
    Delta = sol_obj.width
    symbol_params = symbolic_dictionary(c=c, Delta=Delta, theta=theta, **params.dict)
    symbol_params = get_numerical_parameters(symbol_params, validate=False)
    U, Q, *_ = get_traveling_pulse(symbol_params, validate=False)

    xis = np.linspace(-30, 20, 201)
    ax.plot(xis, U(xis), "b-")
    ax.text(5, 0.05, r"$U(\xi)$")
    ax.plot(xis, Q(xis), "b--")
    ax.text(5, 0.8, r"$Q(\xi)$")
    ax.plot(xis, 0 * xis + theta, "k:")
    ax.text(-20, 0.25, r"$\theta$")
    # add active region?
    ax.set_xlim(xis[0], xis[-1])
    ax.set_yticks([0, 1], [0, 1])
    ax.set_xticks([0], [r"$\xi = 0$"])

ax_prof.text(0.05, 0.85, "Stable", transform=ax_prof.transAxes)
ax_prof.text(
    subplot_label_x,
    subplot_label_y,
    "A",
    fontdict=subplot_label_font,
    transform=ax_prof.transAxes,
)

ax_prof_unstable.text(0.05, 0.45, "Unstable", transform=ax_prof_unstable.transAxes)
ax_prof_unstable.text(
    subplot_label_x,
    subplot_label_y,
    "D",
    fontdict=subplot_label_font,
    transform=ax_prof_unstable.transAxes,
)

# color info and defaults
plot_gammas = np.arange(0.2, 0.125, -0.005)
plot_alphas = np.arange(20.0, 11.0, -0.5)
cmap_alpha = matplotlib.cm.get_cmap("cool")
cmap_gamma = matplotlib.cm.get_cmap("winter")
gamma_color_norm = matplotlib.colors.Normalize(vmin=0.1, vmax=0.2)
gamma_colors = [cmap_gamma(gamma_color_norm(gamma)) for gamma in plot_gammas]
alpha_color_norm = matplotlib.colors.Normalize(vmin=11, vmax=20)
alpha_colors = [cmap_alpha(alpha_color_norm(alpha)) for alpha in plot_alphas]

boundary_color = "gold"
style_legend_color = "grey"
tol = 1e-8

###################################################
# alpha by speed
###################################################
stable_search.solutions.sort(key=lambda sol: sol.alpha)
unstable_search.solutions.sort(key=lambda sol: sol.alpha)
ax_speed_alpha.plot([], [], linestyle="-", color=style_legend_color, label="Stable")
ax_speed_alpha.plot([], [], linestyle=":", color=style_legend_color, label="Unstable")
for gamma, color in zip(plot_gammas, cycle(gamma_colors)):
    if gamma == 0.2:
        fill_alpha, fill_speed = zip(
            *[
                (sol.alpha, sol.speed)
                for sol in stable_search
                if abs(sol.gamma - gamma) < tol
            ]
        )
        ax_speed_alpha.fill_between(
            fill_alpha, fill_speed, [100] * len(fill_alpha), color="gray"
        )
        fill_alpha, fill_speed2 = zip(
            *[
                (sol.alpha, sol.speed)
                for sol in unstable_search
                if abs(sol.gamma - gamma) < tol
            ]
        )
        ax_speed_alpha.fill_between(
            fill_alpha, [-100] * len(fill_alpha), fill_speed2, color="gray"
        )
        ax_speed_alpha.text(10.5, 0.15, "Fronts", color="white")
    ax_speed_alpha.plot(
        *zip(
            *[
                (sol.alpha, sol.speed)
                for sol in stable_search
                if abs(sol.gamma - gamma) < tol
            ]
        ),
        linestyle="-",
        color=color
    )
    ax_speed_alpha.plot(
        *zip(
            *[
                (sol.alpha, sol.speed)
                for sol in unstable_search
                if abs(sol.gamma - gamma) < tol
            ]
        ),
        linestyle=":",
        color=color
    )
ax_speed_alpha.set_xlim(min(sol.alpha for sol in stable_search), 20)
ax_speed_alpha.text(-0.1, 0.5, "$c$", rotation=90, transform=ax_speed_alpha.transAxes)

ax_speed_alpha.set_ylim(0.1, 1.2)
ax_speed_alpha.set_yticks([0, 1.2], [0, 1.2])

# plt.colorbar(
#     matplotlib.cm.ScalarMappable(norm=gamma_color_norm, cmap=cmap),
#     label="$\\gamma$",
#     ax=ax_width_alpha,
#     orientation='horizontal',
#     ticks=[0.1, 0.2],
# )

ax_speed_alpha.text(
    subplot_label_x,
    subplot_label_y,
    "B",
    fontdict=subplot_label_font,
    transform=ax_speed_alpha.transAxes,
)

###################################################
# speed by gamma
###################################################

stable_search.solutions.sort(key=lambda sol: sol.gamma)
unstable_search.solutions.sort(key=lambda sol: sol.gamma)
ax_speed_gamma.plot([], [], linestyle="-", color=style_legend_color, label="Stable")
ax_speed_gamma.plot([], [], linestyle=":", color=style_legend_color, label="Unstable")
for alpha, color in zip(plot_alphas, cycle(alpha_colors)):
    ax_speed_gamma.plot(
        *zip(
            *[
                (sol.gamma, sol.speed)
                for sol in stable_search
                if abs(sol.alpha - alpha) < tol
            ]
        ),
        linestyle="-",
        color=color
    )
    ax_speed_gamma.plot(
        *zip(
            *[
                (sol.gamma, sol.speed)
                for sol in unstable_search
                if abs(sol.alpha - alpha) < tol
            ]
        ),
        linestyle=":",
        color=color
    )
ax_speed_gamma.fill_between([0.2, 0.4], [-10] * 2, [10] * 2, color="gray")
ax_speed_gamma.text(0.205, 0.4, "Fronts", color="white", rotation=90)
# plt.colorbar(
#     matplotlib.cm.ScalarMappable(norm=alpha_color_norm, cmap=cmap),
#     label="$\\tau_q$",
#     ax=ax_speed_gamma,
# )
ax_speed_gamma.text(
    subplot_label_x,
    subplot_label_y,
    "C",
    fontdict=subplot_label_font,
    transform=ax_speed_gamma.transAxes,
)

###################################################
# width by alpha
###################################################
stable_search.solutions.sort(key=lambda sol: sol.alpha)
unstable_search.solutions.sort(key=lambda sol: sol.alpha)
ax_width_alpha.plot([], [], linestyle="-", color=style_legend_color, label="Stable")
ax_width_alpha.plot([], [], linestyle=":", color=style_legend_color, label="Unstable")
for gamma, color in zip(plot_gammas, cycle(gamma_colors)):
    if gamma == 0.2:
        fill_alpha, fill_speed = zip(
            *[
                (sol.alpha, sol.width)
                for sol in stable_search
                if abs(sol.gamma - gamma) < tol
            ]
        )
        ax_width_alpha.fill_between(
            fill_alpha, fill_speed, [100] * len(fill_alpha), color="gray"
        )
        fill_alpha, fill_speed2 = zip(
            *[
                (sol.alpha, sol.width)
                for sol in unstable_search
                if abs(sol.gamma - gamma) < tol
            ]
        )
        ax_width_alpha.fill_between(
            fill_alpha, [-100] * len(fill_alpha), fill_speed2, color="gray"
        )
    ax_width_alpha.plot(
        *zip(
            *[
                (sol.alpha, sol.width)
                for sol in stable_search
                if abs(sol.gamma - gamma) < tol
            ]
        ),
        linestyle="-",
        color=color
    )
    ax_width_alpha.plot(
        *zip(
            *[
                (sol.alpha, sol.width)
                for sol in unstable_search
                if abs(sol.gamma - gamma) < tol
            ]
        ),
        linestyle=":",
        color=color
    )

ax_width_alpha.text(11, 12, "Fronts", color="white")
# ax_width_alpha.set_xlabel(r"$\tau_q$")
# ax_width_alpha.set_ylabel("width")

ax_width_alpha.text(0.5, -0.12, r"$\tau_q$", transform=ax_width_alpha.transAxes)
ax_width_alpha.set_xticks([11, 20], [11, 20])

ax_width_alpha.set_ylim(0, 14)
ax_width_alpha.set_yticks([0, 14], [0, 14])
ax_width_alpha.text(
    -0.15, 0.5, r"$\Delta$", rotation=90, transform=ax_width_alpha.transAxes
)
ax_width_alpha.set_xticks([11, 19], [11, 19])
# plt.colorbar(
#     matplotlib.cm.ScalarMappable(norm=gamma_color_norm, cmap=cmap),
#     ax=ax_width_alpha,
#     label="$\\gamma$",
# )

ax_width_alpha.text(
    subplot_label_x,
    subplot_label_y,
    "E",
    fontdict=subplot_label_font,
    transform=ax_width_alpha.transAxes,
)

###################################################
# width by gamma
###################################################
stable_search.solutions.sort(key=lambda sol: sol.gamma)
unstable_search.solutions.sort(key=lambda sol: sol.gamma)
ax_width_gamma.plot([], [], linestyle="-", color=style_legend_color, label="Stable")
ax_width_gamma.plot([], [], linestyle=":", color=style_legend_color, label="Unstable")
for alpha, color in zip(plot_alphas, cycle(alpha_colors)):
    ax_width_gamma.plot(
        *zip(
            *[
                (sol.gamma, sol.width)
                for sol in stable_search
                if abs(sol.alpha - alpha) < tol
            ]
        ),
        linestyle="-",
        color=color
    )
    ax_width_gamma.plot(
        *zip(
            *[
                (sol.gamma, sol.width)
                for sol in unstable_search
                if abs(sol.alpha - alpha) < tol
            ]
        ),
        linestyle=":",
        color=color
    )
ax_width_gamma.fill_between([0.2, 0.4], [-100] * 2, [100] * 2, color="gray")
ax_width_gamma.text(0.205, 5, "Fronts", color="white", rotation=90)
ax_width_gamma.set_xlim(0.12, 0.22)
ax_width_gamma.set_xticks([0.13, 0.2], [0.13, 0.2])
ax_width_gamma.text(0.5, -0.12, r"$\gamma$", transform=ax_width_gamma.transAxes)
# plt.colorbar(
#     matplotlib.cm.ScalarMappable(norm=alpha_color_norm, cmap=cmap),
#     ax=ax_width_gamma,
#     label="$\\tau_q$",
# )

ax_width_gamma.text(
    subplot_label_x,
    subplot_label_y,
    "F",
    fontdict=subplot_label_font,
    transform=ax_width_gamma.transAxes,
)

# colorbars
plt.colorbar(
    matplotlib.cm.ScalarMappable(norm=alpha_color_norm, cmap=cmap_alpha),
    cax=ax_alpha_colorbar,
    orientation="horizontal",
)
ax_alpha_colorbar.set_xticks([11, 20], [11, 20])
ax_alpha_colorbar.text(0.5, -0.7, r"$\tau_q$", transform=ax_alpha_colorbar.transAxes)

plt.colorbar(
    matplotlib.cm.ScalarMappable(norm=gamma_color_norm, cmap=cmap_gamma),
    cax=ax_gamma_colorbar,
    orientation="horizontal",
)
ax_gamma_colorbar.set_xticks([.10, .2], [.10, .2])
ax_gamma_colorbar.text(0.5, -0.7, r"$\gamma$", transform=ax_gamma_colorbar.transAxes)

# grid.tight_layout(fig, w_pad=0.5, h_pad=-0.5)

plt.savefig(HI_RES_IMAGE_FILE, dpi=300)
for ext in [".pdf", ".png"]:
    plt.savefig(IMAGE_FILE + ext)
