#!/usr/bin/python3
"""
The wave response to the inpulse I(x,t) = epsilon * delta(t - 1)
simulated and compared to asymptotic approximation.
"""

import experiment_defaults

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import os.path

from neural_field_synaptic_depression.neural_field import Parameters
from plotting_helpers.plotting_styles import (
    Q_style,
    U_style,
    solution_styles,
    threshold_style,
)
from front_profile_helpers import (
    Q_progressive,
    Q_regressive,
    U_progressvie,
    U_regressive,
    get_speed_lower,
    get_speed_regressive,
    get_speed_upper,
)


def get_min_gamma(*, mu, alpha, theta):
    A = (2 * theta * mu) ** 2
    B = -4 * theta * mu * alpha * (1 + 2 * theta)
    C = 8 * theta * mu * alpha + (2 * theta - 1) ** 2 * alpha**2
    return [2 * A / (-B + sign * np.sqrt(B**2 - 4 * A * C)) for sign in [1, -1]]


def get_speed(*, mu, alpha, theta, gamma, desc=None):
    A = 2 * theta * mu * alpha
    B = (2 * theta - 1) * alpha + 2 * theta * mu / gamma
    C = 2 * theta / gamma - 1
    if desc is None:
        desc = B**2 - 4 * A * C
    return (-B + np.sqrt(desc)) / (2 * A)


def get_speed2(*, mu, alpha, theta, gamma, desc=None):
    A = 2 * theta * mu * alpha
    B = (2 * theta - 1) * alpha + 2 * theta * mu / gamma
    C = 2 * theta / gamma - 1
    if desc is None:
        desc = B**2 - 4 * A * C
    return (-B - np.sqrt(desc)) / (2 * A)


def regressive_speed(*, mu, theta, gamma):
    return (1 - theta / (gamma - theta)) / 2 / mu


# fmt: off
# yuck
def alpha_crit(gamma, mu, theta):
    return 2*mu*theta*(-2*gamma + 2*theta + 2*np.sqrt(gamma**2 - 2*gamma*theta - gamma + 2*theta) + 1)/(gamma*(4*theta**2 - 4*theta + 1))
# fmt: on


FILE_NAME = "figure1"

plt.rcParams.update(
    {
        "font.size": 12,
        "text.usetex": True,
    }
)


subplot_label_x = -0.05
subplot_label_y = 1.05
subplot_label_font = {
    "size": "x-large",
    "weight": "bold",
}


figsize = (7, 4)
grid = GridSpec(2, 12)
fig = plt.figure(figsize=figsize)
ax_gamma = fig.add_subplot(grid[0, :6])
ax_alpha = fig.add_subplot(grid[0, 6:], sharey=ax_gamma)
ax_stable = fig.add_subplot(grid[1, :4])
ax_unstable = fig.add_subplot(grid[1, 4:8], sharey=ax_stable, sharex=ax_stable)
ax_regressive = fig.add_subplot(grid[1, 8:], sharey=ax_stable, sharex=ax_stable)
for ax in [ax_alpha, ax_unstable, ax_regressive]:
    plt.setp(ax.get_yticklabels(), visible=False)

for ax, label in [
    (ax_gamma, "A"),
    (ax_alpha, "B"),
    (ax_stable, "C"),
    (ax_unstable, "D"),
    (ax_regressive, "E"),
]:
    ax.text(
        subplot_label_x,
        subplot_label_y,
        label,
        fontdict=subplot_label_font,
        transform=ax.transAxes,
    )


###################################################
# gamma plot
###################################################

mu = 1.0
alphas = [1.1, 2, 5, 10]
colors = ["b", "g", "m", "c"]
theta = 0.1

gammas = np.linspace(0, 1, 20001)
mins_curve_gammas = []
mins_curve_speeds = []
for alpha in np.logspace(-1, 1.2, 20001):
    gamma_mins = get_min_gamma(mu=mu, alpha=alpha, theta=theta)
    for gamma in gamma_mins:
        speed = get_speed(alpha=alpha, mu=mu, theta=theta, gamma=gamma)
        mins_curve_gammas.append(gamma)
        mins_curve_speeds.append(speed)

mins_curve_gammas = np.array(mins_curve_gammas)
mins_curve_speeds = np.array(mins_curve_speeds)

nan_mask = np.logical_and(
    np.logical_not(np.isnan(mins_curve_gammas)),
    np.logical_not(np.isnan(mins_curve_speeds)),
)

mins_curve_speeds[mins_curve_speeds < 0] = np.nan

# plt.plot(mins_curve_gammas, mins_curve_speeds, 'k.', markersize=1)
ax_gamma.plot(
    mins_curve_gammas[nan_mask], mins_curve_speeds[nan_mask], "k-"
)  # , label='bifurcation')

for alpha, color in zip(alphas, colors):
    gamma_mins = get_min_gamma(mu=mu, alpha=alpha, theta=theta)
    speeds = get_speed(alpha=alpha, mu=mu, theta=theta, gamma=gammas)
    speeds[speeds < 0] = np.nan
    speeds2 = get_speed2(alpha=alpha, mu=mu, theta=theta, gamma=gammas)
    speeds2[speeds2 < 0] = np.nan
    ax_gamma.plot(gammas, speeds, "-", color=color, label=f"$\\tau_q={alpha}$")
    ax_gamma.plot(gammas, speeds2, ":", color=color)
    for gamma in gamma_mins:
        desc = (
            (2 * theta - 1) * alpha + 2 * theta * mu / gamma
        ) ** 2 - 8 * theta * mu * alpha * (2 * theta / gamma - 1)
        assert np.all(np.abs(desc) < 1e-10)
        speed = get_speed(alpha=alpha, mu=mu, theta=theta, gamma=gamma, desc=0)

regressive_gammas = np.linspace(theta, 2 * theta, 201)
ax_gamma.plot(
    regressive_gammas,
    regressive_speed(mu=mu, theta=theta, gamma=regressive_gammas),
    "r-",
)

# plt.plot([theta]*2, [-5, 5], 'k:', label='$\\theta$')
ax_gamma.plot([2 * theta] * 2, [-5, 5], "k:")  # , label='$\\gamma = 2\\theta$')
ax_gamma.text(0.21, -1.5, r"$\gamma = 2\theta$")
ax_gamma.fill_between(
    [-1, theta], 2 * [5], 2 * [-5], color="grey"
)  # , label='$\\gamma < \\theta$')
ax_gamma.text(0.05, -1.9, "Pulses", color="white", rotation=90)
ax_gamma.plot(2 * theta, 0, "k*", markersize=10)

ax_gamma.legend(loc="lower right", framealpha=1.0)
# ax_gamma.set_xlabel("$\\gamma$")
# ax_gamma.set_ylabel("speed ($c$)")
ax_gamma.text(0.5, -0.13, r"$\gamma$", transform=ax_gamma.transAxes)
ax_gamma.text(-0.08, 0.7, "$c$", transform=ax_gamma.transAxes, rotation=90)

ax_gamma.set_xlim(0.04, 1)
ax_gamma.set_ylim(-2.1, 4.1)
ax_gamma.set_xticks([0, 1], [0, 1])
ax_gamma.set_yticks([0, 4], [0, 4])

###################################################
# alpha plot
###################################################

regressive_linspace = np.linspace(0.1, 0.2, 5)
gammas = list(regressive_linspace) + [0.3, 0.5, 1]
color_norm = matplotlib.colors.Normalize(vmin=0.1, vmax=0.2)
cmap = matplotlib.cm.get_cmap("winter")

colors = [cmap(color_norm(gamma)) for gamma in regressive_linspace] + [
    "g",
    "gold",
    "red",
]

gamma_linspace = np.linspace(2 * theta, theta, 201)
alpha_crit_arr = alpha_crit(gamma_linspace, mu, theta)
speed_crit_arr = get_speed(
    mu=mu, alpha=alpha_crit_arr, theta=theta, gamma=gamma_linspace, desc=0
)


alphas = np.linspace(0, 20, 2001)

for gamma, color in zip(gammas, colors):
    speeds = get_speed(alpha=alphas, mu=mu, theta=theta, gamma=gamma)
    speeds[speeds < 0] = np.nan
    speeds2 = get_speed2(alpha=alphas, mu=mu, theta=theta, gamma=gamma)
    speeds2[speeds2 < 0] = np.nan
    if gamma == theta:
        mask = np.logical_not(np.isnan(speeds))
        ax_alpha.fill_between(
            alphas[mask], speeds2[mask], speeds[mask], color="grey"
        )  # ,label='$\\gamma < \\theta$')
    if gamma == theta or gamma >= 2 * theta:
        ax_alpha.plot(alphas, speeds, "-", color=color, label=f"$\\gamma={gamma}$")
        ax_alpha.plot(alphas, speeds2, ":", color=color)
    else:
        ax_alpha.plot(alphas, speeds, "-", color=color)
        ax_alpha.plot(alphas, speeds2, ":", color=color)
    if gamma != theta and gamma < 2 * theta:
        ax_alpha.plot(
            alphas,
            len(alphas) * [regressive_speed(mu=mu, theta=theta, gamma=gamma)],
            "--",
            color=color,
        )

ax_alpha.plot(alpha_crit_arr[0], 0, "k*")
ax_alpha.plot(alpha_crit_arr, speed_crit_arr, "k-")  # , label='bifurcation')
ax_alpha.text(5.5, 0.8, "Pulses", color="w")
# ax_alpha.set_xlabel(r"$\tau_q$")
ax_alpha.set_xticks([0, 20], [0, 20])
ax_alpha.text(0.5, -0.13, r"$\tau_q$", transform=ax_alpha.transAxes)
ax_alpha.set_xlim(0, 20)
ax_alpha.legend(loc="lower right", framealpha=1.0)

###################################################
# profiles - params
###################################################

xs = np.linspace(-50, 50, 201)
params = Parameters(mu=1.0, alpha=20.0, gamma=0.15)

###################################################
# stable profile
###################################################
c_hi = get_speed_upper(
    mu=params.mu, alpha=params.alpha, theta=theta, gamma=params.gamma
)
ax_stable.plot([], [], "k-", label="$U$")
ax_stable.plot([], [], "k--", label="$Q$")
ax_stable.title("")

ax_stable.plot(
    xs,
    U_progressvie(
        xs, mu=params.mu, alpha=params.alpha, gamma=params.gamma, theta=theta, c=c_hi
    ),
    "b-",
    label=f"$c={c_hi:.2f}$",
)
ax_stable.plot(
    xs,
    Q_progressive(
        xs, mu=params.mu, alpha=params.alpha, gamma=params.gamma, theta=theta, c=c_hi
    ),
    "b--",
)
ax_stable.text(xs[0] + 5, 0.9, f"c={c_hi:.2f}")
ax_stable.text(xs[0] + 5, 0.7, "Stable")
ax_stable.set_xticks([0], [r"$\xi = 0$"])
ax_stable.set_xlim(xs[0], xs[-1])
ax_stable.text(20, 0.1, r"$U(\xi)$")
ax_stable.text(20, 0.8, r"$Q(\xi)$")

###################################################
# unstable profile
###################################################
c_lo = get_speed_lower(
    mu=params.mu, alpha=params.alpha, theta=theta, gamma=params.gamma
)

ax_unstable.plot(
    xs,
    np.nan_to_num(
        U_progressvie(
            xs,
            mu=params.mu,
            alpha=params.alpha,
            gamma=params.gamma,
            theta=theta,
            c=c_lo,
        ),
        nan=0.0,
    ),
    "g-",
    label=f"$c={c_lo:.2f}$",
)
ax_unstable.plot(
    xs,
    Q_progressive(
        xs, mu=params.mu, alpha=params.alpha, gamma=params.gamma, theta=theta, c=c_lo
    ),
    "g--",
)
ax_unstable.text(xs[0] + 5, 0.7, "Unstable")
ax_unstable.text(xs[0] + 5, 0.9, f"c={c_lo:.2f}")
ax_unstable.text(20, 0.1, r"$U(\xi)$")
ax_unstable.text(20, 0.8, r"$Q(\xi)$")

###################################################
# regresive profile
###################################################
c_neg = get_speed_regressive(
    mu=params.mu, alpha=params.alpha, theta=theta, gamma=params.gamma
)

ax_regressive.plot(
    xs,
    U_regressive(
        xs, mu=params.mu, alpha=params.alpha, gamma=params.gamma, theta=theta, c=c_neg
    ),
    "r-",
    label=f"$c={c_neg:.2f}$",
)
ax_regressive.plot(
    xs,
    Q_regressive(
        xs, mu=params.mu, alpha=params.alpha, gamma=params.gamma, theta=theta, c=c_neg
    ),
    "r--",
)
ax_regressive.text(xs[0] + 5, 0.7, "Stable")
ax_regressive.text(xs[0] + 5, 0.9, f"c={c_neg:.2f}")
ax_regressive.text(20, 0.1, r"$U(\xi)$")
ax_regressive.text(20, 0.75, r"$Q(\xi)$")

###################################################
# grid
###################################################

grid.tight_layout(fig, w_pad=0, h_pad=0.05)

for ext in [".png", ".pdf"]:
    plt.savefig(os.path.join(experiment_defaults.media_path, FILE_NAME + ext))
