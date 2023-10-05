#!/usr/bin/python3
"""Spatially locallized stimulation of traveling pulses."""

import experiment_defaults

import matplotlib.pyplot as plt
import numpy as np
import os.path
import pickle

from functools import partial
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

from helper_symbolics import get_speed_and_width
from num_assist import (
    Domain,
    find_delta,
    find_c,
    pulse_profile,
    nullspace_amplitudes,
    v1,
    v2,
    local_interp,
    local_diff,
)

from neural_field_synaptic_depression.neural_field import (
    NeuralField,
    ParametersBeta,
    heaviside_firing_rate,
    exponential_weight_kernel,
)
from neural_field_synaptic_depression.space_domain import BufferedSpaceDomain
from neural_field_synaptic_depression.time_domain import (
    TimeDomain_Start_Stop_MaxSpacing,
)
from neural_field_synaptic_depression.time_integrator import Euler
from neural_field_synaptic_depression.time_integrator_tqdm import TqdmWrapper

from neural_field_synaptic_depression.root_finding_helpers import find_roots

plt.rcParams.update(
    {
        "font.size": 12,
        "text.usetex": True,
    }
)

FIG_FILE = os.path.join(experiment_defaults.media_path, "figure4")

DATA_FILE_NAME = "spatially_localized.pickle"
with open(os.path.join(experiment_defaults.data_path, DATA_FILE_NAME), "rb") as f:
    params, params_dict, stim_width, stim_amp, locations, responses = pickle.load(f)

# pulse solution
xs_left = Domain(-100, 0, 10**4)
xs_right = Domain(0, 200, 10**4)
USE_SAVED_VALUES = True
if USE_SAVED_VALUES:
    c, Delta = 1.0457801654119976, 9.497941970825195
    print(f"c={c}\nDelta={Delta}")
else:
    Delta_interval = (7, 10)
    speed_interval = (1, 2)
    Delta = find_delta(
        *Delta_interval, *speed_interval, xs_left, xs_right, verbose=True, **params_dict
    )
    c = find_c(*speed_interval, xs_right, Delta=Delta, verbose=True, **params_dict)

xs, Us, Qs = pulse_profile(
    xs_right=xs_right, xs_left=xs_left, c=c, Delta=Delta, **params_dict, vanish_tol=1e-4
)
space = Domain(xs[0], xs[-1], len(xs))
Us = [local_interp(z, xs, Us) for z in space.array]
Qs = [local_interp(z, xs, Qs) for z in space.array]
dU = np.array([local_diff(z, space.array, Us, width=2) for z in space.array])
dQ = np.array([local_diff(z, space.array, Qs, width=2) for z in space.array])
A0, AmD = nullspace_amplitudes(space.array, Us, Qs, c, Delta, **params_dict)
v1_arr = v1(space.array, A0=A0, AmD=AmD, c=c, Delta=Delta, **params.dict)
v2_arr = v2(space.array, A0=A0, AmD=AmD, c=c, Delta=Delta, **params_dict)

denom = space.quad(params.mu * dU * v1_arr + params.alpha * dQ * v2_arr)
theory_loc = np.linspace(locations[0], locations[-1], 2001)
theory = []
for loc in theory_loc:
    pulse_profile = stim_amp * (
        np.heaviside(space.array - loc + stim_width / 2, 0)
        - np.heaviside(space.array - loc - stim_width / 2, 0)
    )
    num = space.quad(pulse_profile * v1_arr)
    theory.append(-num / denom)

figsize = (7, 3)
grid = GridSpec(1, 2)
fig = plt.figure(figsize=figsize)

ax_activity = fig.add_subplot(grid[0, 0])

ax_activity.plot(space.array, v1_arr * 0.5, "b-", label="$v$")
ax_activity.plot(locations, responses, "go", label="Simulation")
ax_activity.plot(theory_loc, theory, "k-", label="Theory")
ax_activity.set_xlim(locations[0], 6)
ax_activity.set_xlabel("$x_0$")
ax_activity.set_ylabel("$\\nu_\\infty$")
ax_activity.legend()
ax_activity.set_title("Square pulse: $\\varepsilon I_u$")

ax_synapse = fig.add_subplot(grid[0, 1])
DATA_FILE_NAME = "spatially_localized_q.pickle"
with open(os.path.join(experiment_defaults.data_path, DATA_FILE_NAME), "rb") as f:
    params, params_dict, stim_width, stim_amp, locations, responses = pickle.load(f)
theory = []
for loc in theory_loc:
    pulse_profile = stim_amp * (
        np.heaviside(space.array - loc + stim_width / 2, 0)
        - np.heaviside(space.array - loc - stim_width / 2, 0)
    )
    num = space.quad(pulse_profile * v2_arr)
    theory.append(-num / denom)

ax_synapse.plot(space.array, v2_arr * max(theory) / max(v2_arr), "b-", label="$p$")
ax_synapse.plot(locations, responses, "go", label="Simulation")
ax_synapse.plot(theory_loc, theory, "k-", label="Theory")
ax_synapse.set_xlim(-8, locations[-1])
ax_synapse.set_xlabel("$x_0$")
ax_synapse.set_ylabel("$\\nu_\\infty$")
ax_synapse.legend()
ax_synapse.set_title("Square pulse: $\\varepsilon I_q$")

# panel labels
subplot_label_font = {
    "size": "x-large",
    "weight": "bold",
}
for ax, label in [(ax_activity, "A"), (ax_synapse, "B")]:
    ax.text(
        -0.15,
        1.04,
        label,
        fontdict=subplot_label_font,
        transform=ax.transAxes,
    )

plt.tight_layout()
for ext in [".pdf", ".png"]:
    plt.savefig(os.path.join(experiment_defaults.media_path, FIG_FILE + ext))