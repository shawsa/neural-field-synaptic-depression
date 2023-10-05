#!/usr/bin/python3
"""
A driver for the neural field simulator. Consider this a manual test of
most of the functionality.
"""

import experiment_defaults

import matplotlib.pyplot as plt
import numpy as np
import os.path
import pickle

from functools import partial
from itertools import product
from more_itertools import windowed
from scipy.stats import linregress
from tqdm import tqdm

from neural_field_synaptic_depression.neural_field import (
    NeuralField,
    ParametersBeta,
    heaviside_firing_rate,
    exponential_weight_kernel,
)
from neural_field_synaptic_depression.root_finding_helpers import find_roots
from neural_field_synaptic_depression.space_domain import (
    SpaceDomain,
    BufferedSpaceDomain,
)
from neural_field_synaptic_depression.time_domain import (
    TimeDomain,
    TimeDomain_Start_Stop_MaxSpacing,
)
from neural_field_synaptic_depression.time_integrator import Euler, EulerDelta
from neural_field_synaptic_depression.time_integrator_tqdm import TqdmWrapper

from plotting_helpers.plotting_helpers import make_animation

from num_assist import (
    Domain,
    find_delta,
    find_c,
    pulse_profile,
    nullspace_amplitudes,
    v1,
    v2,
    local_interp,
)

FILE_NAME = os.path.join(experiment_defaults.data_path, "entrainment.pickle")

FIG_FILE_NAME = os.path.join(experiment_defaults.media_path, "entrainment_contour")

with open(FILE_NAME, "rb") as f:
    stim_magnitudes, stim_speeds, results, params, params_dict = pickle.load(f)


# plt.xlim(np.min(stim_magnitudes), np.max(stim_magnitudes))
# plt.ylim(np.min(stim_speeds), np.max(stim_speeds))

mag_mat, speed_mat = np.meshgrid(stim_magnitudes, stim_speeds)

res_mat = np.zeros_like(mag_mat, dtype=bool)
for index, (mag, speed) in enumerate(zip(mag_mat.flat, speed_mat.flat)):
    for sol in results:
        if sol["stim_magnitude"] == mag and sol["stim_speed"] == speed:
            np.ravel(res_mat)[index] = sol["entrained"]
            break

plt.figure(figsize=(5, 3))
contour_set = plt.contour(mag_mat, speed_mat, res_mat, [0.5], colors=["black"])
plt.plot(mag_mat.flat, speed_mat.flat, ".", color="lightgray")
plt.text(0.25, 2, "Entrainment", fontsize="x-large")
plt.text(0.05, 4.0, "Non-Entrainment", fontsize="x-large")
plt.xlabel("Stimulus Magnitude")
plt.ylabel("Stimulus Speed")
plt.title("Entrainment to a moving Gaussian")
plt.tight_layout()
plt.show()
for ext in [".png", ".eps"]:
    plt.savefig(FIG_FILE_NAME + ext)
