#!/usr/bin/python3
"""
A driver for the neural field simulator. Consider this a manual test of
most of the functionality.
"""

import experiment_defaults

import matplotlib.pyplot as plt
import os.path
import pickle
from adaptive_front import response

plt.rcParams.update({"text.usetex": True})

DATA_FILE_NAME = os.path.join(
    experiment_defaults.data_path, "spatially_homogeneous_limit.pickle"
)
FIG_FILE_NAME = os.path.join(
    experiment_defaults.media_path, "fig_spatially_homogeneous_limit"
)

with open(DATA_FILE_NAME, "rb") as f:
    params, theta, epsilons, responses = pickle.load(f)

response_slope = response(1, **params.dict, theta=theta)

plt.figure(figsize=(5, 3))
plt.plot(epsilons, epsilons * response_slope, "k-", label="Theory")
plt.plot(epsilons, responses, "go", label="Simulation")

plt.xlabel("$\\epsilon$")
plt.ylabel("$\\nu_\\infty$")
plt.title("Front response to global stimulus")
plt.legend(loc="upper left")
plt.tight_layout()

for extension in [".png", ".pdf"]:
    plt.savefig(FIG_FILE_NAME + extension)
