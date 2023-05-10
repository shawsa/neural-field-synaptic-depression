#!/usr/bin/python3
'''
A driver for the neural field simulator. Consider this a manual test of
most of the functionality.
'''

import experiment_defaults

import matplotlib.pyplot as plt
import numpy as np
import os.path
import pickle

from functools import partial
from itertools import product
from neural_field import NeuralField, ParametersBeta, heaviside_firing_rate, exponential_weight_kernel
from num_assist import Domain, find_delta, find_c, pulse_profile, nullspace_amplitudes, v1, v2, local_interp
from plotting_helpers import make_animation
from root_finding_helpers import find_roots
from space_domain import SpaceDomain, BufferedSpaceDomain
from time_domain import TimeDomain, TimeDomain_Start_Stop_MaxSpacing
from time_integrator import Euler, EulerDelta
from time_integrator_tqdm import TqdmWrapper
from tqdm import tqdm

from scipy.stats import linregress
from more_itertools import windowed

FILE_NAME = os.path.join(experiment_defaults.data_path,
                         'entrainment_square.pickle')

FIG_FILE_NAME = os.path.join(experiment_defaults.media_path,
                             'entrainment_square_contour')

with open(FILE_NAME, 'rb') as f:
    stim_magnitudes, stim_speeds, results, stim_width, slope, params, params_dict = pickle.load(f)


# plt.xlim(np.min(stim_magnitudes), np.max(stim_magnitudes))
# plt.ylim(np.min(stim_speeds), np.max(stim_speeds))

mag_mat, speed_mat = np.meshgrid(stim_magnitudes, stim_speeds)

res_mat = np.zeros_like(mag_mat, dtype=bool)
for index, (mag, speed) in enumerate(zip(mag_mat.flat, speed_mat.flat)):
    for sol in results:
        if sol['stim_magnitude'] == mag and sol['stim_speed'] == speed:
            np.ravel(res_mat)[index] = sol['entrained']
            break

plt.figure(figsize=(5, 4))
# contour_set = plt.contour(mag_mat, speed_mat, res_mat, [0.5], colors=['green'])
plt.plot([mag for mag, entrained in zip(mag_mat.flat, res_mat.flat) if entrained==True],
         [speed for speed, entrained in zip(speed_mat.flat, res_mat.flat) if entrained==True],
         '+', color='green', label='Entrained')
plt.plot([mag for mag, entrained in zip(mag_mat.flat, res_mat.flat) if entrained==False],
         [speed for speed, entrained in zip(speed_mat.flat, res_mat.flat) if entrained==False],
         'x', color='magenta', label='Failed to Entrain')
plt.plot(stim_magnitudes, params_dict['c'] + stim_magnitudes*slope, 'k-', label='Theory')
plt.text(.25, 2, 'Entrainment', fontsize='x-large')
plt.text(0.05, 4.0, 'Non-Entrainment', fontsize='x-large')
plt.xlabel('Stimulus Magnitude')
plt.ylabel('Stimulus Speed')
plt.title('Entrainment to a moving square stimulus')
plt.legend()
plt.tight_layout()
plt.show()
for ext in ['.png', '.eps']:
    plt.savefig(FIG_FILE_NAME + ext)
