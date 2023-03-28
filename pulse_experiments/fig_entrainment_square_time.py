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

import multiprocessing as mp

DATA_FILE_NAME = os.path.join(experiment_defaults.data_path,
                              'entrainment_square_time.pickle')

FIG_FILE_NAME = os.path.join(experiment_defaults.media_path,
                             'entrainment_square_time')

with open(DATA_FILE_NAME, 'rb') as f:
    stim_magnitudes, stim_speeds, stim_freqs, results, params, params_dict = pickle.load(f)

# pannels figure
freq_mat, stim_mat = np.meshgrid(stim_freqs, stim_speeds)
fig, axes = plt.subplots(1, len(stim_magnitudes), figsize=(15, 5))
if not hasattr(axes, '__len__'):
    axes = [axes]
for ax, mag in zip(axes, stim_magnitudes):
    res_mat = -np.ones_like(freq_mat, dtype=int)
    for index, (freq, speed) in enumerate(zip(freq_mat.flat, stim_mat.flat)):
        for sol in results:
            if sol['stim_magnitude'] == mag and sol['stim_speed'] == speed and sol['stim_freq'] == freq:
                np.ravel(res_mat)[index] = int(sol['entrained'])

    temp = ax.pcolormesh(freq_mat, stim_mat, res_mat,
                         cmap='seismic', shading='gouraud')
    ax.plot(freq_mat, stim_mat, '.', color='gray')
    ax.set_xlabel('Stimulus frequency')
    ax.set_ylabel('Stimulus Speed')
    ax.set_title(f'stim mag = {mag}')

plt.suptitle('Entrainment to a moving pulsing Gaussian')
# plt.colorbar(temp, ax=axes[-1])
# plt.tight_layout()
plt.show()
plt.savefig('media/entrainment_pulsing_pannels.png')

# contours figure
freq_mat, speed_mat = np.meshgrid(stim_freqs, stim_speeds)
colors = ['k', 'b', 'g', 'm', 'r']
plt.figure(figsize=(5, 3))
plt.plot(freq_mat.flat, speed_mat.flat, '.', color='lightgray')
for mag, color in zip(stim_magnitudes, colors):
    res_mat = -np.ones_like(freq_mat, dtype=int)
    for index, (freq, speed) in enumerate(zip(freq_mat.flat, speed_mat.flat)):
        for sol in results:
            if sol['stim_magnitude'] == mag and sol['stim_speed'] == speed and sol['stim_freq'] == freq:
                np.ravel(res_mat)[index] = int(sol['entrained'])

    contour_set = plt.contour(freq_mat, speed_mat, res_mat, [0.5], colors=[color])
    plt.clabel(contour_set, [0.5], fmt={0.5: f'{mag:.2f}'})

plt.text(0.7, 1.3, 'Entrainment')
plt.text(0.15, 3.3, 'Non-Entrainment')

plt.title('Entrainment to a moving pulsing Gaussian')
plt.xlabel('Stimulus Frequency')
plt.ylabel('Stimulus Speed')
plt.tight_layout()
plt.show()
for ext in ['png', 'eps']:
    plt.savefig(FIG_FILE_NAME + '.' + ext)
