#!/usr/bin/python3
"""Spatially locallized stimulation of traveling pulses."""

import experiment_defaults

from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import os.path
import pickle
from tqdm import tqdm
from helper_symbolics import get_speed_and_width
from num_assist import Domain, find_delta, find_c, pulse_profile, nullspace_amplitudes, v1, v2, local_interp, local_diff
from neural_field import NeuralField, ParametersBeta, heaviside_firing_rate, exponential_weight_kernel
from space_domain import BufferedSpaceDomain
from time_domain import TimeDomain_Start_Stop_MaxSpacing
from time_integrator import Euler
from time_integrator_tqdm import TqdmWrapper

from root_finding_helpers import find_roots


DATA_FILE_NAME = 'spatially_localized.pickle'
FIG_FILE_NAME = 'spatially_localized'
with open(os.path.join(experiment_defaults.data_path, DATA_FILE_NAME), 'rb') as f:
    params, params_dict, stim_width, stim_amp, locations, responses = pickle.load(f)

# pulse solution
xs_left = Domain(-100, 0, 10**4)
xs_right = Domain(0, 200, 10**4)
USE_SAVED_VALUES = True
if USE_SAVED_VALUES:
    c, Delta = 1.0457801654119976, 9.497941970825195
    print(f'c={c}\nDelta={Delta}')
else:
    Delta_interval = (7, 10)
    speed_interval = (1, 2)
    Delta = find_delta(*Delta_interval, *speed_interval,
                       xs_left, xs_right, verbose=True,
                       **params_dict)
    c = find_c(*speed_interval,  xs_right,
               Delta=Delta, verbose=True,
               **params_dict)

xs, Us, Qs = pulse_profile(xs_right=xs_right, xs_left=xs_left,
                           c=c, Delta=Delta, **params_dict, vanish_tol=1e-4)
space = Domain(xs[0], xs[-1], len(xs))
Us = [local_interp(z, xs, Us) for z in space.array]
Qs = [local_interp(z, xs, Qs) for z in space.array]
dU = np.array([local_diff(z, space.array, Us, width=2) for z in space.array])
dQ = np.array([local_diff(z, space.array, Qs, width=2) for z in space.array])
A0, AmD = nullspace_amplitudes(space.array, Us, Qs, c, Delta, **params_dict)
v1_arr = v1(space.array, A0=A0, AmD=AmD, c=c, Delta=Delta, **params.dict)
v2_arr = v2(space.array, A0=A0, AmD=AmD, c=c, Delta=Delta, **params_dict)

denom = space.quad(params.mu*dU*v1_arr + params.alpha*dQ*v2_arr)
theory_loc = np.linspace(locations[0], locations[-1], 2001)
theory = []
for loc in theory_loc:
    pulse_profile = stim_amp * (np.heaviside(space.array-loc+stim_width/2, 0) -
                                    np.heaviside(space.array-loc-stim_width/2, 0))
    num = space.quad(pulse_profile*v1_arr)
    theory.append(-num/denom)

plt.figure(figsize=(5,3))
plt.plot(space.array, v1_arr*0.5, 'b-', label='$v_1$')
plt.plot(locations, responses, 'go', label='Simulation')
plt.plot(theory_loc, theory, 'k-', label='Theory')
plt.xlim(locations[0], locations[-1])
plt.xlabel('$x_0$')
plt.ylabel('$\\nu_\\infty$')
plt.legend(loc='upper right')
plt.title('Pulse response to spatially localized stimulus (u)')
plt.tight_layout()
for ext in ['.eps', '.png']:
    plt.savefig(os.path.join(experiment_defaults.media_path, FIG_FILE_NAME+ext))
