#!/usr/bin/python3
'''
'''

import experiment_defaults

from adaptive_front import get_speed
from neural_field import Parameters

import matplotlib.pyplot as plt
import numpy as np
import os.path
import pickle


def epsilon_u_q_slope(*, mu, alpha, beta, theta, c, **_):
    """The asymptotic approximation gives a response that is
    linear with the magnitude of the delta input to either u
    or q. The slope of the theory lines will be the ratio
    of these values. This function computes that ratio.
    """
    return (1+c*alpha)/2/(1+beta+c*alpha)/(1+c*mu) / (c*mu)


data_file_name = os.path.join(experiment_defaults.data_path,
                              'delta_time_amp_grid.pickle')
image_file_name = os.path.join(experiment_defaults.media_path,
                               'delta_time_amp_grid')

# params = Parameters(mu=1.0, alpha=7.0, gamma=0.2)
# params = Parameters(mu=1.0, alpha=20.0, gamma=0.2)
# theta = 0.1

with open(data_file_name, 'rb') as f:
    params, theta, eps_u, eps_q, res = pickle.load(f)

speed = get_speed(theta=theta, **params.dict)

plt.figure(figsize=(7, 5))

z_max = np.floor(np.max(np.nan_to_num(res, nan=-np.inf)))
z_min = np.ceil(np.min(np.nan_to_num(res, nan=np.inf)))
z_max, z_min = max(z_max, -z_min), min(-z_max, z_min)

plt.pcolormesh(eps_u, eps_q, res, vmin=z_min, vmax=z_max,
               cmap='seismic', shading='gouraud')
plt.colorbar(label='$\\nu_\\infty$')
contour_set = plt.contour(eps_u, eps_q, res, levels=range(-4, 15, 2), colors='k')

slope = -1/epsilon_u_q_slope(c=speed, theta=theta, **params.dict)

plt.plot([-.08, .08], [slope*-.08, slope*.08], 'k:')

plt.clabel(contour_set, fmt='%.1f')
plt.xlabel('$\\epsilon_u$')
plt.ylabel('$\\epsilon_q$')
plt.xlim(np.min(eps_u), np.max(eps_u))
plt.ylim(np.min(eps_q), np.max(eps_q))
plt.title('Wave response to spatially homogeneous delta.')

for ext in ['.eps', '.png']:
    plt.savefig(image_file_name + ext)

plt.show()
