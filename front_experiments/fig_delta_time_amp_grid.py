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


def main():

    data_file_name = os.path.join(experiment_defaults.data_path,
                                  'delta_time_amp_grid.pickle')
    image_file_name = os.path.join(experiment_defaults.media_path,
                                   'delta_time_amp_grid')

    params = Parameters(mu=1.0, alpha=20.0, gamma=0.2)
    theta = 0.1
    speed = get_speed(mu=params.mu,
                      alpha=params.alpha,
                      gamma=params.gamma,
                      theta=theta)

    with open(data_file_name, 'rb') as f:
        eps_u, eps_q, res = pickle.load(f)

    plt.figure(figsize=(7, 5))

    z_max = np.floor(np.max(np.nan_to_num(res, nan=-np.inf)))
    z_min = np.ceil(np.min(np.nan_to_num(res, nan=np.inf)))
    z_max, z_min = max(z_max, -z_min), min(-z_max, z_min)

    plt.pcolormesh(eps_u, eps_q, res, vmin=z_min, vmax=z_max,
                   cmap='seismic', shading='gouraud')
    plt.colorbar(label='$\\nu_\\infty$')
    contour_set = plt.contour(eps_u, eps_q, res, levels=range(-4, 15, 2), colors='k')

    slope = 1/(2*(1+speed*params.mu)*(1+params.beta+speed*params.alpha))

    plt.plot([-.08, .08], [slope*-.08, slope*.08], 'k:')

    plt.clabel(contour_set, fmt='%.1f')
    plt.xlabel('$\\epsilon_u$')
    plt.ylabel('$\\epsilon_q$')
    plt.title('Wave response to spatially homogeneous delta.')

    for ext in ['.eps', '.png']:
        plt.savefig(image_file_name + ext)

    plt.show()


if __name__ == '__main__':
    main()
