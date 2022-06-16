#!/usr/bin/python3
'''
The wave response to the inpulse I(x,t) = epsilon * delta(t - 1)
simulated and compared to asymptotic approximation.
'''

import experiment_defaults

import matplotlib.pyplot as plt
import numpy as np
import os.path

from adaptive_front import U_numeric, Q_numeric
from neural_field import ParametersBeta
from plotting_styles import U_style, Q_style, solution_styles, threshold_style


def main():

    file_name = 'front_profile'

    params = ParametersBeta(mu=1.0, alpha=20.0, beta=0.2)
    theta = 0.1

    profile_xs = np.linspace(-100, 20, 201)
    plt.plot(profile_xs,
             U_numeric(profile_xs, theta=theta, **params.dict),
             **U_style,
             **solution_styles[0],
             label='$U$')
    plt.plot(profile_xs,
             Q_numeric(profile_xs, theta=theta, **params.dict),
             **Q_style,
             **solution_styles[0],
             label='$Q$')
    plt.plot(profile_xs,
             theta+0*profile_xs,
             **threshold_style,
             label='$\\theta$')
    plt.legend()
    plt.title('Traveling Pulse Profile')
    plt.xlabel('$\\xi$')
    plt.tight_layout()

    for extension in ['.png', '.eps']:
        plt.savefig(os.path.join(experiment_defaults.media_path,
                                 file_name + extension))

if __name__ == '__main__':
    main()
