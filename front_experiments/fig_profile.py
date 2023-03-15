#!/usr/bin/python3
'''
The wave response to the inpulse I(x,t) = epsilon * delta(t - 1)
simulated and compared to asymptotic approximation.
'''

import experiment_defaults

import matplotlib.pyplot as plt
import numpy as np
import os.path

from adaptive_front import U_numeric, Q_numeric, v1, v2, get_speed
from neural_field import Parameters
from plotting_styles import U_style, Q_style, solution_styles, threshold_style


PROFILE_FILE = 'front_profile'
NULLSPACE_FILE = 'nullspace'

params = Parameters(mu=1.0, alpha=20.0, gamma=0.2)
theta = 0.1

c = get_speed(theta=theta, **params.dict)

xs = np.linspace(-50, 20, 201)
custom = {
    'linewidth': 2.0,
}

plt.figure(figsize=(5, 3))
plt.plot(xs,
         U_numeric(xs, theta=theta, **params.dict),
         **U_style,
         **solution_styles[0],
         **custom,
         label='$U$')
plt.plot(xs,
         Q_numeric(xs, theta=theta, **params.dict),
         **Q_style,
         **solution_styles[0],
         **custom,
         label='$Q$')
plt.plot(xs,
         theta+0*xs,
         **threshold_style,
         **custom,
         label='$\\theta$')
plt.plot([xs[0], 0], [0, 0], color='gray', linewidth=6.0, label='Active Region')
plt.legend()
plt.title('Traveling Front Profile')
plt.xlabel('$\\xi$')
plt.xlim(xs[0], xs[-1])
plt.tight_layout()

for extension in ['.png', '.eps']:
    plt.savefig(os.path.join(experiment_defaults.media_path,
                             PROFILE_FILE + extension))

xs = np.linspace(-10, 60, 201)
plt.figure(figsize=(5, 3))
plt.plot(xs, v1(xs=xs, theta=theta, c=c, **params.dict), 'b-', label='$v_1$')
plt.plot(xs, v2(xs=xs, theta=theta, c=c, **params.dict), 'b--', label='$v_2$')
plt.ylim(-.01, 0.05)
plt.xlabel('$\\xi$')
plt.xlim(xs[0], xs[-1])
plt.title('Front Nullspace')
plt.legend()
plt.tight_layout()
for extension in ['.png', '.eps']:
    plt.savefig(os.path.join(experiment_defaults.media_path,
                             NULLSPACE_FILE + extension))


