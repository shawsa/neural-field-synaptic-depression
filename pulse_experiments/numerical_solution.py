# Numerical solution to traveling wave soltuion.

import matplotlib.pyplot as plt 
import numpy as np

from num_assist import *

def weight_kernel(x):
    return .5*np.exp(-np.abs(x))

params = {
    'theta': 0.2,
    'alpha': 20.0,
    'beta': 5.0,
    'mu': 1.0
}

params['gamma'] = 1/(1+params['beta'])
params['weight_kernel'] = weight_kernel

xs_right = Domain(0, 200, 2001)
xs_left = Domain(-200, 0, 2001)

Delta = find_delta(10, 50, 1, 10, xs_left, xs_right, verbose=True, **params)
c = find_c(1, 10, xs_right, Delta=Delta, verbose=True, **params)
Us_right = U_shoot_forward(xs_right, c=c, Delta=Delta, **params)
Us_left = U_shoot_backward(xs_left, c=c, Delta=Delta, **params)
plt.plot(xs_right.array, Us_right)
plt.plot(xs_left.array, Us_left)
plt.plot([-Delta, 0], [params['theta']]*2, 'ko')
plt.ylim(-.1, 1.1)
