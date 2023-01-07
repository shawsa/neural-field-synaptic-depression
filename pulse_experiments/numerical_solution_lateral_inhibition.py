# Numerical solution to traveling wave soltuion.

import matplotlib.pyplot as plt 
import numpy as np

from scipy.interpolate import BarycentricInterpolator
from tqdm import tqdm

params = {
    'theta': 0.2,
    'alpha': 20,
    'beta': 5.0,
    'mu': 1.0
}

params['gamma'] = 1/(1+params['beta'])

def weight_kernel(x):
    abs_x = np.abs(x)
    return (1-.5*abs_x)*np.exp(-abs_x)

def Q_mid(x, alpha, gamma, c, **params):
    return gamma + (1-gamma)*np.exp(x/(c*alpha*gamma))

def Q_left(x, alpha, gamma, c, Delta, **params):
    QmDelta = Q_mid(-Delta, alpha, gamma, c)
    return 1 + (QmDelta-1)*np.exp((x+Delta)/(c*alpha))

def shoot_forward(Delta, c):
    N = 4001
    xs = np.linspace(-200, 200, N)
    h = (xs[-1] - xs[0])/(N-1)
    zs = np.linspace(-Delta, 0, 201)
    trap_weights = np.ones_like(zs)
    trap_weights[0] = .5
    trap_weights[-1] = .5
    trap_spacing = Delta/(len(zs)-1)

    quad = lambda ys: (ys @ trap_weights)*trap_spacing
    forcing = lambda x, U: (U-quad(weight_kernel(x-zs)*Q_mid(zs, c=c, **params)))/c/params['mu']

    xs_right = xs[N//2:]
    U_right = [params['theta']]

    for x in xs_right[:-1]:
        U_right.append(U_right[-1] + h*forcing(x, U_right[-1]))

    return xs_right, U_right

def improve_c(c_min, c_max, Delta):
    c_mid = (c_min + c_max)/2
    _, U_right = shoot_forward(Delta, c_mid)
    if U_right[-1] > 0:
        return (c_min, c_mid)
    return (c_mid, c_max)

def find_c(Delta, c_interval=(.1, 10), tol=1e-5):
    c_min, c_max = c_interval
    _, U_right = shoot_forward(Delta, c_min)
    assert U_right[-1] < 0
    _, U_right = shoot_forward(Delta, c_max)
    assert U_right[-1] > 0
    while c_max - c_min > tol:
        c_min, c_max = improve_c(c_min, c_max, Delta)
    return (c_min + c_max)/2

def shoot_backward(Delta, c):
    N = 4001
    xs = np.linspace(-200, 200, N)
    h = (xs[-1] - xs[0])/(N-1)
    zs = np.linspace(-Delta, 0, 201)
    trap_weights = np.ones_like(zs)
    trap_weights[0] = .5
    trap_weights[-1] = .5
    trap_spacing = Delta/(len(zs)-1)

    quad = lambda ys: (ys @ trap_weights)*trap_spacing
    forcing = lambda x, U: (U-quad(weight_kernel(x-zs)*Q_mid(zs, c=c, **params)))/c/params['mu']
    xs_left = xs[: N//2+1]
    U_left = [params['theta']]

    for x in xs_left[::-1][:-1]:
        U_left.append(U_left[-1] - h*forcing(x, U_left[-1]))

    return xs_left, U_left[::-1]

def generate_profile(Delta, c):
    xs_right, U_right = shoot_forward(Delta, c)
    xs_left, U_left = shoot_backward(Delta, c)

    xs = np.zeros(len(xs_right) + len(xs_left) - 1)
    Us = np.zeros_like(xs)
    xs[:len(xs_left)] = xs_left
    Us[:len(xs_left)] = U_left
    xs[len(xs_left):] = xs_right[1:]
    U_index = max([index for index in range(len(U_right))
                   if abs(U_right[index]) < 1e-5])
    Us[len(xs_left):len(xs_left)+U_index-1] = U_right[1: U_index]

    Qs = np.ones_like(xs)
    Qs[xs < 0] = Q_mid(xs, c=c, **params)[xs < 0]
    Qs[xs < -Delta] = Q_left(xs, c=c, Delta=Delta, **params)[xs < -Delta]

    return xs, Us, Qs


def get_stencil(x0, xs, width=5):
    index = np.argmin(np.abs(x0 - xs))
    return slice(index-width//2, index+(width+1)//2)

def find_delta(Delta_interval=(5, 40), c_interval=(0.1, 10)):

    def func(Delta):
        c = find_c(Delta, c_interval=c_interval)
        xs_right, U_right = shoot_forward(Delta, c)
        xs_left, U_left = shoot_backward(Delta, c)
        stencil = get_stencil(-Delta, xs_left)
        poly = BarycentricInterpolator(xs_left[stencil], U_left[stencil])
        return poly(-Delta) - params['theta']

    def improve_interval(Delta_min, Delta_max):
        f_upper = func(Delta_max)
        f_lower = func(Delta_min)
        assert f_lower*f_upper < 0
        Delta_mid = (Delta_min + Delta_max)/2
        f_new = func(Delta_mid)
        if f_new*f_upper > 0:
            return Delta_min, Delta_mid
        return Delta_mid, Delta_max

    Delta_min, Delta_max = Delta_interval
    while Delta_max - Delta_min > 1e-2:
        Delta_min, Delta_max = improve_interval(Delta_min, Delta_max)
    return (Delta_min + Delta_max)/2


Delta_interval = (5, 10)
c_interval = c_interval=(.1, 4)
Delta = find_delta(Delta_interval, c_interval)
c = find_c(Delta, c_interval=c_interval, tol=1e-10)

xs, Us, Qs = generate_profile(Delta, c)
plt.plot(xs, Us, 'b-')
plt.plot(xs, Qs, 'b--')
plt.ylim(-.1, 1.1)
plt.plot([0, -Delta], [params['theta']]*2, 'k.')
plt.show()

if False:
    # testing
    c = 1 
    Delta = 10
    c = find_c(Delta, c_interval=(.5, 1), tol=1e-10)
    print(f'c={c}')
    xs_right, Us_right = shoot_forward(Delta, c)
    plt.plot(xs_right, Us_right)
    plt.ylim(-.1, 1.1)
    plt.xlim(-20, 20)
    xs_left, Us_left = shoot_backward(Delta, c)
    plt.plot(xs_left, Us_left)
    plt.plot([0, -Delta], [params['theta']]*2, 'k.')
