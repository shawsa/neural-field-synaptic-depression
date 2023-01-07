import numpy as np

from functools import partial
from numpy.polynomial.polynomial import Polynomial
from scipy.interpolate import BarycentricInterpolator
from scipy.optimize import root
from scipy.signal import fftconvolve

def Q_mid(x, alpha, gamma, c, **_):
    return gamma + (1-gamma)*np.exp(x/(c*alpha*gamma))

def Q_left(x, alpha, gamma, c, Delta, **_):
    QmDelta = Q_mid(-Delta, alpha, gamma, c)
    return 1 + (QmDelta-1)*np.exp((x+Delta)/(c*alpha))

def Q_profile(xs: np.ndarray, alpha, gamma, c, Delta, **_):
    Qs = np.ones_like(xs)
    left_mask = xs < -Delta
    mid_mask = np.logical_and(xs < 0, ~left_mask)
    Qs[left_mask] = Q_left(xs[left_mask], alpha, gamma, c, Delta)
    Qs[mid_mask] = Q_mid(xs[mid_mask], alpha, gamma, c)
    return Qs

class Domain:
    def __init__(self, x0: float, xf: float, n: int):
        assert x0 < xf
        assert n > 1
        self.x0 = x0
        self.xf = xf
        self.n = n
        self.h = (xf - x0)/(n-1)
        self.array = np.linspace(x0, xf, n)

    def quad(self, ys: np.ndarray) -> float:
        assert ys.shape == self.array.shape
        return (np.sum(ys[1:-1]) + .5 * (ys[0] + ys[-1]))*self.h

    def __iter__(self):
        yield from self.array

    def __len__(self):
        return len(self.array)

def forcing_U(x, U, zs: Domain, weight_kernel, c, Delta, alpha, gamma, mu, **_):
    return (U-zs.quad(weight_kernel(x-zs.array)*Q_mid(zs.array, alpha, gamma, c)))/c/mu

def shoot_forward(xs: Domain, U0, forcing):
    us = [U0]
    for x in xs.array[:-1]:
        us.append(us[-1] + xs.h*forcing(x, us[-1]))

    return np.array(us)

def U_shoot_forward(xs_right: Domain, alpha, gamma, mu, c, Delta, theta, weight_kernel, **_):
    zs = Domain(-Delta, 0, 201)
    my_forcing = partial(forcing_U,
                         alpha=alpha,
                         zs=zs,
                         weight_kernel=weight_kernel,
                         gamma=gamma,
                         Delta=Delta,
                         c=c,
                         mu=mu)
    return shoot_forward(xs_right, theta, my_forcing)

def bin_search(a, b, func, tol=1e-10, format_str=None):
    assert a < b
    func_a = func(a)
    func_b = func(b)
    assert func_a*func_b <= 0
    done = False
    while not done:
        c = (a+b)/2
        func_c = func(c)
        if func_a*func_c < 0:
            b, func_b = c, func_c
        else:
            a, func_a = c, func_c
        if format_str is not None:
            print(format_str % (a, b))
        if b - a < tol:
            done = True
    return c

def find_c(c_min, c_max, xs_right: Domain, alpha, gamma, mu, Delta, theta, weight_kernel, verbose=False, tol=1e-10, **_):
    format_str = 'c in (%f, %f)' if verbose else None
    def func(c):
        return U_shoot_forward(xs_right, alpha, gamma, mu, c, Delta, theta, weight_kernel)[-1]
    return bin_search(c_min, c_max, func, format_str=format_str, tol=tol)

def shoot_backward(xs: Domain, U0, forcing):
    us = [U0]
    for x in xs.array[::-1][:-1]:
        us.append(us[-1] - xs.h*forcing(x, us[-1]))

    return np.array(us[::-1])

def U_shoot_backward(xs_left: Domain, alpha, gamma, mu, c, Delta, theta, weight_kernel, **_):
    zs = Domain(-Delta, 0, 201)
    my_forcing = partial(forcing_U,
                         alpha=alpha,
                         zs=zs,
                         weight_kernel=weight_kernel,
                         gamma=gamma,
                         Delta=Delta,
                         c=c,
                         mu=mu)
    return shoot_backward(xs_left, theta, my_forcing)

def get_stencil(x0, xs, width=5):
    index = np.argmin(np.abs(x0 - xs))
    min_index = max(index-width//2, 0)
    return slice(min_index, min_index+width)

def local_interp(z, xs, ys):
    stencil = get_stencil(z, xs)
    poly = BarycentricInterpolator(xs[stencil], ys[stencil])
    return float(poly(z))

def local_diff(z, xs, ys, width=5):
    stencil = get_stencil(z, xs, width=width)
    poly = Polynomial.fit(xs[stencil], ys[stencil], deg=width-1).deriv()
    return float(poly(z))

def find_delta(Delta_min, Delta_max,
               c_min, c_max,
               xs_left: Domain,
               xs_right: Domain,
               mu,
               alpha,
               gamma,
               theta,
               weight_kernel,
               verbose=False,
               tol=1e-5,
               **_):

    format_str = 'Delta in (%f, %f)' if verbose else None
    def func(Delta):
        c = find_c(c_min, c_max, xs_right, alpha, gamma, mu, Delta, theta, weight_kernel)
        U_left = U_shoot_backward(xs_left,
                                  alpha=alpha,
                                  gamma=gamma,
                                  mu=mu,
                                  c=c,
                                  Delta=Delta,
                                  theta=theta,
                                  weight_kernel=weight_kernel)
        return local_interp(-Delta, xs_left.array, U_left) - theta

    return bin_search(Delta_min, Delta_max, func, format_str=format_str, tol=tol)

def pulse_profile(xs_right: Domain,
                  xs_left: Domain,
                  *, c, Delta, alpha, gamma, mu, theta, weight_kernel, **_):

    xs = np.hstack((xs_left.array[:-1], xs_right.array))
    Us_right = U_shoot_forward(xs_right, c=c, Delta=Delta,
                               alpha=alpha,
                               gamma=gamma,
                               mu=mu,
                               theta=theta,
                               weight_kernel=weight_kernel)
    Us_left = U_shoot_backward(xs_left, c=c, Delta=Delta,
                               alpha=alpha,
                               gamma=gamma,
                               mu=mu,
                               theta=theta,
                               weight_kernel=weight_kernel)
    vanish_index = max(i for i in range(len(Us_right)) if abs(Us_right[i]) < 1e-5)
    Us_right[vanish_index:] = 0
    Us = np.hstack((Us_left[:-1], Us_right))
    Qs = Q_profile(xs, alpha, gamma, c, Delta)
    return xs, Us, Qs


def v1(xs, A0, AmD, *, mu, c, Delta, **_):
    return np.exp(-xs/(c*mu))*(
            AmD*np.heaviside(xs+Delta, 0) +
            A0*np.heaviside(xs, 0))

def make_wv1(*, 
             zs: np.ndarray,
             A0,
             AmD,
             mu,
             c,
             Delta,
             weight_kernel,
             **_):
    v1_arr = v1(zs, A0, AmD, mu=mu, c=c, Delta=Delta)
    ys = fftconvolve(weight_kernel(zs), v1_arr, mode='same')
    my_slice = slice(10, -10)
    return lambda z: local_interp(z, zs[my_slice], ys[my_slice])

def v2_mid(zs: Domain, *, wv1, c, alpha, Delta, A0, AmD, beta, v0=1, **_):
    def my_forcing(x, v):
        return -1/(c*alpha)*(v - wv1(x)-beta*v)
    return shoot_backward(zs, v0, my_forcing)

def v2_shoot_forward(xs_right: Domain, *,
                     c, alpha, v0=1, **_):
    def my_forcing(x, v):
        return -1/c/alpha*v
    return shoot_forward(xs_right, v0, my_forcing)

def v2_shoot_backward(xs_left: Domain, *,
                      c, alpha, beta, Delta,
                      wv1, v0=1, **_):
    def my_forcing(x, v):
        if x <= -Delta:
            return -1/c/alpha*v
        return -1/c/alpha*((1+beta)*v - wv1(x))
    return shoot_backward(xs_left, v0, my_forcing)
    
# fix me
# def find_nullspace_roots(
#         A0, AmD, *,
#         dU0, Q0, dUmD, QmD,
#         c, Delta,
#         mu, alpha, beta,
#         weight_kernel,
#         **_):
# 
#     xs = np.linspace(-4*Delta, 4*Delta, 2001)
#     conv0_vals = fftconvolve(weight_kernel(xs),
#                              np.heaviside(xs, 0)*np.exp(-(1/c/mu)*xs),
#                              mode='same')
#     conv0 = lambda z: local_interp(z, xs, conv0_vals)
#     convmD_vals = fftconvolve(weight_kernel(xs),
#                               np.heaviside(xs+Delta, 0)*np.exp(-(1/c/mu)*xs),
#                               mode='same')
#     convmD = lambda z: local_interp(z, xs, convmD_vals)
#     A0_conv0_coeff = Q0/abs(dU0)*conv0(0)-1
#     A0_convmD_coeff = Q0/abs(dU0)*convmD(0)
#     AmD_conv0_coeff = QmD/abs(dUmD)*conv0(0)
#     AmD_convmD_coeff = QmD/abs(dUmD)*convmD(0)-1
# 
#     zs = Domain(-Delta, 0, 2001)
# 
#     v2 = v2_mid(zs=zs, wv1=wv1, A0=0, AmD=AmD,
#                 c=c, alpha=alpha, Delta=Delta, beta=beta)
# 
#     A0_root = -A0 + 1/c/mu * (Q0/abs(dU0)*wv1(0) - alpha*beta*v2[-1])
#     AmD_root = -AmD + 1/c/mu * (QmD/abs(dUmD)*wv1(-Delta) - alpha*beta*v2[0])
# 
#     return A0_root, AmD_root

def nullspace_amplitudes(
        dU0, Q0, dUmD, QmD,
        c, Delta,
        mu, alpha, beta,
        weight_kernel,
        **_):
    
    g0_domain = Domain(0, 4*Delta, 2001)
    def g0(z):
        return g0_domain.quad(weight_kernel(z-g0_domain.array) * \
                              np.exp(-g0_domain.array/c/mu))

    gmD_domain = Domain(-Delta, 4*Delta, 2001)
    def gmD(z):
        return gmD_domain.quad(weight_kernel(z-gmD_domain.array) * \
                               np.exp(-gmD_domain.array/c/mu))

    A0 = 1
    AmD = g0(-Delta) / (c*mu * abs(dUmD)/QmD * np.exp(Delta/c/mu) - gmD(-Delta))
    return A0, AmD


if __name__ == '__main__':
    import matplotlib.pyplot as plt
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

    xs_right = Domain(0, 200, 8001)
    xs_left = Domain(-200, 0, 8001)

    Delta = find_delta(7, 20, 1, 10, xs_left, xs_right, verbose=True, **params)
    c = find_c(1, 10, xs_right, Delta=Delta, verbose=True, **params)
    # Us_right = U_shoot_forward(xs_right, c=c, Delta=Delta, **params)
    # Us_left = U_shoot_backward(xs_left, c=c, Delta=Delta, **params)
    # plt.plot(xs_right.array, Us_right)
    # plt.plot(xs_left.array, Us_left)
    # plt.plot([-Delta, 0], [params['theta']]*2, 'ko')
    # plt.ylim(-.1, 1.1)

    params_full = {
        'c': c,
        'Delta': Delta,
        **params
    }

    xs, Us, Qs = pulse_profile(xs_right, xs_left, **params_full)
    plt.figure('Traveling wave.')
    plt.plot(xs, Us, 'b-')
    plt.plot(xs, Qs, 'b--')
    plt.plot([-Delta, 0], [params['theta']]*2, 'k.')
    plt.show()

    # nullspace
    nullspace_params = {
            **params_full,
            'dU0': local_diff(0, xs, Us),
            'dUmD' : local_diff(-Delta, xs, Us),
            'Q0':  local_interp(0, xs, Qs),
            'QmD': local_interp(-Delta, xs, Qs)
    }

    A0, AmD = nullspace_amplitudes(**nullspace_params)
    print(A0, AmD)

    plt.figure('Nullspace')
    plt.plot(xs, v1(xs, A0, AmD, **params_full))
    plt.show()
