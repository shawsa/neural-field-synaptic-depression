"""Numerics for pulses.

Specifically, we assume a Heaviside firing-rate function to make some
analytical simplifications, and can then use numerical techniques to
approximate the traveling pulse solutions and null-spaces given a weight kernel
and parameters.

Given an estimate for the pulse width (Delta) and pulse speed (c), we assume
that the fore-threshold crossing occurs at coordinate x = 0. We then use a
variation of the shooting method to determine the pulse width and speed, and
in general the profiles of the activity (U) and synaptic efficacy (Q) variables
for the traveling pulse solution.

The speed and width are determined via successive binary search using
specified intervals for each. For a given value of pulse width and a
pair of values for the speed, we propagate the activity variable forward.
We expect U -> +/- oo for each of the speeds, while the true value would give
U -> 0. We use this to perform a binary search on the pulse speed.

Given this method for finding a consistent speed for a given pulse width,
we propagate the activity variable backwards and expect U(-Delta) = theta.
We determine the true value of Delta using a binary search that enforces this
condition.

The null-space calculations have not yet been implemented, but preliminary
work suggests this can be expressed analytically as the function of two
quadratures dependent on the pulse profile.
"""

import numpy as np

from functools import partial
from numpy.polynomial.polynomial import Polynomial
from scipy.interpolate import BarycentricInterpolator
from scipy.optimize import root
from scipy.signal import fftconvolve


def Q_mid(x, alpha, gamma, c, **_):
    """The analytic formula for the synaptic efficacy in the
    active region.
    """
    return gamma + (1-gamma)*np.exp(x/(c*alpha*gamma))


def Q_left(x, alpha, gamma, c, Delta, **_):
    """The analytic formula for the synaptic efficacy to the
    left of the active region.
    """
    QmDelta = Q_mid(-Delta, alpha, gamma, c)
    return 1 + (QmDelta-1)*np.exp((x+Delta)/(c*alpha))


""" Note that to the right of the active region, the synaptic
efficacy has a constant value of 1.
"""


def Q_profile(xs: np.ndarray, alpha, gamma, c, Delta, **_):
    """The profile of the synaptic efficacy variable."""
    Qs = np.ones_like(xs)
    left_mask = xs < -Delta
    mid_mask = np.logical_and(xs < 0, ~left_mask)
    Qs[left_mask] = Q_left(xs[left_mask], alpha, gamma, c, Delta)
    Qs[mid_mask] = Q_mid(xs[mid_mask], alpha, gamma, c)
    return Qs

class Domain:
    """A helper class to represent the spatial domain of the
    neural feild model. It's primary purpose is to avoid
    code duplication for the several shooting-method-like 
    functions, and to compute quadratures.
    """
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
    """The spatial derivative of the activity variable for the traveling pulse
    solution. We use the name `forcing` because we interpret the profile as the
    solution to the ODE with this as a forcing function.
    """
    return (U-zs.quad(weight_kernel(x-zs.array)*Q_mid(zs.array, alpha, gamma, c)))/c/mu

def shoot_forward(xs: Domain, U0, forcing):
    """Use forward Euler to solve an IVP on the given domain."""
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
    if format_str is not None:
        print(format_str % (a, b))
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
    """Use binary search to find the wave speed consistent with this pulse
    width. Note, due to the exponential growth of the activity variable,
    we do not expect the activity to approach zero, rather we try to force
    it to be close to zero for as long as possible and only accept this part
    of the pulse to be an accurate approximation.
    """
    format_str = 'c in (%f, %f)' if verbose else None
    def func(c):
        return U_shoot_forward(xs_right, alpha, gamma, mu, c, Delta, theta, weight_kernel)[-1]
    return bin_search(c_min, c_max, func, format_str=format_str, tol=tol)

def shoot_backward(xs: Domain, U0, forcing):
    """Use forward Euler to solve a final value problem on the given domain.
    The naming can be a bit confusing here. Given an *ending point* rather than
    a starting point, we can do a change of variables -t = tau, and perform
    forward Euler on tau -> tau_0 = -t_0. This is not to be confused with
    backward Euler which solve the same problem as forward Euler, but with
    different stability criteria.
    """
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

def get_stencil(x0, xs: np.ndarray, width=5):
    """Find the <width> closest points in the array xs to the point x0."""
    index = np.argmin(np.abs(x0 - xs))
    min_index = max(index-width//2, 0)
    return slice(min_index, min_index+width)

def local_interp(z, xs, ys):
    """Use local polynomial interpolation to approximate y(z)."""
    stencil = get_stencil(z, xs)
    poly = BarycentricInterpolator(xs[stencil], ys[stencil])
    return float(poly(z))

def local_diff(z, xs, ys, width=5):
    """Use local polynomial interpolation to approximate y'(z)."""
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
    """Use a binary search to find the pulse width. We enforce
    U(-Delta) = theta.

    Note, each evaluation of Delta performs a binary search
    to find the consistent wave speed c.
    """
    format_str = 'Delta in (%f, %f)' if verbose else None

    def func(Delta):
        """The function to root-find."""
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

def pulse_profile(xs_left: Domain,
                  xs_right: Domain,
                  *, c, Delta, alpha, gamma, mu, theta, weight_kernel,
                  vanish_tol=1e-5, **_):
    """Approximate the profile of the activity (U) and synaptic
    efficacy (Q) variables on the domain. This should only be used
    after correct values of the pulse width (Delta) and pulse speed
    (c) are known (to sufficient accuracy).

    The numerical profile is used for plotting, approximating derivatives,
    and approximating quadratures.
    """

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
    if vanish_tol is not None:
        vanish_index = max(i for i in range(len(Us_right)) if abs(Us_right[i]) < vanish_tol)
        Us_right[vanish_index:] = 0
    Us = np.hstack((Us_left[:-1], Us_right))
    Qs = Q_profile(xs, alpha, gamma, c, Delta)
    return xs, Us, Qs

def nullspace_amplitudes(
        xs, Us, Qs,
        c, Delta,
        mu, beta,
        weight_kernel,
        **_):
    """Find the values A0 (A_0) and AmD (A_{-Delta}) that denote the
    aplitude of the two components of the first null-space function (v1).
    WOLOG, A0 = 1. AmD is computed semi-analytically, using a formula that
    requires two numerical quadratures.
    """
    dUmD = local_diff(0, xs, Us)
    QmD = local_interp(-Delta, xs, Qs)

    g0_domain = Domain(0, 4*Delta, 4001)
    def g0(z):
        return g0_domain.quad(weight_kernel(z-g0_domain.array) *
                              np.exp(-g0_domain.array/c/mu))

    gmD_domain = Domain(-Delta, 4*Delta, 4001)
    def gmD(z):
        return gmD_domain.quad(weight_kernel(z-gmD_domain.array) *
                               np.exp(-gmD_domain.array/c/mu))

    A0 = 1
    AmD = g0(-Delta) / (c*mu*abs(dUmD)/QmD*np.exp(Delta/c/mu) - gmD(-Delta))
    return A0, AmD

def v1(xs, *, A0, AmD, mu, c, Delta, **_):
    """The first function of the adjoint null-space pair,
    up to two unknown constants A0, and AmD (A_{-Delta}).
    """
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
    """Create a function to approximate the weight kernel convolved
    with v1 (see above). This is done efficiently using fft to
    convolve array samples of each function along the domain, then
    local interpolation to evaluate arbitrary locations.
    """
    v1_arr = v1(zs, A0=A0, AmD=AmD, mu=mu, c=c, Delta=Delta)
    spacing = (zs[-1]-zs[0])/(len(zs) - 1)
    ys = fftconvolve(spacing*weight_kernel(zs), v1_arr, mode='same')
    my_slice = slice(10, -10)
    return lambda z: local_interp(z, zs[my_slice], ys[my_slice])

# To satisfy BCs we require v2_left = 0. We can shoot forward from
# v2(-Delta) = 0 to v2(0), and then use the analytic solution for
# v2_right.


def v2_mid(xs: Domain, *, mu, alpha, beta,
           c, Delta,
           weight_kernel,
           A0, AmD, **_
           ):
    vmD = 0
    zs = np.linspace(-4*Delta, 4*Delta, 4001)
    wv1 = make_wv1(zs=zs, A0=A0, AmD=AmD,
                   mu=mu, c=c, Delta=Delta, weight_kernel=weight_kernel)
    forcing = lambda x, v: 1/(c*alpha) * (-(1+beta)*v + wv1(x))
    return shoot_forward(xs, vmD, forcing)

def v2_right(xs: np.ndarray, *, v0, c, alpha):
    return v0*np.exp(-xs/(c*alpha))

def v2(xs: np.ndarray, *, mu, alpha, beta,
       c, Delta,
       weight_kernel,
       A0, AmD, **_
       ):
    ys = np.zeros_like(xs)
    inner_slice = slice(np.argmax(xs >= -Delta), np.argmin(xs <= 0))
    zs = Domain(xs[inner_slice][0], xs[inner_slice][-1],
                len(xs[inner_slice]))
    ys[inner_slice] = v2_mid(zs, mu=mu, alpha=alpha, beta=beta,
                             c=c, Delta=Delta,
                             weight_kernel=weight_kernel,
                             A0=A0, AmD=AmD)
    ys[xs >= 0] = v2_right(xs[xs >= 0], v0=ys[inner_slice][-1],
                           c=c, alpha=alpha)
    return ys
