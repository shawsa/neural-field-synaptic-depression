'''
A class that solves the neural field equation  with synaptic depression, using
method of lines.
'''

from dataclasses import dataclass
from math import ceil
import numpy as np
from scipy.signal import fftconvolve
from .space_domain import SpaceDomain
from typing import Callable


class Parameters:

    def __init__(self, *, mu: float, alpha: float, gamma: float):
        self.mu = mu
        self.alpha = alpha
        self.gamma = gamma
        self.beta = 1/gamma - 1

    @property
    def dict(self):
        return {
            'mu': self.mu,
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma}

    def __repr__(self):
        return 'Params(' + str(self.dict) + ')'


class ParametersBeta(Parameters):

    def __init__(self, *, mu: float, alpha: float, beta: float):
        self.mu = mu
        self.alpha = alpha
        self.beta = beta

        self.gamma = 1/(1 + beta)


class NeuralField:

    def __init__(self,
                 space: SpaceDomain,
                 firing_rate: Callable[[np.ndarray], np.ndarray],
                 weight_kernel: Callable[[np.ndarray], np.ndarray],
                 params: Parameters):

        self.space = space
        self.firing_rate = firing_rate
        self.weight_kernel = weight_kernel
        self.params = params

        self.initialize_convolution()

    def initialize_convolution(self):
        h = self.space.spacing
        kernel_half_width = ceil(16/h*np.log(10))
        kernel_xs = h*np.arange(-kernel_half_width, kernel_half_width+1)
        self.kernel = h*self.weight_kernel(kernel_xs)

    def conv(self, y: np.ndarray):
        return fftconvolve(y, self.kernel, mode='same')

    def rhs(self, t, v):
        u, q = v
        temp = self.firing_rate(u)
        v_new = np.empty_like(v)
        v_new[0] = 1/self.params.mu * (-u + self.conv(q*temp))
        v_new[1] = (1 - q - self.params.beta*q*temp)/self.params.alpha
        return v_new


class NeuralFieldMatrixConvolv(NeuralField):

    def initialize_convolution(self):
        # A matrix of quatrature weights
        # Assuming the values at the boundary are zero, this is
        # essentially a trapezoidal rule.
        self.M = self.space.spacing*self.weight_kernel(np.subtract.outer(
                     self.space.array,
                     self.space.array))

    def conv(self, y: np.ndarray):
        return self.M@y

# sample firing rate functions

def heaviside_firing_rate(y: np.ndarray, theta: float) -> np.ndarray:
    return np.heaviside(y-theta, .5)

# weight kernel functions

def exponential_weight_kernel(y: np.ndarray) -> np.ndarray:
    return .5*np.exp(-np.abs(y))

def wizzard_hat(y: np.ndarray) -> np.ndarray:
    yabs = np.abs(y)
    return (1 - yabs)*np.exp(-yabs)
