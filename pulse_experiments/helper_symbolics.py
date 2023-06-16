"""
Helper file that loads symbolic expressions derived in
synaptic_depression_analysis.ipnb, computes dependent
parameters, and gives numeric functions for the traveling
pulse solution and basis functions for the adjoint-nullspace.

This helper file relies on two dictionaries of sympy objects generated by
the Jupyter Notebook synaptic_depression_analysis.ipnb, and saved to the
file below.

expr_dict has keys which are strings, and values which are sympy expressions.
sub_dict has keys which are sympy symbols found in those expressions, and
values which are also sympy expressions.

NOTE: The symbolic expressions were generated before the switch from
alpha*beta -> beta. I plan on converting this file, but in the meantime,
be sure that any calls to it make the appropriate dictionary substitution.
"""

import pickle
from functools import partial, reduce
from typing import Callable

import numpy as np
import numpy.linalg as la
import sympy as sym

from newton import NewtonRootFind

# Heaviside numerics workaround
sympy_modules = [{'Heaviside': lambda x: np.heaviside(x, 0.5)}, 'numpy']


# FILE_PATH = 'synaptic_depression_symbolics.pickle'
FILE_PATH = 'bi-exponential_symbolics.pickle'
with open(FILE_PATH, 'rb') as f:
    expr_dict = pickle.load(f)
sub_dict = expr_dict['sub_dict']


def recursive_reduce(expression):
    """
    Recursively substitute using the global sub_dict in the provided expression
    until none of the keys of sub_dict can be found in the free symbols of
    the expression.

    The intended effect is to reduce the expression by substitution until it
    only depends on the parameters: alpha, beta, theta, mu, and possibly xi.
    After this, we may substitute values for those parameters to determine
    the value of the expression, or to reduce it to something that can be
    lambdified.
    """
    reduced_expression = expression
    while any(v in reduced_expression.free_symbols for v in sub_dict.keys()):
        reduced_expression = reduced_expression.subs(sub_dict)
    return reduced_expression


def find_symbol_by_string(symbol_set, string):
    """
    Finds a sympy symbol from the symbol_set that has the specified string
    representation. 

    This is necessary since sympy symbols are not uniquely defined by their
    string representations (for example, they may include other information
    such as real=True, positive=False, etc.). In order to substitute properly
    into an expression, we must be sure we have the same symbol object from
    that expression.
    """
    for symbol in symbol_set:
        if str(symbol) in [string, '\\'+string]:
            return symbol


def free_symbols_in(coll):
    """
    Finds all of the sympy free symbols in a collection of sympy expressions,
    and returns them as a python set.
    """
    def get_symbols(expr):
        return expr.free_symbols

    return reduce(set.union, map(get_symbols, coll))


def ensure_sympy_floats(input_params, symbol_set, precision=15):
    """
    Takes a dictionary of parameters and returns a dictionary
    where the keys are sympy symbols from symbol_set and the
    values are sympy floats with the given precision.
    """
    params = {}
    for key, value in input_params.items():
        if type(key) is str:
            symbol = find_symbol_by_string(symbol_set, key)
        else:
            assert type(key) is sym.core.symbol.Symbol
            symbol = key
        params[symbol] = sym.Float(str(value), precision=precision)
    return params

def generate_Newton_args_for_speed_width(input_params):
    """Generate numerical functions for evaluating the equations that
    implicitly define the pulse width and speed, and also the Jacobian of that
    system with respect to the speed and width."""
    symbol_set = free_symbols_in([
            recursive_reduce(expr)
            for expr in expr_dict['speed_width_conditions']
    ])
    
    params = ensure_sympy_floats(input_params, symbol_set)

    c = find_symbol_by_string(symbol_set, 'c')
    Delta = find_symbol_by_string(symbol_set, 'Delta')
    if c in params.keys():
        del params[c]
    if Delta in params.keys():
        del params[Delta]
    vec = sym.Matrix([[c], [Delta]])
    F = sym.Matrix(
            [[recursive_reduce(condition).evalf(subs=params)]
             for condition in expr_dict['speed_width_conditions']])
    J = F.jacobian(vec)

    Fnum_col = sym.lambdify((c, Delta), F.T)
    Fnum = lambda c, Delta: Fnum_col(c, Delta).flatten()
    J = F.jacobian(vec)
    Jnum = sym.lambdify((c, Delta), J)
    return Fnum, Jnum


def get_speed_and_width(input_params,
                        speed_guess=1.2, width_guess=12.0,
                        max_iterations=100, cauchy_tol=1e-10,
                        verbose=False):
    """Generate the speed and width using Newtons method for the given
    parameter set."""
    F, J = generate_Newton_args_for_speed_width(input_params)
    newton = NewtonRootFind(F,
                            J,
                            max_iterations=max_iterations,
                            verbose=verbose)
    vec0 = np.array((speed_guess, width_guess))
    return tuple(newton.cauchy_tol(vec0, tol=cauchy_tol))

def get_speed_and_width_backup(input_params, precision=15,
                               iterations=100,
                               speed_guess=1.2, width_guess=12.0,
                               verbose=False):
    """An old version. The new one may be buggy, so this is a backup."""
    symbol_set = free_symbols_in([
            recursive_reduce(expr)
            for expr in expr_dict['speed_width_conditions']
    ])
    params = ensure_sympy_floats(input_params, symbol_set, precision)

    c = find_symbol_by_string(symbol_set, 'c')
    Delta = find_symbol_by_string(symbol_set, 'Delta')
    if c in params.keys():
        del params[c]
    if Delta in params.keys():
        del params[Delta]
    vec = sym.Matrix([[c], [Delta]])
    F = sym.Matrix(
            [[recursive_reduce(condition).evalf(precision, subs=params)]
             for condition in expr_dict['speed_width_conditions']])
    J = F.jacobian(vec)

    params[c] = sym.Float(speed_guess, precision)
    params[Delta] = sym.Float(width_guess, precision)
    for _ in range(iterations):
        update_vec = J.evalf(precision, subs=params).inv() * F.evalf(precision, subs=params)
        params[c] -= update_vec[0]
        params[Delta] -= update_vec[1]
        if verbose:
            print(params[c], params[Delta])
    return params

def get_numerical_parameters(input_params, precision=15, validate=True):
    if validate:
        params = get_speed_and_width_backup(input_params, precision=precision)
    else:
        params = input_params.copy()
    conditions = expr_dict['adjoint_nullspace_conditions']
    symbol_set = free_symbols_in(conditions)
    my_vars = [find_symbol_by_string(symbol_set, string)
               for string in ['A_0', 'A_{-\\Delta}', 'C1']]

    numerical_conditions = [
        recursive_reduce(condition).evalf(precision, subs=params)
        for condition in conditions
    ]

    M = sym.Matrix(
            [[condition.coeff(v) for v in my_vars]
             for condition in numerical_conditions])
    eigen_vals, eigen_vect_mat = la.eig(np.array(M, dtype=float))
    # find closest to 0
    eigen_val, eigen_vect = sorted(zip(eigen_vals, eigen_vect_mat.T),
                                   key=lambda tup: abs(tup[0]))[0]
    eigen_vect /= eigen_vect[0] # normalize according to A0 = 1
    for var, sub in zip(my_vars, eigen_vect):
        params[var] = sub

    return params

def get_traveling_pulse(input_params, precision=15, validate=True):
    if validate:
        params = get_speed_and_width_backup(input_params, precision=precision)
    else:
        params = input_params.copy()
    xi = find_symbol_by_string(expr_dict['U'].free_symbols, 'xi')
    Q = sym.lambdify(xi, recursive_reduce(expr_dict['Q']).evalf(precision, params), modules=sympy_modules)
    U = sym.lambdify(xi, recursive_reduce(expr_dict['U']).evalf(precision, params), modules=sympy_modules)
    Qp = sym.lambdify(xi, recursive_reduce(expr_dict['Q'].diff(xi)).evalf(precision, params), modules=sympy_modules)
    Up = sym.lambdify(xi, recursive_reduce(expr_dict['U'].diff(xi)).evalf(precision, params), modules=sympy_modules)

    return U, Q, Up, Qp


def get_adjoint_nullspace(input_params, precision=15, validate=True):
    if validate:
        params = get_numerical_parameters(input_params, precision=precision)
    else:
        params = input_params.copy()

    xi = find_symbol_by_string(expr_dict['v1'].free_symbols, 'xi')
    v1 = sym.lambdify(xi, recursive_reduce(expr_dict['v1']).evalf(precision, subs=params), modules=sympy_modules)
    v2 = sym.lambdify(xi, recursive_reduce(expr_dict['v2']).evalf(precision, subs=params), modules=sympy_modules)
    return v1, v2


if __name__ == '__main__':
    import os
    import matplotlib.pyplot as plt
    import experiment_defaults

    NULLSPACE_FILE_NAME = os.path.join(
            experiment_defaults.media_path,
            'bi-exponential nullspace (semi-analytical).png')

    PULSE_FILE_NAME = os.path.join(
            experiment_defaults.media_path,
            'bi-exponential pulse (semi-analytical).png')

    params = {
            r'\mu':    1.0,
            r'\alpha': 20.0,
            r'\beta':  0.25, # this was before the switch from alpha*beta |--> beta
            r'\theta':  0.2
    }
    params = get_speed_and_width_backup(params, iterations=10, verbose=False)
    params = get_numerical_parameters(params)
    for key, val in params.items():
        print(f'{str(key):>12}: {val}')

    Delta, c, mu, theta = (float(params[find_symbol_by_string(params.keys(), var)])
                    for var in ['Delta', 'c', 'mu', 'theta'])

    U_num, Q_num, Up_num, Qp_num = get_traveling_pulse(params, validate=False)
    xs = np.linspace(-40, 40, 401)
    plt.figure('Traveling wave.')
    plt.plot(xs, U_num(xs), 'b-', label='$U$')
    plt.plot(xs, Q_num(xs), 'b--', label='$Q$')
    plt.plot([-Delta, 0], [theta]*2, 'k.')
    plt.xlim(-30, 20)
    plt.legend()
    plt.title('Traveling Pulse (semi-analytical)')
    plt.savefig(PULSE_FILE_NAME)
    plt.show()

    # test nullspace amplitudes formula

    v1_num, v2_num = get_adjoint_nullspace(params, validate=False)
    xs = np.linspace(-20, 20, 401)
    plt.figure('Nullspace')
    plt.plot(xs, v1_num(xs), label='$v_1$')
    plt.plot(xs, v2_num(xs), label='$v_2$')
    plt.xlim(-15, 15)
    plt.ylim(-2e-3, 1e-2)
    plt.title('bi-exponential nullspace (semi-analytical)')
    plt.legend()
    plt.savefig(NULLSPACE_FILE_NAME)
    plt.show()


if False:
    params = {
            r'\mu':    1.0,
            r'\alpha': 20.0,
            r'\beta':  0.25, # this was before the switch from alpha*beta |--> beta
            r'\theta':  0.2
    }
    speed, width = get_speed_and_width(params, verbose=True,
                                       speed_guess=1.03,
                                       width_guess=9.343)
    speed, width = get_speed_and_width(params, verbose=True,
                                 speed_guess=0.227,
                                 width_guess=1.76)
    # used for initial root-finding for unstable branch
    # load c, Delta, and F from the get_speed_and_width function
    foo = sym.lambdify((c, Delta), F.norm())
    Cs, Ds = np.meshgrid(np.linspace(1e-10, 1.1, 3001),
                         np.linspace(0, 15, 201))
    Zs = np.log(foo(Cs, Ds))
    plt.pcolormesh(Cs, Ds, Zs)
    plt.colorbar()
