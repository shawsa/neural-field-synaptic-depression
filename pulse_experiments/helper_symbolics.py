"""
Helper file that loads symbolic expressions derived in
synaptic_depression_analysis.ipnb, computes dependent
parameters, and gives numeric functions for the traveling
pulse solution and basis functions for the adjoint-nullspace.
"""

import pickle
from functools import reduce
import numpy as np
import numpy.linalg as la
import sympy as sym


# Heaviside numerics workaround
sympy_modules = [{'Heaviside': lambda x: np.heaviside(x, 0.5)}, 'numpy']


"""
This helper file relies on two dictionaries of sympy objects generated by
the Jupyter Notebook synaptic_depression_analysis.ipnb, and saved to the
file below.

expr_dict has keys which are strings, and values which are sympy expressions.
sub_dict has keys which are sympy symbols found in those expressions, and
values which are also sympy expressions.
"""
FILE_PATH = 'synaptic_depression_symbolics.pickle'
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


def ensure_sympy_floats(input_params, symbol_set, precision):
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


def get_speed_and_width(input_params, precision=15):
    symbol_set = free_symbols_in([
            recursive_reduce(expr)
            for expr in expr_dict['speed_width_conditions']
    ])
    params = ensure_sympy_floats(input_params, symbol_set, precision)

    c = find_symbol_by_string(symbol_set, 'c')
    Delta = find_symbol_by_string(symbol_set, 'Delta')
    vec = sym.Matrix([[c], [Delta]])
    F = sym.Matrix(
            [[recursive_reduce(condition).evalf(precision, subs=params)]
             for condition in expr_dict['speed_width_conditions']])
    J = F.jacobian(vec)

    params[c] = sym.Float('1.2', precision)
    params[Delta] = sym.Float('12', precision)
    for _ in range(100):
        update_vec = J.evalf(precision, subs=params).inv() * F.evalf(precision, subs=params)
        params[c] -= update_vec[0]
        params[Delta] -= update_vec[1]
    return params

def get_numerical_parameters(input_params, precision=15, validate=True):
    if validate:
        params = get_speed_and_width(input_params, precision=precision)
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

    eigen_vals, eigen_vect_mat = la.eig(np.array(M, dtype=np.float))
    # find closest to 0
    eigen_val, eigen_vect = sorted(zip(eigen_vals, eigen_vect_mat.T),
                                   key=lambda tup: abs(tup[0]))[0]
    for var, sub in zip(my_vars, eigen_vect):
        params[var] = sub

    return params

def get_traveling_pulse(input_params, precision=15, validate=True):
    if validate:
        params = get_speed_and_width(input_params, precision=precision)
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
