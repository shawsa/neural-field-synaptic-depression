
import experiment_defaults

import matplotlib.pyplot as plt
import numpy as np
import os.path
import pickle

from neural_field import ParametersBeta, exponential_weight_kernel
from helper_symbolics import (
        find_symbol_by_string,
        free_symbols_in,
        expr_dict,
        get_traveling_pulse,
        get_speed_and_width,
        get_adjoint_nullspace,
        recursive_reduce)



FILE_NAME = os.path.join(experiment_defaults.data_path,
                         'apparant_motion.pickle')

params = ParametersBeta(**{
    'alpha': 20.0,
    'beta': 5.0,
    'mu': 1.0,
})
params_dict = params.dict | {
        'theta': 0.2,
        'weight_kernel': exponential_weight_kernel
}
sympy_params = {key: val for key, val in params_dict.items()
                if key not in ['weight_kernel', 'gamma', 'beta']}
# helper_symbolics uses old definition of beta
sympy_params['beta'] = params.beta/params.alpha
USE_SAVED_PARAMETERS = True
if USE_SAVED_PARAMETERS:
    c = 1.0300285382703882
    Delta = 9.342281657861955
else:
    c, Delta = get_speed_and_width(
            sympy_params,
            speed_guess=1.05,
            width_guess=9.5)
params_dict = params_dict | {
        'c': c,
        'Delta': Delta,
}
sympy_params |= {'c': c, 'Delta': Delta}

# using an ancient sympy incantation
symbol_set = free_symbols_in([
        recursive_reduce(expr)
        for expr in expr_dict['speed_width_conditions']
])

symbol_params = {find_symbol_by_string(symbol_set, key): value
                 for key, value in sympy_params.items()}

U, Q, *_ = get_traveling_pulse(symbol_params, validate=False)

xs = np.linspace(-100, 100, 2001)
plt.plot(xs, U(xs))
