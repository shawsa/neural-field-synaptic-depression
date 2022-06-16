'''
Uses the tqdm package to wrap solution generators from the 
TimeIntegrator class.
'''


from time_integrator import TimeIntegrator
from time_domain import TimeDomain

from tqdm import tqdm

class TqdmWrapper:

    def __init__(self, solver: TimeIntegrator):
        self.solver = solver


    def solve(self, u0, rhs, time: TimeDomain):
        return list(tqdm(self.solver.solution_generator(u0, rhs, time),
                         total=time.steps))

    def t_final(self, u0, rhs, time: TimeDomain):
        for u in tqdm(self.solver.solution_generator(u0, rhs, time),
                      total=time.steps):
            pass
        return u
