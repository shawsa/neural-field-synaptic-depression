'''
Extending the time integrator to accept delta forcing terms as well.
'''

from abc import ABC, abstractmethod, abstractproperty
from time_domain import TimeDomain
from time_integrator import TimeIntegrator

class EulerDeltaTimeIntegrator:

    def update(self, t, u, f, h):
        return u + h*f(t, u)

    def solution_generator(self, u0, time: TimeDomain, rhs, deltas):
        u = u0
        yield u
        for t in time.array[:-1]:
            u = self.update(t, u, rhs, time.spacing)
            yield u

    def solve(self, u0, rhs, time: TimeDomain):
        return time.array, list(self.solution_generator(u0, rhs, time))

    def t_final(self, u0, rhs, time: TimeDomain):
        for u in self.solution_generator(u0, rhs, time):
            pass

