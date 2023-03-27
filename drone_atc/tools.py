from numba import jit
from numpy import sqrt


@jit(nopython=True, cache=True)
def mag(x):
    return sqrt(x.dot(x))
