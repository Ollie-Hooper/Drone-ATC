from numba import njit
from numpy import sqrt


@njit(cache=True)
def mag(x):
    return sqrt(x.dot(x))
