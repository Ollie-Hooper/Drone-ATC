import numpy as np
from numba import njit
from numpy.typing import ArrayLike

from drone_atc.tools import mag


@njit(cache=True)
def agents_in_range(index, agent: int, r: float) -> np.ndarray:
    list_mag = lambda x: np.array([mag(v) for v in x])
    # r_ab = index[agent] - index
    # np.sqrt(np.dot(r_ab, r_ab))
    in_range = np.argwhere(list_mag(index[agent] - index) <= r).flatten()
    return in_range