from multiprocessing.shared_memory import SharedMemory

import numpy as np

from drone_atc.config import SHM


def create_shm(a):
    shm = SharedMemory(create=True, size=a.nbytes)
    b = np.ndarray(a.shape, dtype=a.dtype, buffer=shm.buf)
    b[:] = a[:]
    return SHM(shm, a.shape, a.dtype)


def get_shm_array(shm):
    return np.ndarray(shm.shape, dtype=shm.dtype, buffer=shm.shm.buf)
