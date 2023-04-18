from numpy import pi

from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory

from drone_atc.index import BaseIndex


def params_from_non_dim(Na, d, T, Rc, Ra, A):
    S = 1
    dt = 1

    area = (pi * Na * S ** 2) / d
    l = area ** 0.5

    vcs = S / (T * dt)

    tc_max = (Ra * S) / vcs

    r_com = Rc * S

    a_max = (A * dt) / vcs

    return ModelParameters(n_agents=Na, s=S, a_max=a_max, v_cs=vcs, r_com=r_com, l=l, tc_max=tc_max, A=A)


@dataclass
class SHM:
    shm: SharedMemory
    shape: tuple
    dtype: type


@dataclass
class ModelSHM:
    map: SHM
    agents: SHM
    analytics: SHM


@dataclass
class ModelParameters:
    n_agents: int
    s: float
    a_max: float
    v_cs: float
    r_com: float
    l: float
    tc_max: float
    A: float


@dataclass
class ModelConfig:
    agent: object
    spatial_index: BaseIndex
    n_processes: int
    n_steps: int
    params: ModelParameters
    animate: bool
