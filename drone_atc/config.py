from numpy import pi

from dataclasses import dataclass
from enum import Enum
from multiprocessing.shared_memory import SharedMemory

from drone_atc.agent import Agent
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
    agent: Agent
    spatial_index: BaseIndex
    n_processes: int
    n_steps: int
    params: ModelParameters
    animate: bool


class Analytics(Enum):
    STEP_EXECUTION_TIME = 0
    READ_TIME = 1
    WRITE_TIME = 2
    INDEX_UPDATE_TIME = 3
    AGENT_STEP_MIN = 4
    AGENT_STEP_MAX = 5
    AGENT_STEP_MEAN = 6
    CONFLICTS = 7
    CALCULATIONS = 8
    AVOIDING = 9
    COLLISIONS = 10
    # RANGE_SEARCH_MIN = 2
    # RANGE_SEARCH_MAX = 3
    # RANGE_SEARCH_MEAN = 4
