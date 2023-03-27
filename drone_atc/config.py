from dataclasses import dataclass
from enum import Enum
from multiprocessing.shared_memory import SharedMemory

from drone_atc.agent import Agent
from drone_atc.index import BaseIndex


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
    # RANGE_SEARCH_MIN = 2
    # RANGE_SEARCH_MAX = 3
    # RANGE_SEARCH_MEAN = 4
