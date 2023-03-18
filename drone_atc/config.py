from dataclasses import dataclass
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
