import time
from multiprocessing import Process, Barrier
from multiprocessing.shared_memory import SharedMemory

import importlib
import numpy as np

from drone_atc.analytics import Analyser
from drone_atc.config import SHM, ModelSHM, ModelConfig, Analytics
from drone_atc.shared_mem import create_shm, get_shm_array


class MPModelManager:
    #  Creates and assigns agents
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model_shm = None

        self.agent = importlib.import_module(f'.{self.config.agent}', __name__.split('.')[0])

    def go(self):
        n_model_processes = self.config.n_processes - 1 if self.config.animate else self.config.n_processes

        process_agent_map = self.create_process_agent_map(n_model_processes)
        agent_attrs = self.create_agent_attrs()
        analytics = self.create_analytics_array(n_model_processes)

        self.setup_shared_memory(process_agent_map, agent_attrs, analytics)

        barrier = Barrier(n_model_processes)

        processes = []
        for i in range(n_model_processes):
            processes.append(
                Model(self.config, self.model_shm, barrier, i))

        if self.config.animate:
            processes.append(Analyser(self.model_shm))
        
        for s in processes:
            s.start()

        for s in processes:
            s.join()

        analytics = get_shm_array(self.model_shm.analytics).copy()

        return analytics

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        if self.model_shm is not None:
            for k, shm in self.model_shm.__dict__.items():
                shm.shm.close()
                shm.shm.unlink()

    def create_process_agent_map(self, n_processes):
        split_evenly = n_processes*(self.config.params.n_agents // n_processes)
        leftover = self.config.params.n_agents % n_processes
        initial_array = np.arange(self.config.params.n_agents, dtype=np.int32)
        agents = np.append(initial_array, -np.ones(n_processes - leftover, dtype=np.int32)) if leftover else initial_array
        return agents.reshape(n_processes, -1)

    def create_agent_attrs(self):
        return np.random.random((self.config.params.n_agents, len(self.agent.attributes)))

    def create_analytics_array(self, n_processes):
        return np.zeros((n_processes, self.config.n_steps, len(Analytics)), dtype=np.float64)

    def setup_shared_memory(self, map_array, agent_attrs, analytics):
        map_shm = create_shm(map_array)
        agent_shm = create_shm(agent_attrs)
        analytics_shm = create_shm(analytics)
        self.model_shm = ModelSHM(map_shm, agent_shm, analytics_shm)


class Model(Process):
    def __init__(self, config: ModelConfig, model_shm: ModelSHM, barrier, process_id):
        super().__init__()
        self.config = config
        self.model_shm = model_shm
        self.barrier = barrier
        self.id = process_id

        self.map = None
        self.global_agent_attrs = None
        self.agent_attrs = None
        self.analytics = None

        self.agent = None
        self.attrs = self.config.params
        self.index = self.config.spatial_index(self.config.params.n_agents)

    def reconnect_shm(self):
        self.model_shm.map.shm = SharedMemory(name=self.model_shm.map.shm.name)
        self.model_shm.agents.shm = SharedMemory(name=self.model_shm.agents.shm.name)
        self.model_shm.analytics.shm = SharedMemory(name=self.model_shm.analytics.shm.name)

    def run(self):
        self.agent = importlib.import_module(f'.{self.config.agent}', __name__.split('.')[0])

        self.map = get_shm_array(self.model_shm.map)
        self.global_agent_attrs = get_shm_array(self.model_shm.agents)
        self.analytics = get_shm_array(self.model_shm.analytics)

        self.agent_attrs = np.ndarray(self.global_agent_attrs.shape, dtype=self.global_agent_attrs.dtype)
        self.agent_attrs[:] = self.global_agent_attrs[:]

        for i in range(self.config.n_steps):
            self.step(i)

        for k, shm in self.model_shm.__dict__.items():
            shm.shm.close()

    def update_agents(self, additions, deletions, modifications):
        pass

    def step(self, n):
        ts = time.time()
        self.read(n)
        self.update_index(n)
        self.barrier.wait()
        self.write(n)
        self.barrier.wait()
        te = time.time()
        self.analytics[self.id, n, Analytics.STEP_EXECUTION_TIME.value] = te - ts

    def read(self, n):
        ts = time.time()
        self.agent_attrs[:] = self.global_agent_attrs[:]
        te = time.time()
        self.analytics[self.id, n, Analytics.READ_TIME.value] = te - ts

    def update_index(self, n):
        ts = time.time()
        self.index.update(self.agent_attrs[:, 0:2])
        te = time.time()
        self.analytics[self.id, n, Analytics.INDEX_UPDATE_TIME.value] = te - ts

    def write(self, n):
        _ts = time.time()
        times = np.empty(len(self.map[self.id]))
        for i, agent in enumerate(self.map[self.id]):
            if agent == -1:
                continue
            ts = time.time()
            self.global_agent_attrs[agent] = self.agent.step(agent, self.attrs, self.index, self.agent_attrs)
            te = time.time()
            times[i] = te - ts
        _te = time.time()

        self.analytics[self.id, n, Analytics.AGENT_STEP_MIN.value] = min(times)
        self.analytics[self.id, n, Analytics.AGENT_STEP_MAX.value] = max(times)
        self.analytics[self.id, n, Analytics.AGENT_STEP_MEAN.value] = times.mean()
        self.analytics[self.id, n, Analytics.WRITE_TIME.value] = _te - _ts
