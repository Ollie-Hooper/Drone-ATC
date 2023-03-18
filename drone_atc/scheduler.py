import time
from multiprocessing import Process, Barrier
from multiprocessing.shared_memory import SharedMemory

import numpy as np

from drone_atc.config import SHM, ModelSHM, ModelConfig


class MPModelManager:
    #  Creates and assigns agents
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model_shm = None

    def go(self):
        n_model_processes = self.config.n_processes - 1

        process_agent_map = self.create_process_agent_map(n_model_processes)
        agent_attrs = self.create_agent_attrs()
        analytics = self.create_analytics_array(n_model_processes)

        self.setup_shared_memory(process_agent_map, agent_attrs, analytics)

        barrier = Barrier(n_model_processes)

        schedulers = []
        for i in range(n_model_processes):
            schedulers.append(
                Model(self.config, self.model_shm, barrier, i))

        for s in schedulers:
            s.start()

        for s in schedulers:
            s.join()

        analytics = self.get_shm_array(self.model_shm.analytics)[:]

        return analytics

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        if self.model_shm is not None:
            for k, shm in self.model_shm.__dict__.items():
                shm.shm.close()
                shm.shm.unlink()

    def create_process_agent_map(self, n_processes):
        return np.arange(self.config.params.n_agents, dtype=np.int32).reshape(n_processes, -1)

    def create_agent_attrs(self):
        return np.random.random((self.config.params.n_agents, len(self.config.agent.attributes)))

    def create_analytics_array(self, n_processes):
        return np.ndarray((n_processes, self.config.n_steps), dtype=np.float64)

    def setup_shared_memory(self, map_array, agent_attrs, analytics):
        map_shm = self.create_shm(map_array)
        agent_shm = self.create_shm(agent_attrs)
        analytics_shm = self.create_shm(analytics)
        self.model_shm = ModelSHM(map_shm, agent_shm, analytics_shm)

    @staticmethod
    def create_shm(a):
        shm = SharedMemory(create=True, size=a.nbytes)
        b = np.ndarray(a.shape, dtype=a.dtype, buffer=shm.buf)
        b[:] = a[:]
        return SHM(shm, a.shape, a.dtype)

    @staticmethod
    def get_shm_array(shm):
        return np.ndarray(shm.shape, dtype=shm.dtype, buffer=shm.shm.buf)


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

        self.agent = self.config.agent
        self.attrs = self.config.params
        self.index = self.config.spatial_index(self.config.params.n_agents)

    def reconnect_shm(self):
        self.model_shm.map.shm = SharedMemory(name=self.model_shm.map.shm.name)
        self.model_shm.agents.shm = SharedMemory(name=self.model_shm.agents.shm.name)
        self.model_shm.analytics.shm = SharedMemory(name=self.model_shm.analytics.shm.name)

    def run(self):
        self.map = MPModelManager.get_shm_array(self.model_shm.map)
        self.global_agent_attrs = MPModelManager.get_shm_array(self.model_shm.agents)
        self.analytics = MPModelManager.get_shm_array(self.model_shm.analytics)

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
        self.read()
        self.update_index()
        self.barrier.wait()
        self.write()
        self.barrier.wait()
        te = time.time()
        self.analytics[self.id, n] = te - ts

    def read(self):
        self.agent_attrs[:] = self.global_agent_attrs[:]

    def update_index(self):
        self.index.update(self.agent_attrs[:, 0:2])

    def write(self):
        for agent in self.map[self.id]:
            self.global_agent_attrs[agent] = self.agent.step(agent, self, self.agent_attrs)
