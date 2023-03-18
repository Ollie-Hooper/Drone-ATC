import time
from multiprocessing import Process, Barrier
from multiprocessing.shared_memory import SharedMemory

import numpy as np

from drone_atc.index import NoIndex, BruteForceIndex, RTree, Quadtree, BallTree


class MPModelManager:
    #  Creates and assigns agents
    def __init__(self, child, n_children, agent, n_agents, n_steps):
        self.analytics_shm = None
        self.agent = agent
        self.agent_attrs_shm = None
        self.n_attributes = len(agent.attributes)
        self.map_shm = None
        self.child = child
        self.n_children = n_children
        self.n_agents = n_agents
        self.n_steps = n_steps
        self.create_process_agent_map()
        self.create_agent_attrs()
        self.create_analytics()

    def go(self):
        barrier = Barrier(self.n_children)

        schedulers = []
        for i in range(self.n_children):
            schedulers.append(
                Model(barrier, self.agent, self.map_shm.name, self.agent_attrs_shm.name, self.analytics_shm.name,
                      self.n_children, self.n_agents, self.n_attributes, self.n_steps, i))

        for s in schedulers:
            s.start()

        for s in schedulers:
            s.join()

        return np.ndarray((self.n_children, self.n_steps), dtype=np.float64, buffer=self.analytics_shm.buf)[:]

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.map_shm.close()
        self.map_shm.unlink()
        self.agent_attrs_shm.close()
        self.agent_attrs_shm.unlink()
        self.analytics_shm.close()
        self.analytics_shm.unlink()

    def create_process_agent_map(self):
        a = np.arange(self.n_agents * self.n_children, dtype=np.int32).reshape(self.n_children, -1)
        self.map_shm = SharedMemory(create=True, size=a.nbytes)
        b = np.ndarray(a.shape, dtype=a.dtype, buffer=self.map_shm.buf)
        b[:] = a[:]

    def create_agent_attrs(self):
        a = np.random.random((self.n_children * self.n_agents, self.n_attributes))  # , dtype=np.float64)
        self.agent_attrs_shm = SharedMemory(create=True, size=a.nbytes)
        b = np.ndarray(a.shape, dtype=a.dtype, buffer=self.agent_attrs_shm.buf)
        b[:] = a[:]

    def create_analytics(self):
        a = np.ndarray((self.n_children, self.n_steps), dtype=np.float64)
        self.analytics_shm = SharedMemory(create=True, size=a.nbytes)
        b = np.ndarray(a.shape, dtype=a.dtype, buffer=self.analytics_shm.buf)
        b[:] = a[:]


class Model(Process):
    def __init__(self, barrier: Barrier, agent, map_shm_name, attr_shm_name, analytics_shm_name, n_processes, n_agents,
                 n_attributes, n_steps, process_id):
        super().__init__()
        self.analytics = None
        self.analytics_shm_name = analytics_shm_name
        self.analytics_shm = None
        self.n_agents = n_agents
        self.n_attributes = n_attributes
        self.agent = agent
        self.agent_attrs = None
        self.global_agent_attrs = None
        self.agent_attrs_shm = None
        self.map = None
        self.map_shm = None
        self.barrier = barrier
        self.id = process_id
        self.map_shm_name = map_shm_name
        self.attr_shm_name = attr_shm_name
        self.n_processes = n_processes
        self.n_steps = n_steps

        self.attrs = dict(
            s=0.1,
            a_max=0.1,
            v_cs=0.1,
            r_com=0.1,
        )
        self.index = BallTree(n_agents*n_processes)

    def run(self):
        self.map_shm = SharedMemory(name=self.map_shm_name)
        self.map = np.ndarray((self.n_processes, self.n_agents), dtype=np.int32, buffer=self.map_shm.buf)

        self.analytics_shm = SharedMemory(name=self.analytics_shm_name)
        self.analytics = np.ndarray((self.n_processes, self.n_steps), dtype=np.float64, buffer=self.analytics_shm.buf)

        self.agent_attrs_shm = SharedMemory(name=self.attr_shm_name)
        self.global_agent_attrs = np.ndarray((self.n_agents * self.n_processes, self.n_attributes), dtype=np.float64,
                                             buffer=self.agent_attrs_shm.buf)
        self.agent_attrs = np.ndarray(self.global_agent_attrs.shape, dtype=self.global_agent_attrs.dtype)
        self.agent_attrs[:] = self.global_agent_attrs[:]

        for i in range(self.n_steps):
            self.step()

        self.map_shm.close()
        self.agent_attrs_shm.close()

    def update_agents(self, additions, deletions, modifications):
        pass

    def step(self):
        ts = time.time()
        self.read()
        self.update_index()
        self.barrier.wait()
        self.write()
        self.barrier.wait()
        te = time.time()
        self.analytics[self.id] = te - ts

    def read(self):
        self.agent_attrs[:] = self.global_agent_attrs[:]

    def update_index(self):
        self.index.update(self.agent_attrs[:, 0:2])

    def write(self):
        for agent in self.map[self.id]:
            self.global_agent_attrs[agent] = self.agent.step(agent, self, self.agent_attrs)
