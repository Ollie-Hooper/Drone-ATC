import time
from multiprocessing import Process, Barrier
from multiprocessing.shared_memory import SharedMemory

import importlib

import memray
import numpy as np
from numpy import pi, cos, sin
from scipy.stats._qmc import PoissonDisk

from drone_atc import drone
from drone_atc.analytics import Analyser
from drone_atc.config import SHM, ModelSHM, ModelConfig
from drone_atc.analytics_config import Analytics
from drone_atc.index import grid
from drone_atc.shared_mem import create_shm, get_shm_array
from drone_atc.tools import generate_poisson_disk_samples, generate_uniform_points_with_min_distance, plot_points


class MPModelManager:
    #  Creates and assigns agents

    def __init__(self, config: ModelConfig):
        if isinstance(config, ModelConfig):
            self.config = config
            self.configs = None
        elif isinstance(config, list):
            self.config = config[0]
            self.configs = config
            self.config.n_steps = len(self.configs) + 1
        else:
            raise Exception()
        self.model_shm = None

        self.agent = importlib.import_module(f'.{self.config.agent}', __name__.split('.')[0])

    def go(self):
        n_model_processes = self.config.n_processes - 1 if self.config.animate else self.config.n_processes

        process_agent_map = self.create_process_agent_map(n_model_processes)
        agent_attrs = self.create_agent_attrs(self.config, self.agent)
        analytics = self.create_analytics_array(n_model_processes)

        self.setup_shared_memory(process_agent_map, agent_attrs, analytics)

        read_barrier = Barrier(self.config.n_processes)
        write_barrier = Barrier(self.config.n_processes)

        processes = []
        for i in range(n_model_processes):
            processes.append(Model(self.config, self.model_shm, read_barrier, write_barrier, i, self.configs))

        if self.config.animate:
            processes.append(Analyser(self.config, self.model_shm, read_barrier, write_barrier))
        
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

    @staticmethod
    def create_agent_attrs(config, agent):
        n_agents = config.params.n_agents
        n_attrs = len(agent.attributes)

        a = np.zeros((n_agents, n_attrs))

        l = config.params.l
        v_cs = config.params.v_cs
        s = config.params.s

        square_size = l
        min_distance = s*2

        rx, ry = generate_uniform_points_with_min_distance(square_size, min_distance, n_agents)

        # plot_points(rx, ry, min_distance, square_size)

        angles = 2*pi*np.random.random(n_agents)
        vx = v_cs*cos(angles)
        vy = v_cs*sin(angles)

        a[:, drone.RX] = rx
        a[:, drone.RY] = ry
        a[:, drone.VX] = vx
        a[:, drone.VY] = vy
        a[:, drone.T_RX] = rx + 1e6 * vx
        a[:, drone.T_RY] = ry + 1e6 * vy

        return a

    def create_analytics_array(self, n_processes):
        return np.zeros((n_processes, self.config.n_steps, len(Analytics)), dtype=np.float64)

    def setup_shared_memory(self, map_array, agent_attrs, analytics):
        map_shm = create_shm(map_array)
        agent_shm = create_shm(agent_attrs)
        analytics_shm = create_shm(analytics)
        self.model_shm = ModelSHM(map_shm, agent_shm, analytics_shm)


class Model(Process):
    def __init__(self, config: ModelConfig, model_shm: ModelSHM, read_barrier, write_barrier, process_id, mult_configs=None):
        super().__init__()
        self.config = config
        self.model_shm = model_shm
        self.read_barrier = read_barrier
        self.write_barrier = write_barrier
        self.id = process_id

        self.map = None
        self.global_agent_attrs = None
        self.agent_attrs = None
        self.analytics = None

        self.agent = None
        self.attrs = self.config.params
        self.index = self.config.spatial_index(self.config.params.n_agents, self.config.params.l, self.config.params.r_com)
        # self.index = Grid(self.config.params.n_agents, self.config.params.l,
        #                                        self.config.params.r_com)

        self.configs = mult_configs

        self.n_agents = None

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

        n_steps = self.config.n_steps

        if (idx := np.argwhere(self.map[self.id] == -1)).any():
            self.n_agents = idx[0][0]
        else:
            self.n_agents = len(self.map[self.id])

        for i in range(n_steps):
            if self.configs and i:
                n = i-1
                if self.id == 0:
                    self.recreate_agent_params(n)
                self.config = self.configs[n]
                self.attrs = self.config.params
                # self.index = self.config.spatial_index(self.config.params.n_agents)
                self.read_barrier.wait()
            self.step(i)

        for k, shm in self.model_shm.__dict__.items():
            shm.shm.close()

    def update_agents(self, additions, deletions, modifications):
        pass

    def step(self, n):
        if self.id == 0:
            print(n)
        ts = time.time()
        self.read(n)
        self.update_index(n)
        self.read_barrier.wait()
        self.write(n)
        self.write_barrier.wait()
        te = time.time()
        self.analytics[self.id, n, Analytics.STEP_EXECUTION_TIME.value] = te - ts

    def recreate_agent_params(self, n):
        self.global_agent_attrs[:] = MPModelManager.create_agent_attrs(self.configs[n], self.agent)

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
        times = np.empty(self.n_agents)
        rq_times = np.empty(self.n_agents)
        for i, agent in enumerate(self.map[self.id]):
            if agent == -1:
                continue
            ts = time.time()
            self.global_agent_attrs[agent], rq_time = self.agent.step(agent, self.attrs, self.index, self.agent_attrs)
            te = time.time()
            times[i] = te - ts
            rq_times[i] = rq_time
        _te = time.time()

        conflict_total = sum(self.global_agent_attrs[self.map[self.id]][:, drone.CONFLICTS])
        calculation_total = sum(self.global_agent_attrs[self.map[self.id]][:, drone.CALCULATIONS])
        avoiding_total = sum(self.global_agent_attrs[self.map[self.id]][:, drone.AVOIDING])
        collision_total = sum(self.global_agent_attrs[self.map[self.id]][:, drone.COLLISIONS])

        self.analytics[self.id, n, Analytics.CONFLICTS.value] = conflict_total
        self.analytics[self.id, n, Analytics.CALCULATIONS.value] = calculation_total
        self.analytics[self.id, n, Analytics.AVOIDING.value] = avoiding_total
        self.analytics[self.id, n, Analytics.COLLISIONS.value] = collision_total

        self.analytics[self.id, n, Analytics.AGENT_STEP_MIN.value] = min(times)
        self.analytics[self.id, n, Analytics.AGENT_STEP_MAX.value] = max(times)
        self.analytics[self.id, n, Analytics.AGENT_STEP_MEAN.value] = times.mean()
        self.analytics[self.id, n, Analytics.WRITE_TIME.value] = _te - _ts

        self.analytics[self.id, n, Analytics.RANGE_SEARCH_MIN.value] = min(rq_times)
        self.analytics[self.id, n, Analytics.RANGE_SEARCH_MAX.value] = max(rq_times)
        self.analytics[self.id, n, Analytics.RANGE_SEARCH_MEAN.value] = rq_times.mean()
