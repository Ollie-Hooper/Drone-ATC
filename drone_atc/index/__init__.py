from abc import ABC, abstractmethod

import numpy as np
import shapely
from numba import njit
from numba.typed import List

from sklearn.neighbors import KDTree, BallTree as SKLearnBallTree

from numpy.typing import ArrayLike

from drone_atc.index import grid as _Grid, brute_force
from drone_atc.tools import mag


class BaseIndex(ABC):
    @abstractmethod
    def __init__(self, n_agents: int):
        pass

    @abstractmethod
    def update(self, agents: ArrayLike):
        pass

    @abstractmethod
    def agents_in_range(self, agent: int, r: float) -> np.ndarray:
        pass


class NoIndex(BaseIndex):
    def __init__(self, n_agents, *args, **kwargs):
        self.index = np.arange(n_agents)

    def update(self, agents: ArrayLike):
        pass

    def agents_in_range(self, agent: int, r: float) -> np.ndarray:
        return self.index


class BruteForceIndex(BaseIndex):
    def __init__(self, n_agents, *args, **kwargs):
        self.index = np.empty((n_agents, 2))

    def update(self, agents: ArrayLike):
        self.index = agents

    def agents_in_range(self, agent: int, r: float) -> np.ndarray:
        return brute_force.agents_in_range(self.index, agent, r)


class RTree(BaseIndex):
    def __init__(self, *args, **kwargs):
        self.tree = None
        self.points = []

    def update(self, agents: ArrayLike):
        self.points = [shapely.Point(r[0], r[1]) for r in agents]
        self.tree = shapely.STRtree(self.points)

    def agents_in_range(self, agent: int, r: float) -> np.ndarray:
        query_geom = self.points[agent].buffer(r)
        in_range = self.tree.query(query_geom)
        return in_range


class Quadtree(BaseIndex):
    def __init__(self, *args, **kwargs):
        self.tree = None
        self.points = []

    def update(self, agents: ArrayLike):
        self.points = agents
        self.tree = KDTree(self.points)

    def agents_in_range(self, agent: int, r: float) -> np.ndarray:
        in_range = self.tree.query_radius(self.points[agent].reshape(1, -1), r)[0]
        return in_range


class BallTree(BaseIndex):
    def __init__(self, *args, **kwargs):
        self.tree = None
        self.points = []

    def update(self, agents: ArrayLike):
        self.points = agents
        self.tree = SKLearnBallTree(self.points)

    def agents_in_range(self, agent: int, r: float) -> np.ndarray:
        in_range = self.tree.query_radius(self.points[agent].reshape(1, -1), r)[0]
        return in_range


class Grid(BaseIndex):
    def __init__(self, n_agents, l, r):
        self.gcs = r*0.5
        self.num_rows_cols = int(l / self.gcs)
        self.gcs = l / self.num_rows_cols
        self.grid = None

        self.secondary_index = np.empty((n_agents, 4), dtype=np.float64)

        self.initialised = False

    def update(self, agents):
        grid, secondary_index, initialised = _Grid.update(self.grid, self.secondary_index, self.initialised,
                                                     self.num_rows_cols, self.gcs, agents)
        self.grid = grid
        self.secondary_index = secondary_index
        self.initialised = initialised

    def agents_in_range(self, agent: int, r: float) -> np.ndarray:
        return _Grid.agents_in_range(self.grid, self.secondary_index, self.num_rows_cols, self.gcs, agent, r)
