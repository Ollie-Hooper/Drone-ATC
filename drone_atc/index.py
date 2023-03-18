from abc import ABC, abstractmethod

import numpy as np
import shapely

from sklearn.neighbors import KDTree, BallTree as SKLearnBallTree

from numpy.typing import ArrayLike

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
    def __init__(self, n_agents):
        self.index = np.arange(n_agents)

    def update(self, agents: ArrayLike):
        pass

    def agents_in_range(self, agent: int, r: float) -> np.ndarray:
        return self.index


class BruteForceIndex(BaseIndex):
    def __init__(self, n_agents):
        self.index = np.empty((n_agents, 2))

    def update(self, agents: ArrayLike):
        self.index = agents

    def agents_in_range(self, agent: int, r: float) -> np.ndarray:
        list_mag = lambda x: np.array([mag(v) for v in x])
        in_range = np.argwhere(list_mag(self.index[agent] - self.index) <= r).flatten()
        return in_range


class RTree(BaseIndex):
    def __init__(self, n_agents):
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
    def __init__(self, n_agents):
        self.tree = None
        self.points = []

    def update(self, agents: ArrayLike):
        self.points = agents
        self.tree = KDTree(self.points)

    def agents_in_range(self, agent: int, r: float) -> np.ndarray:
        in_range = self.tree.query_radius(self.points[agent].reshape(1, -1), r)[0]
        return in_range


class BallTree(BaseIndex):
    def __init__(self, n_agents):
        self.tree = None
        self.points = []

    def update(self, agents: ArrayLike):
        self.points = agents
        self.tree = SKLearnBallTree(self.points)

    def agents_in_range(self, agent: int, r: float) -> np.ndarray:
        in_range = self.tree.query_radius(self.points[agent].reshape(1, -1), r)[0]
        return in_range
