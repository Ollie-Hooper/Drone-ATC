from abc import ABC, abstractmethod

import numpy as np
import shapely
from numba import njit
from numba.typed import List

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
        list_mag = lambda x: np.array([mag(v) for v in x])
        in_range = np.argwhere(list_mag(self.index[agent] - self.index) <= r).flatten()
        return in_range


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
    def __init__(self, n_agents, l, r):  # grid_area, gcs):
        self.grid_area = [0, 0, l, l]
        self.gcs = 0.5 * r
        self.num_cols = int((self.grid_area[2] - self.grid_area[0]) / self.gcs)
        self.num_rows = int((self.grid_area[3] - self.grid_area[1]) / self.gcs)
        self.grid = np.empty((self.num_rows, self.num_cols), dtype=object)
        self.secondary_index = {}

        self.points = []

    def update(self, agents):
        for oid, (x, y) in enumerate(agents):
            self.update_agent(oid, x, y)

    def update_agent(self, oid, x, y):
        if oid in self.secondary_index:
            old_row, old_col = self.secondary_index[oid]
            old_cell = self.grid[old_row][old_col]
            if oid in old_cell:
                old_pos = old_cell.pop(oid)
                if not old_cell:
                    self.grid[old_row][old_col] = None
            else:
                old_pos = [oid, x, y]
            ptr = old_pos
        else:
            row, col = self.get_row_col(x, y)
            old_row, old_col = row, col
            old_pos = [oid, x, y]
            ptr = old_pos
            if self.grid[row][col] is None:
                self.grid[row][col] = {}
        self.secondary_index[oid] = (row, col)
        new_row, new_col = self.get_row_col(x, y)
        if new_row != old_row or new_col != old_col:
            if self.grid[new_row][new_col] is None:
                self.grid[new_row][new_col] = {}
            ptr = [oid, x, y]
            self.grid[new_row][new_col][oid] = ptr
            self.secondary_index[oid] = (new_row, new_col)
        old_pos[1], old_pos[2] = x, y

    def agents_in_range(self, xqmin, yqmin, xqmax, yqmax):
        result = []
        min_row, min_col = self.get_row_col(xqmin, yqmin)
        max_row, max_col = self.get_row_col(xqmax, yqmax)
        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                if self.grid[row][col] is not None:
                    for oid, ptr in self.grid[row][col].items():
                        if xqmin <= ptr[1] <= xqmax and yqmin <= ptr[2] <= yqmax:
                            result.append(oid)
        return result

    def get_row_col(self, x, y):
        col = int((x - self.grid_area[0]) / self.gcs)
        row = int((y - self.grid_area[1]) / self.gcs)
        return row, col


class Grid(BaseIndex):
    def __init__(self, n_agents, l, r):
        self.gcs = r / 2
        self.num_rows_cols = int(l / self.gcs)
        self.gcs = l / self.num_rows_cols
        # self.grid = np.empty(shape=(self.num_rows_cols, self.num_rows_cols), dtype=object)
        # for i in range(self.grid.shape[0]):
        #     for j in range(self.grid.shape[1]):
        #         self.grid[i][j] = []
        self.grid = None

        # for i in range(len(self.grid)):
        #     for j in range(len(self.grid[i])):
        #         self.grid[i][j].remove(1)

        self.secondary_index = np.empty((n_agents, 4), dtype=np.float64)

        self.initialised = False

    def update(self, agents):
        if not self.initialised:
            self.grid = List(
                [List([List([1, ]) for j in range(self.num_rows_cols)]) for i in range(self.num_rows_cols)])
            for i in range(len(self.grid)):
                for j in range(len(self.grid[i])):
                    self.grid[i][j].remove(1)
        grid, secondary_index, initialised = _update(self.grid, self.secondary_index, self.initialised,
                                                     self.num_rows_cols, self.gcs, agents)
        self.grid = grid
        self.secondary_index = secondary_index
        self.initialised = initialised
        # for oid, (x, y) in enumerate(agents):
        #     self.update_agent(oid, x, y)
        # self.initialised = True

    def update_agent(self, oid, x, y):
        row, col = self.get_row_col(x, y)
        if not self.initialised:
            self.secondary_index[oid] = np.array((x, y, row, col))
            self.grid[row, col].add(oid)
        _, _, old_row, old_col = self.secondary_index[oid]
        old_row, old_col = int(old_row), int(old_col)

        if row != old_row or col != old_col:
            self.grid[old_row, old_col].remove(oid)
        self.grid[row, col].add(oid)
        self.secondary_index[oid] = np.array((x, y, row, col))

    def agents_in_range(self, agent: int, r: float) -> np.ndarray:
        return _agents_in_range(self.grid, self.secondary_index, self.num_rows_cols, self.gcs, agent, r)
        # cx, cy = self.secondary_index[agent][:2]
        # xqmin, xqmax = cx - r, cx + r
        # yqmin, yqmax = cy - r, cy + r
        #
        # min_row, min_col = self.get_row_col(xqmin, yqmin)
        # max_row, max_col = self.get_row_col(xqmax, yqmax)
        #
        # in_range = []
        #
        # for row in range(min_row, max_row + 1):
        #     for col in range(min_col, max_col + 1):
        #         in_range.extend(list(self.grid[row, col]))
        #
        # return np.array(in_range)

    def get_row_col(self, x, y):
        col = min(self.num_rows_cols - 1, max(0, int(np.floor(x / self.gcs))))
        row = min(self.num_rows_cols - 1, max(0, int(np.floor(y / self.gcs))))
        return row, col


@njit(cache=True)
def _update(grid, secondary_index, initialised, num_rows_cols, gcs, agents):
    oid = 0
    for x, y in agents:
        update_agent(grid, secondary_index, initialised, num_rows_cols, gcs, oid, x, y)
        oid+=1
    initialised = True
    return grid, secondary_index, initialised


@njit(cache=True)
def update_agent(grid, secondary_index, initialised, num_rows_cols, gcs, oid, x, y):
    row, col = get_row_col(num_rows_cols, gcs, x, y)
    if not initialised:
        secondary_index[oid] = np.array((x, y, row, col))
        grid[row][col].append(oid)
    _, _, old_row, old_col = secondary_index[oid]
    old_row, old_col = int(old_row), int(old_col)

    if row != old_row or col != old_col:
        grid[old_row][old_col].remove(oid)
    grid[row][col].append(oid)
    secondary_index[oid] = np.array((x, y, row, col))


@njit(cache=True)
def get_row_col(num_rows_cols, gcs, x, y):
    col = min(num_rows_cols - 1, max(0, int(np.floor(x / gcs))))
    row = min(num_rows_cols - 1, max(0, int(np.floor(y / gcs))))
    return row, col


@njit(cache=True)
def _agents_in_range(grid, secondary_index, num_rows_cols, gcs, agent: int, r: float) -> np.ndarray:
    cx, cy = secondary_index[agent][:2]
    xqmin, xqmax = cx - r, cx + r
    yqmin, yqmax = cy - r, cy + r

    min_row, min_col = get_row_col(num_rows_cols, gcs, xqmin, yqmin)
    max_row, max_col = get_row_col(num_rows_cols, gcs, xqmax, yqmax)

    in_range = []

    for row in range(min_row, max_row + 1):
        for col in range(min_col, max_col + 1):
            in_range.extend(list(grid[row][col]))

    return np.array(in_range)
