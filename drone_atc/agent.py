from dataclasses import dataclass
from typing import List

import numpy as np
from numpy import sqrt, dot


@dataclass
class AgentAttributes:
    unique_id: str


class Agent:
    @staticmethod
    def step(attr: AgentAttributes, *args, **kwargs) -> AgentAttributes:
        return attr


class DroneAttributes(AgentAttributes):
    pass


class Drone(Agent):
    _rx = 0
    _ry = 1
    _vx = 2
    _vy = 3
    _avoid = 4

    @staticmethod
    def step(attr: DroneAttributes, alters: List[DroneAttributes], *args, **kwargs) -> DroneAttributes:
        for alt in alters:
            d_min = Drone.calc_min_distance(attr, alt)
            #print(d_min)

        return attr

    @staticmethod
    def accelerate(attr):
        return attr

    @staticmethod
    def calc_min_distance(attr, alter):
        mag = lambda x: sqrt(x.dot(x))

        r = np.array([attr[0], attr[1]])
        v = np.array([attr[2], attr[3]])
        alt_r = np.array([alter[0], alter[1]])
        alt_v = np.array([alter[2], alter[3]])

        r_ab = alt_r - r
        v_ab = alt_v - v

        return sqrt(mag(r_ab) ** 2 - (dot(r_ab, v_ab)) ** 2 / (mag(v_ab)) ** 2)
