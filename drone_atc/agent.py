from dataclasses import dataclass
from typing import List

import numpy as np
from numpy import sqrt, dot


class Agent:
    @staticmethod
    def step(attr: np.array, alters: np.array, *args, **kwargs) -> np.array:
        return attr


class Drone(Agent):
    global RX
    global RY
    global VX
    global VY
    global AVOID
    attributes = ['RX', 'RY', 'VX', 'VY', 'AVOID']
    for i, k in enumerate(attributes):
        globals()[k] = i

    @staticmethod
    def step(attr: np.array, alters: np.array, *args, **kwargs) -> np.array:
        for alt in alters:
            d_min = Drone.calc_min_distance(attr, alt)
            # print(d_min)

        return attr

    @staticmethod
    def accelerate(attr):
        return attr

    @staticmethod
    def calc_min_distance(attr, alter):
        mag = lambda x: sqrt(x.dot(x))

        r = np.array([attr[RX], attr[RY]])
        v = np.array([attr[VX], attr[VY]])
        alt_r = np.array([alter[RX], alter[RY]])
        alt_v = np.array([alter[VX], alter[VY]])

        r_ab = alt_r - r
        v_ab = alt_v - v

        return sqrt(mag(r_ab) ** 2 - (dot(r_ab, v_ab)) ** 2 / (mag(v_ab)) ** 2)
