import numpy as np
from numpy import sqrt, dot

from drone_atc.tools import mag


class Agent:
    attributes = []

    @staticmethod
    def step(uid, model, attr: np.array, alters: np.array, *args, **kwargs) -> np.array:
        return attr


class Drone(Agent):
    global RX
    global RY
    global VX
    global VY
    global T_RX
    global T_RY
    global AVOIDING
    attributes = ['RX', 'RY', 'VX', 'VY', 'T_RX', 'T_RY', 'AVOIDING']
    for i, k in enumerate(attributes):
        globals()[k] = i

    @staticmethod
    def step(uid, model, attrs: np.array, *args, **kwargs) -> np.array:
        s = model.attrs.s
        a_max = model.attrs.a_max
        v_cs = model.attrs.v_cs
        r_com = model.attrs.r_com

        attr = attrs[uid]
        r = np.array([attr[RX], attr[RY]])
        v = np.array([attr[VX], attr[VY]])
        target_r = np.array([attr[T_RX], attr[T_RY]])
        avoiding = False

        in_range = model.index.agents_in_range(uid, r_com)
        in_range = np.delete(in_range, in_range == uid)

        for alt in attrs[in_range]:
            d_min = Drone.calc_min_distance(attr, alt)
            if d_min < s:
                avoiding = True

        if avoiding:
            target_v = np.array([v[1], -v[0]])
        else:
            target_v = v_cs * (target_r - r) / mag(target_r - r)

        if np.not_equal(v, target_v).any():
            v = Drone.accelerate(v, target_v, a_max, v_cs)

        attr[AVOIDING] = avoiding

        return attr

    @staticmethod
    def accelerate(v, target_v, a_max, v_cs):
        delta_v = target_v - v
        a = min(delta_v, a_max * delta_v / mag(delta_v), key=mag)
        v = v_cs * (v + a) / mag(v + a)
        return v

    @staticmethod
    def calc_min_distance(attr, alter):
        r = np.array([attr[RX], attr[RY]])
        v = np.array([attr[VX], attr[VY]])
        alt_r = np.array([alter[RX], alter[RY]])
        alt_v = np.array([alter[VX], alter[VY]])

        r_ab = alt_r - r
        v_ab = alt_v - v

        return sqrt(mag(r_ab) ** 2 - (dot(r_ab, v_ab)) ** 2 / (mag(v_ab)) ** 2)
