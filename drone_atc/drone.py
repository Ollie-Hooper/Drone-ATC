import numpy as np
from numpy import sqrt, dot
from numba import jit, njit

from drone_atc.tools import mag

RX = 0
RY = 1
VX = 2
VY = 3
T_RX = 4
T_RY = 5
AVOIDING = 6

attributes = ['RX', 'RY', 'VX', 'VY', 'T_RX', 'T_RY', 'AVOIDING']


# for i, k in enumerate(attributes):
#     globals()[k] = i


def step(uid, model_attrs, spatial_index, attrs: np.array) -> np.array:
    s = model_attrs.s
    a_max = model_attrs.a_max
    v_cs = model_attrs.v_cs
    r_com = model_attrs.r_com
    in_range = spatial_index.agents_in_range(uid, r_com)
    in_range = np.delete(in_range, in_range == uid)

    return _step(uid, s, a_max, v_cs, in_range, attrs)


@njit(cache=True)
def _step(uid, s, a_max, v_cs, in_range, attrs):

    attr = attrs[uid]
    r = np.array([attr[RX], attr[RY]])
    v = np.array([attr[VX], attr[VY]])
    target_r = np.array([attr[T_RX], attr[T_RY]])
    avoiding = False

    for alt in attrs[in_range]:
        d_min = calc_min_distance(attr, alt)
        if d_min < s:
            avoiding = True

    if avoiding:
        target_v = np.array([v[1], -v[0]])
    else:
        target_v = v_cs * (target_r - r) / mag(target_r - r)

    if np.not_equal(v, target_v).any():
        v = accelerate(v, target_v, a_max, v_cs)

    attr[AVOIDING] = avoiding

    r = r + v

    attr[RX] = r[0]
    attr[RY] = r[1]

    attr[VX] = v[0]
    attr[VY] = v[1]

    return attr


@njit(cache=True)
def accelerate(v, target_v, a_max, v_cs):
    delta_v = target_v - v
    max_a = a_max * delta_v / mag(delta_v)
    a = max_a if mag(delta_v) > mag(max_a) else delta_v
    v = v_cs * (v + a) / mag(v + a)
    return v


@njit(cache=True)
def calc_min_distance(attr, alter):
    r = np.array([attr[RX], attr[RY]])
    v = np.array([attr[VX], attr[VY]])
    alt_r = np.array([alter[RX], alter[RY]])
    alt_v = np.array([alter[VX], alter[VY]])

    r_ab = alt_r - r
    v_ab = alt_v - v

    return sqrt(mag(r_ab) ** 2 - (dot(r_ab, v_ab)) ** 2 / (mag(v_ab)) ** 2)
