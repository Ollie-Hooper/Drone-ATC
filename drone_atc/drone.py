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
CONFLICTS = 7
CALCULATIONS = 8
COLLISIONS = 9

attributes = ['RX', 'RY', 'VX', 'VY', 'T_VX', 'T_VY', 'AVOIDING', 'CONFLICTS', 'CALCULATIONS', 'COLLISIONS']


# for i, k in enumerate(attributes):
#     globals()[k] = i


def step(uid, model_attrs, spatial_index, attrs: np.array) -> np.array:
    s = model_attrs.s
    a_max = model_attrs.a_max
    v_cs = model_attrs.v_cs
    r_com = model_attrs.r_com
    tc_max = model_attrs.tc_max
    l = model_attrs.l
    A = model_attrs.A
    in_range = spatial_index.agents_in_range(uid, r_com)
    in_range = np.delete(in_range, in_range == uid)

    return _step(uid, s, a_max, v_cs, in_range, tc_max, l, A, attrs)


@njit(cache=True)
def _step(uid, s, a_max, v_cs, in_range, tc_max, l, A, attrs):
    attr = attrs[uid]
    r = np.array([attr[RX], attr[RY]])
    v = np.array([attr[VX], attr[VY]])
    target_r = np.array([attr[T_RX], attr[T_RY]])
    # target_r = r + 1e6 * v
    conflicts = 0
    avoiding = False
    calculations = 0
    collisions = 0

    a = np.zeros(2)

    traj = (target_r - r) / mag(target_r - r)

    vt = v_cs * traj

    dt = 1
    a += (vt - v) / dt

    for alt in attrs[in_range]:
        calculations += 1
        mag_r, d_min, tc = calc_min_distance_and_tc(attr, alt)
        if mag_r < s:
            collisions += 1
        if d_min < s and tc >= 0:
            conflicts += 1
            if tc <= tc_max:
                avoiding = True

                rot_v = np.array([v[1], -v[0]])
                a += ((1 / A*(tc/dt)) * (rot_v - v))/dt
                #print(rot_v-v)
                # a += (rot_v - v)/dt

    # if np.not_equal(v, target_v).any():
    #     v = accelerate(v, target_v, a_max, v_cs)

    attr[AVOIDING] = avoiding
    attr[CONFLICTS] = conflicts
    attr[CALCULATIONS] = calculations
    attr[COLLISIONS] = collisions

    mag_a = mag(a)
    if mag_a > a_max:
        a = a_max * a / mag_a

    # if avoiding:
    #     print(a)

    v = (v + a) / dt

    mag_v = mag(v)
    v = v_cs * v / mag_v

    r = (r + v) / dt

    if (r < 0).any() or (r > l).any():
        r = respawn(l)

    attr[RX] = r[0]
    attr[RY] = r[1]

    attr[VX] = v[0]
    attr[VY] = v[1]

    return attr


@njit(cache=True)
def respawn(l):
    which_end = np.array([0, l])
    np.random.shuffle(which_end)
    end = which_end[0]
    r = np.array([end, l*np.random.rand()])
    np.random.shuffle(r)
    return r

@njit(cache=True)
def accelerate(v, target_v, a_max, v_cs):
    delta_v = target_v - v
    max_a = a_max * delta_v / mag(delta_v)
    a = max_a if mag(delta_v) > mag(max_a) else delta_v
    v = v_cs * (v + a) / mag(v + a)
    return v


@njit(cache=True)
def calc_min_distance_and_tc(attr, alter):
    r = np.array([attr[RX], attr[RY]])
    v = np.array([attr[VX], attr[VY]])
    alt_r = np.array([alter[RX], alter[RY]])
    alt_v = np.array([alter[VX], alter[VY]])

    r_ab = alt_r - r
    v_ab = alt_v - v

    mag_r = mag(r_ab)
    mag_v = mag(v_ab)
    dot_r_v = dot(r_ab, v_ab)

    d_min = sqrt(mag_r ** 2 - (dot_r_v ** 2) / (mag_v ** 2))

    tc = -dot_r_v / (mag_v ** 2)

    return mag_r, d_min, tc
