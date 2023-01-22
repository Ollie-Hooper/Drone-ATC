from dataclasses import dataclass

import numpy as np

from numpy import sqrt, dot
from mesa import Agent


@dataclass
class DroneConfig:
    cruising_speed = 40


mag = lambda x: np.sqrt(x.dot(x))


class Drone(Agent):
    def __init__(self, unique_id, model, arena_size, config=DroneConfig()):
        super().__init__(unique_id, model)
        self.target = None
        self.v = None
        self.target_v = None
        self.config = config
        self.r = np.random.rand(3)*arena_size
        self.set_initial_velocity()
        self.pos_history = []
        self.vel_history = []
        self.s = 20
        self.dt = model.dt
        self.max_a = 0.8
        self.range = 200

    def set_initial_velocity(self):
        v = 1-2*np.random.rand(3)
        v[2] = 0
        self.v = self.config.cruising_speed * v/np.sqrt(v[0]**2+v[1]**2)
        self.target_v = self.v
        self.target = self.v

    def step(self):
        if np.not_equal(self.v, self.target_v).all():
            self.accelerate()
        avoiding = False
        for alter in self.model.schedule.agents:
            if self != alter:
                # if self.calc_collision(alter):
                #     pass
                if self.calc_expect_collision(alter):
                    self.set_new_velocity(np.array([self.v[1],-self.v[0],0]))
                    #self.set_new_velocity(np.array([self.config.cruising_speed,0,0]))
                    avoiding = True
        if not avoiding and np.not_equal(self.v, self.target).any():
            self.set_new_velocity(self.target)
        self.r = self.r + self.dt*self.v
        self.pos_history.append(np.array([*self.r[:2], 1 if avoiding else 0]))
        self.vel_history.append(np.array(self.v[:2]))
        # print(f"{self.unique_id}: {self.r}")

    def accelerate(self):
        target_a = self.target_v - self.v
        if m := mag(target_a) > self.max_a:
            a = self.max_a*target_a / m
        else:
            a = target_a
        v = self.v + a*self.dt
        self.v = self.config.cruising_speed*v/mag(v)

    def set_new_velocity(self, v):
        self.target_v = v
        self.accelerate()

    def calc_collision(self, alter):
        if mag(alter.r - self.r):
            return True

    def calc_expect_collision(self, alter):
        r_ab = (alter.r-self.r)[:2]
        if mag(r_ab) > self.range:
            return False
        v_ab = (alter.v-self.v)[:2]
        d_min = sqrt(mag(r_ab)**2-(dot(r_ab,v_ab))**2/(mag(v_ab))**2)
        # d_min = mag(r_ab)*(mag(np.cross(r_ab,v_ab))/(mag(v_ab)*mag(v_ab)))
        if d_min < self.s:
            return True
