from typing import Any

import mesa
import matplotlib.pyplot as plt
import numpy as np

from mesa import Model
from mesa.time import RandomActivation

from drone_atc.drone import Drone


class DroneATCModel(Model):
    def __init__(self, N, fps, arena_size=1000, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.num_agents = N
        self.arena_size = arena_size
        self.schedule = RandomActivation(self)
        self.dt = 1/fps
        for i in range(self.num_agents):
            a = Drone(i, self, arena_size)
            self.schedule.add(a)

    def step(self) -> None:
        self.schedule.step()

    def get_positions_and_velocities(self):
        positions = []
        velocities = []
        for agent in self.schedule.agent_buffer():
            positions.append(np.array(agent.pos_history).T[:3])
            velocities.append(np.array(agent.vel_history).T[:2])
        return np.array(positions).T, np.array(velocities).T
