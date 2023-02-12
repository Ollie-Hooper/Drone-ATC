from dataclasses import dataclass
from typing import List


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
    @staticmethod
    def step(attr: DroneAttributes, alters: List[DroneAttributes], *args, **kwargs) -> DroneAttributes:
        return attr

    @staticmethod
    def accelerate(attr):
        return attr

    @staticmethod
    def calc_min_distance(attr, alter):
        return
