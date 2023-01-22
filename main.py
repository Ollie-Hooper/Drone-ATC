import matplotlib.pyplot as plt
import numpy as np

from drone_atc.animation import Animator
from drone_atc.model import DroneATCModel

from matplotlib.animation import FuncAnimation


def main():
    N = 3
    length = 5
    fps = 10
    frames = length*fps
    model = DroneATCModel(N, fps, 1000)
    for i in range(frames):
        model.step()

    print("ANIMATING")
    anim = Animator(fps)
    anim.multiprocessing_anim(*model.get_positions_and_velocities(), 20, model.arena_size)
    #anim.create_anim(*model.get_positions_and_velocities(), 20, model.arena_size)
    anim.anims[0].save('test5.gif', fps=fps, dpi=200)


if __name__ == '__main__':
    main()
