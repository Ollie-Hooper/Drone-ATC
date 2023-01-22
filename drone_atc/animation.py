import math

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

import multiprocessing


class Animator:
    def __init__(self, fps):
        self.anims = {}
        self.positions = None
        self.ax = None
        self.scatter = None
        self.fps = fps

    def multiprocessing_anim(self, positions, velocities, safety_disc, arena_size):
        n_frames = len(positions)
        n_cores = multiprocessing.cpu_count()
        factor = math.ceil(n_frames/n_cores)

        processes = []

        for i in range(n_cores):
            start_frame = i*factor
            end_frame = (i+1)*factor
            self.create_anim(positions, velocities, safety_disc, arena_size, start_frame, end_frame)
            processes.append(multiprocessing.Process(target=self.anims[start_frame].save, args=(f'multiprocessing{i}.gif'),kwargs=dict(fps=self.fps,dpi=200)))

        for p in processes:
            p.start()

        for p in processes:
            p.join()

        print("Done!")


    def create_anim(self, positions, velocities, safety_disc, arena_size, start_frame=0, end_frame=-1):
        self.positions = positions
        self.velocities = velocities

        def update_scatter(frame_number):
            frame_number += start_frame
            points = self.positions[frame_number][:2]
            colours = ['r' if c == 1 else 'b' for c in self.positions[frame_number][2]]
            self.ax = plt.axes(xlim=(0, arena_size), ylim=(0, arena_size))

            s = ((safety_disc*self.ax.get_window_extent().width / (arena_size+1.) * 72./fig.dpi) ** 2)

            # self.ax.scatter(points[0], points[1], c=colours)
            self.ax.scatter(points[0], points[1], c=colours, s=s)

            history = self.positions[start_frame:frame_number + 1, :2, :].T

            for x, y in history:
                self.ax.plot(x, y, alpha=0.2, zorder=0)

            vel = self.velocities[frame_number][:2]

            for i, v in enumerate(vel.T):
                self.ax.arrow(points[0][i], points[1][i], v[0], v[1], width=safety_disc/5, head_length=safety_disc/2.5, length_includes_head=True, color=colours[i])

        fig = plt.figure(figsize=(7, 7))
        self.anims[start_frame] = FuncAnimation(fig, update_scatter, frames=len(positions[start_frame:end_frame])+1, interval=1)
