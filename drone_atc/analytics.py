from multiprocessing import Process

import matplotlib;
import numpy as np

from drone_atc import drone
from drone_atc.tools import mag

from matplotlib import pyplot as plt, animation

from drone_atc.config import Analytics
from drone_atc.shared_mem import get_shm_array


class Analyser(Process):
    def __init__(self, config, model_shm, read_barrier, write_barrier):
        super().__init__()
        self.config = config
        self.model_shm = model_shm
        self.read_barrier = read_barrier
        self.write_barrier = write_barrier
        self.analytics = None
        self.global_agent_attrs = None
        self.agent_attrs = None

    def run(self):
        self.analytics = get_shm_array(self.model_shm.analytics)
        self.global_agent_attrs = get_shm_array(self.model_shm.agents)
        self.agent_attrs = self.global_agent_attrs.copy()

        l = self.config.params.l
        safety_disc_rad = self.config.params.s

        fig = plt.figure(figsize=(10, 10))
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_xlim(0, l)
        ax1.set_ylim(0, l)

        s = ((safety_disc_rad * ax1.get_window_extent().width / (l + 1.) * 72. / fig.dpi) ** 2)

        def animate(i):
            self.agent_attrs[:] = self.global_agent_attrs[:]
            self.read_barrier.wait()

            x = self.agent_attrs[:, drone.RX]
            y = self.agent_attrs[:, drone.RY]
            v = self.agent_attrs[:, drone.VX:drone.VY + 1]

            avoiding = self.agent_attrs[:, drone.AVOIDING]
            collisions = self.agent_attrs[:, drone.COLLISIONS]

            colour = np.empty(len(x), dtype=str)

            for i, (a, c) in enumerate(zip(avoiding, collisions)):
                if c:
                    colour[i] = 'r'
                    continue
                elif a:
                    colour[i] = 'y'
                    continue
                else:
                    colour[i] = 'b'

            ax1.clear()

            ax1.scatter(x, y, c=colour, s=s)

            for i, d in enumerate(v):
                d = 2 * safety_disc_rad * d / mag(d)
                ax1.arrow(x[i], y[i], d[0], d[1], width=safety_disc_rad / 5, head_length=safety_disc_rad / 2.5,
                          length_includes_head=True, color=colour[i])

            ax1.set_xlim(0, self.config.params.l)
            ax1.set_ylim(0, self.config.params.l)
            self.write_barrier.wait()

        matplotlib.use("TkAgg")
        ani = animation.FuncAnimation(fig, animate, interval=1)
        plt.show()
