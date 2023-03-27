from multiprocessing import Process

from matplotlib import pyplot as plt

from drone_atc.config import Analytics
from drone_atc.shared_mem import get_shm_array


class Analyser(Process):
    def __init__(self, model_shm):
        super().__init__()
        self.model_shm = model_shm
        self.analytics = None
        self.agent_attrs = None

    def run(self):
        self.analytics = get_shm_array(self.model_shm.analytics)
        self.agent_attrs = get_shm_array(self.model_shm.agents)
        while (self.analytics[:, -1, :] == 0).all():
            # exec_time = self.analytics[:,:,Analytics.STEP_EXECUTION_TIME.value]
            # plt.plot(exec_time)
            # plt.show()
            x = self.agent_attrs[:, 0]
            y = self.agent_attrs[:, 1]
            velocities = self.agent_attrs[:, 2:4]

            plt.scatter(x, y)
            plt.show()
