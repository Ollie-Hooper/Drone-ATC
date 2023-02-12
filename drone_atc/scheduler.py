from multiprocessing import Process, Barrier


class MPScheduler(Process):
    def __init__(self, barrier: Barrier, attr_buffer):
        super().__init__()
        self.barrier = barrier
        self.attr_buffer = attr_buffer

    def update_agents(self, additions, deletions, modifications):
        pass

    def step(self):
        self.read()
        self.barrier.wait()
        self.write()
        self.barrier.wait()

    def read(self):
        print(self.attr_buffer)

    def write(self):
        pass
