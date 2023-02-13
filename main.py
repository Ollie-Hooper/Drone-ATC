import multiprocessing
import time

from drone_atc.agent import Drone
from drone_atc.scheduler import MPModelManager, Model


def main():
    n_processes = multiprocessing.cpu_count()
    agents_per_process = 10
    n_steps = 2

    with MPModelManager(Model, n_processes, Drone, agents_per_process, n_steps) as model:
        ts = time.time()
        analytics = model.go()
        te = time.time()
        print(te - ts)
        print(analytics)


if __name__ == "__main__":
    main()
