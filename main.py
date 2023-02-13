from drone_atc.agent import Drone
from drone_atc.scheduler import MPModelManager, Model


def main():
    n_processes = 10#multiprocessing.cpu_count()
    agents_per_process = 20
    n_steps = 1

    with MPModelManager(Model, n_processes, Drone, agents_per_process, n_steps) as model:
        model.go()


if __name__ == "__main__":
    main()
