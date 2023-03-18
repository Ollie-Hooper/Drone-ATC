import multiprocessing
import time

from drone_atc.agent import Drone
from drone_atc.config import ModelConfig, ModelParameters
from drone_atc.index import NoIndex
from drone_atc.scheduler import MPModelManager, Model


def main():
    n_processes = multiprocessing.cpu_count()
    agents_per_process = 100
    n_steps = 5

    params = ModelParameters(
        n_agents=agents_per_process * (n_processes - 1),
        s=0.1,
        a_max=0.1,
        v_cs=0.1,
        r_com=0.1,
    )
    config = ModelConfig(
        agent=Drone,
        spatial_index=NoIndex,
        n_processes=n_processes,
        n_steps=n_steps,
        params=params,
    )

    with MPModelManager(config) as model:
        ts = time.time()
        analytics = model.go()
        te = time.time()
        print(te - ts)
        print(analytics)


if __name__ == "__main__":
    main()
