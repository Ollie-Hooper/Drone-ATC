import multiprocessing
import time

from drone_atc.config import ModelConfig, ModelParameters, params_from_non_dim
from drone_atc.index import NoIndex, BallTree
from drone_atc.scheduler import MPModelManager, Model


def main():
    n_processes = multiprocessing.cpu_count()
    n_steps = 1000
    animate = True

    n_agents = 500

    params = params_from_non_dim(n_agents, d=0.05, T=5, Rc=10, Ra=10, A=0.01)

    config = ModelConfig(
        agent='drone',
        spatial_index=BallTree,
        n_processes=n_processes,
        n_steps=n_steps,
        params=params,
        animate=animate,
    )

    with MPModelManager(config) as model:
        ts = time.time()
        analytics = model.go()
        te = time.time()
        print(te - ts)


if __name__ == "__main__":
    main()
