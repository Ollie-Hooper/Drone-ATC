import multiprocessing
import time

from drone_atc.config import ModelConfig, ModelParameters, params_from_non_dim, Analytics
from drone_atc.index import NoIndex, BallTree, Grid, BruteForceIndex
from drone_atc.scheduler import MPModelManager, Model


def main():
    n_processes = 8#multiprocessing.cpu_count()
    n_steps = 3
    animate = False

    n_agents = 1000

    params = params_from_non_dim(n_agents, d=0.01, T=5, Rc=10, Ra=10, A=0.01)

    config = ModelConfig(
        agent='drone',
        spatial_index=NoIndex,
        n_processes=n_processes,
        n_steps=n_steps,
        params=params,
        animate=animate,
    )

    with MPModelManager(config) as model:
        ts = time.time()
        analytics = model.go()
        exec_time = analytics[:, 2, Analytics.RANGE_SEARCH_MEAN.value]
        print(exec_time)
        te = time.time()
        print(te - ts)


if __name__ == "__main__":
    main()
