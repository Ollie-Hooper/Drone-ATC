import multiprocessing
import time

import numpy as np

from drone_atc.config import ModelConfig, ModelParameters, params_from_non_dim
from drone_atc.analytics_config import Analytics
from drone_atc.index import NoIndex, BallTree, Grid, BruteForceIndex, Quadtree, RTree
from drone_atc.scheduler import MPModelManager, Model
from drone_atc.tools import dump_analytics


def main():
    n_processes = multiprocessing.cpu_count()
    n_steps = 3
    animate = False

    n_agents = 100000

    params = params_from_non_dim(n_agents, d=0.01, T=4, Rc=74, Ra=10, A=0.01)

    config = ModelConfig(
        agent='drone',
        spatial_index=Grid,
        n_processes=n_processes,
        n_steps=n_steps,
        params=params,
        animate=animate,
        index_params={},
    )

    with MPModelManager(config) as model:
        ts = time.time()
        analytics = model.go()
        exec_time = analytics[:, :, Analytics.CALCULATIONS.value]
        print(np.sum(exec_time,axis=0))
        te = time.time()
        print(te - ts)

    # dump_analytics(analytics, 'compiled')


if __name__ == "__main__":
    main()
