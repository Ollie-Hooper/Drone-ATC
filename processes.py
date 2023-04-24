import multiprocessing
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from drone_atc import drone
from drone_atc.config import ModelConfig, ModelParameters, params_from_non_dim
from drone_atc.analytics_config import Analytics
from drone_atc.index import NoIndex, BruteForceIndex, RTree, Quadtree, BallTree
from drone_atc.scheduler import MPModelManager, Model


def bulk_sim(n_processes, n_sims):
    params = params_from_non_dim(Na=5000, d=0.01, T=4, Rc=20, Ra=10, A=0.01)
    config = ModelConfig(
        agent='drone',
        spatial_index=NoIndex,
        n_processes=n_processes,
        n_steps=3,
        params=params,
        animate=False,
    )

    ses = np.empty((n_sims, n_processes, 3))
    read = np.empty((n_sims, n_processes, 3))
    write = np.empty((n_sims, n_processes, 3))

    step = 2

    for i in range(n_sims):
        ts = time.time()
        print(f"""
        ==================SIM {i+1}/{n_sims}==================
        """)
        for j in range(n_processes):
            config.n_processes = j+1
            analytics = run_sim(config)
            ses[i, j, 0] = np.mean(analytics[:, step, Analytics.STEP_EXECUTION_TIME.value], axis=0)
            read[i, j, 0] = np.mean(analytics[:, step, Analytics.READ_TIME.value], axis=0)
            write[i, j, 0] = np.mean(analytics[:, step, Analytics.WRITE_TIME.value], axis=0)

            ses[i, j, 1] = np.min(analytics[:, step, Analytics.STEP_EXECUTION_TIME.value], axis=0)
            read[i, j, 1] = np.min(analytics[:, step, Analytics.READ_TIME.value], axis=0)
            write[i, j, 1] = np.min(analytics[:, step, Analytics.WRITE_TIME.value], axis=0)

            ses[i, j, 2] = np.max(analytics[:, step, Analytics.STEP_EXECUTION_TIME.value], axis=0)
            read[i, j, 2] = np.max(analytics[:, step, Analytics.READ_TIME.value], axis=0)
            write[i, j, 2] = np.max(analytics[:, step, Analytics.WRITE_TIME.value], axis=0)
        te = time.time()
        print(f'Took {te-ts}')

        np.save('results/ses.npy', ses)
        np.save('results/read.npy', read)
        np.save('results/write.npy', write)


def run_sim(config):
    with MPModelManager(config) as model:
        ts = time.time()
        analytics = model.go()
        te = time.time()
        print(te - ts)

    return analytics


if __name__ == '__main__':
    bulk_sim(multiprocessing.cpu_count(), 50)
