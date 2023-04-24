import multiprocessing
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from drone_atc import drone
from drone_atc.config import ModelConfig, ModelParameters, params_from_non_dim
from drone_atc.analytics_config import Analytics
from drone_atc.index import NoIndex, BruteForceIndex, RTree, Quadtree, BallTree, Grid
from drone_atc.scheduler import MPModelManager, Model


def bulk_agents(agent_list, n_sims):
    calcs = np.empty((n_sims, len(agent_list)))

    step = 2

    for i in range(n_sims):
        ts = time.time()
        print(f"""
        ==================SIM {i+1}/{n_sims}==================
        """)
        for j, n_agents in enumerate(agent_list):
            print(f"n_agents: {n_agents}")
            params = params_from_non_dim(Na=n_agents, d=0.01, T=4, Rc=20, Ra=10, A=0.01)
            config = ModelConfig(
                agent='drone',
                spatial_index=Grid,
                n_processes=multiprocessing.cpu_count(),
                n_steps=3,
                params=params,
                animate=False,
            )
            analytics = run_sim(config)
            calcs[i, j] = np.sum(analytics[:, step, Analytics.CALCULATIONS.value], axis=0)
        te = time.time()
        print(f'Took {te-ts}')

        np.save('results/rcom_calcs.npy', calcs)

        pd.DataFrame(calcs).to_csv('results/rcom_calcs.csv')


def run_sim(config):
    with MPModelManager(config) as model:
        ts = time.time()
        analytics = model.go()
        te = time.time()
        print(te - ts)

    return analytics


if __name__ == '__main__':
    agent_list = np.linspace(100, 10000, num=20, dtype=np.int32)
    bulk_agents(agent_list, 50)
