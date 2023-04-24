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


def bulk_agents(name, spa_ind, agent_list, n_sims):
    analytics_list = [Analytics.STEP_EXECUTION_TIME, Analytics.READ_TIME, Analytics.WRITE_TIME,
                      Analytics.INDEX_UPDATE_TIME,
                      Analytics.CALCULATIONS, Analytics.RANGE_SEARCH_MAX, Analytics.RANGE_SEARCH_MEAN,
                      Analytics.RANGE_SEARCH_MIN]

    anal_d = {
        anal.name: np.empty((n_sims, len(agent_list))) for anal in analytics_list
    }

    step = 2

    for i in range(n_sims):
        ts = time.time()
        print(f"""
        =================={name} SIM {i + 1}/{n_sims}==================
        """)
        for j, n_agents in enumerate(agent_list):
            print(f"n_agents: {n_agents}")
            params = params_from_non_dim(Na=n_agents, d=0.01, T=4, Rc=20, Ra=10, A=0.01)
            config = ModelConfig(
                agent='drone',
                spatial_index=spa_ind,
                n_processes=multiprocessing.cpu_count(),
                n_steps=3,
                params=params,
                animate=False,
            )
            analytics = run_sim(config)

            for anal in analytics_list:
                anal_d[anal.name][i, j] = np.mean(analytics[:, step, anal.value], axis=0)

            anal_d[Analytics.CALCULATIONS.name][i, j] *= multiprocessing.cpu_count()
        te = time.time()
        print(f'Took {te - ts}')

        meanmean = np.empty((len(analytics_list), len(agent_list)))

        index = name
        for i, anal in enumerate(analytics_list):
            np.save(f'results/{index}_{anal.name}.npy', anal_d[anal.name])
            pd.DataFrame(anal_d[anal.name]).to_csv(f'results/{index}_{anal.name}.csv')

            meanmean[i, :] = np.mean(anal_d[anal.name][:i+1], axis=0)

        np.save(f'results/{index}.npy', meanmean)
        pd.DataFrame(meanmean).to_csv(f'results/{index}.csv')


def run_sim(config):
    with MPModelManager(config) as model:
        ts = time.time()
        analytics = model.go()
        te = time.time()
        print(te - ts)

    return analytics


if __name__ == '__main__':
    agent_list = np.linspace(100, 10000, num=20, dtype=np.int32)
    # for name, index in zip(['BruteForceIndex', 'BallTree', 'RTree', 'Quadtree', 'Grid'],
    #                        [BruteForceIndex, BallTree, RTree, Quadtree, Grid]):
    bulk_agents('Grid2', Grid, agent_list, 20)
