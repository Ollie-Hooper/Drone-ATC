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


def sweep_parameter(vary, fixed, min_val, max_val, n_sims):
    index = NoIndex
    n_processes = multiprocessing.cpu_count()

    vals = np.linspace(min_val, max_val, num=50)

    configs = []

    for val in vals:
        params = params_from_non_dim(**{
            **fixed,
            vary: val,
        })
        config = ModelConfig(
            agent='drone',
            spatial_index=index,
            n_processes=n_processes,
            n_steps=2,
            params=params,
            animate=False,
        )
        configs.append(config)

    conflicts = np.empty((n_sims, len(configs)))
    calculations = np.empty((n_sims, len(configs)))
    avoiding = np.empty((n_sims, len(configs)))
    collisions = np.empty((n_sims, len(configs)))

    for i in range(n_sims):
        print(f"""
        ==================SIM {i+1}/{n_sims}==================
        """)
        analytics = run_sim(configs)
        conflicts[i, :] = np.sum(analytics[:, 1:, Analytics.CONFLICTS.value], axis=0)
        calculations[i, :] = np.sum(analytics[:, 1:, Analytics.CALCULATIONS.value], axis=0)
        avoiding[i, :] = np.sum(analytics[:, 1:, Analytics.AVOIDING.value], axis=0)
        collisions[i, :] = np.sum(analytics[:, 1:, Analytics.COLLISIONS.value], axis=0)

    pd.DataFrame(conflicts).to_csv(f'results/{vary}_conflicts.csv')
    pd.DataFrame(calculations).to_csv(f'results/{vary}_calculations.csv')
    pd.DataFrame(avoiding).to_csv(f'results/{vary}_avoiding.csv')
    pd.DataFrame(collisions).to_csv(f'results/{vary}_collisions.csv')

    # np.save(f'results/{vary}.npy', analytics)
    #
    # import pandas as pd
    #
    # writer = pd.ExcelWriter(f'results/{fixed}.xlsx', engine='xlsxwriter')
    #
    # for i in range(analytics.shape[2]):
    #     df = pd.DataFrame(analytics[:, :, i])
    #     df.to_excel(writer, sheet_name=f'{Analytics(i).name}')
    #
    # writer.close()
    #
    # print()

    # calcs = np.sum(analytics[:, 1:, Analytics.CALCULATIONS.value], axis=0)
    # conflicts = np.sum(analytics[:, 1:, Analytics.CONFLICTS.value], axis=0)
    # avoiding = np.sum(analytics[:, 1:, Analytics.AVOIDING.value], axis=0)
    # collisions = np.sum(analytics[:, 1:, Analytics.COLLISIONS.value], axis=0)
    #
    # plt.plot(vals, collisions)
    # plt.show()


def run_sim(config):
    with MPModelManager(config) as model:
        ts = time.time()
        analytics = model.go()
        te = time.time()
        print(te - ts)

    return analytics


if __name__ == '__main__':
    # d = {
    #     'd': [0.001, 0.1],
    #     'T': []
    # }

    sweep_parameter('d', dict(Na=2000, d=0.001, T=2, Rc=10, Ra=5, A=1), 0.001, 0.1, 50)
