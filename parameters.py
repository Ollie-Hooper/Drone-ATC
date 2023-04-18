import multiprocessing
import time

import numpy as np
from matplotlib import pyplot as plt

from drone_atc import drone
from drone_atc.config import ModelConfig, ModelParameters, Analytics, params_from_non_dim
from drone_atc.index import NoIndex, BruteForceIndex, RTree, Quadtree, BallTree
from drone_atc.scheduler import MPModelManager, Model


def sweep_parameter(vary, fixed, min_val, max_val):
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

    analytics = run_sim(configs)

    np.save(f'results/{vary}.npy', analytics)

    import pandas as pd

    writer = pd.ExcelWriter(f'results/{fixed}.xlsx', engine='xlsxwriter')

    for i in range(analytics.shape[2]):
        df = pd.DataFrame(analytics[:, :, i])
        df.to_excel(writer, sheet_name=f'{Analytics(i).name}')

    writer.close()

    print()

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
    d = {
        'd': [0.001, 0.1],
        'T': []
    }

    sweep_parameter('d', dict(Na=2000, d=0.001, T=2, Rc=10, Ra=5, A=1), 0.001, 0.1)
