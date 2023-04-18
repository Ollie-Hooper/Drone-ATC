import multiprocessing
import time

import numpy as np
from matplotlib import pyplot as plt

from drone_atc import drone
from drone_atc.config import ModelConfig, ModelParameters, Analytics, params_from_non_dim
from drone_atc.index import NoIndex, BruteForceIndex, RTree, Quadtree, BallTree, Grid
from drone_atc.scheduler import MPModelManager, Model


def run_sim(n_steps=3, n_processes=multiprocessing.cpu_count(), n_agents=5000, index=NoIndex):
    params = params_from_non_dim(n_agents, d=0.01, T=5, Rc=10, Ra=5, A=0.01)

    config = ModelConfig(
        agent='drone',
        spatial_index=index,
        n_processes=n_processes,
        n_steps=n_steps,
        params=params,
        animate=False,
    )

    with MPModelManager(config) as model:
        ts = time.time()
        analytics = model.go()
        te = time.time()
        print(te - ts)

    return analytics


# 2000 agents: step execution time vs n_processes
def processes():
    mins = []
    means = []
    maxs = []
    exec_times = []
    reads = []
    writes = []

    p_range = list(range(1, multiprocessing.cpu_count() + 1))

    step = 1

    for n_processes in p_range:
        analytics = run_sim(n_processes=n_processes, n_steps=2, n_agents=5000)
        # exec_time = analytics[step, :, Analytics.STEP_EXECUTION_TIME.value]
        # read = analytics[step, :, Analytics.READ_TIME.value]
        # write = analytics[step, :, Analytics.WRITE_TIME.value]

        np.save(f'results/{n_processes}.npy', analytics)

        import pandas as pd

        writer = pd.ExcelWriter(f'results/{n_processes}.xlsx', engine='xlsxwriter')

        for i in range(analytics.shape[2]):
            df = pd.DataFrame(analytics[:, :, i])
            df.to_excel(writer, sheet_name=f'{Analytics(i).name}')

        writer.close()

        # min_t = min(exec_time)
        # mean_t = exec_time.mean()
        # max_t = max(exec_time)
        #
        # mins.append(min_t)
        # means.append(mean_t)
        # maxs.append(max_t)
        #
        # reads.append(read.mean())
        # writes.append(write.mean())
        #
        # exec_times.append(exec_time)

    # plt.bar(p_range, maxs)
    # plt.xlabel('# of processes')
    # plt.ylabel('Execution time of one step (s)')
    # plt.show()
    #
    # # perf per process
    #
    # process_eff = means[0] / (np.array(p_range) * means)
    # plt.bar(p_range, process_eff)
    # plt.xlabel('# of processes')
    # plt.ylabel('Process efficiency (assuming single process as 100%)')
    # plt.show()
    #
    # # stacked bar
    # means = np.array(means)
    # reads = np.array(reads)
    # writes = np.array(writes)
    # misc = means - (reads + writes)
    #
    # plt.bar(p_range, reads)
    # plt.bar(p_range, writes, bottom=reads)
    # plt.bar(p_range, misc, bottom=reads + writes)
    # plt.legend(['Read', 'Write', 'Misc.'])
    # plt.xlabel('# of processes')
    # plt.ylabel('Execution time of one step (s)')
    # plt.show()


# 4000 agents: spatial index vs step execution time

def index_plots():
    x = ['None', 'Brute Force', 'RTree', 'Quadtree', 'BallTree', 'Grid']

    exec_times = []
    update_times = []
    for index in [NoIndex, BruteForceIndex, RTree, Quadtree, BallTree, Grid]:
        analytics = run_sim(index=index, n_agents=2000)
        exec_times.append(analytics[:, 1, Analytics.STEP_EXECUTION_TIME.value])
        update_times.append(analytics[:, 1, Analytics.INDEX_UPDATE_TIME.value])

    exec_times = np.array(exec_times)
    update_times = np.array(update_times)

    mean_exec = exec_times.mean(axis=1)
    mean_update = update_times.mean(axis=1)
    not_update = mean_exec - mean_update

    plt.bar(x, mean_exec)
    # plt.bar(x, mean_update, bottom=not_update)
    plt.xlabel('Spatial index used')
    plt.ylabel('Time (s)')
    # plt.legend(['Execution', 'Updating index'])
    plt.show()


# n_agents vs step execution time

def agents(index=NoIndex):
    agent_range = np.linspace(100, 10000, 10, dtype=np.int32)
    exec_times = []

    step = 2

    for n in agent_range:
        analytics = run_sim(n_agents=n, index=index)
        exec_times.append(analytics[:, step, Analytics.STEP_EXECUTION_TIME.value])

    means = np.array(exec_times).mean(axis=1)

    return agent_range, means


def plot_agents():
    agent_range, means = agents()

    plt.plot(agent_range, means)
    plt.xlabel('# of agents')
    plt.ylabel('Execution time (s)')
    plt.show()

    plt.loglog(agent_range, means)
    plt.xlabel('# of agents')
    plt.ylabel('Execution time (s)')
    plt.show()


def index_n_agents():
    x = ['BallTree', 'Grid']

    for index in [BallTree, Grid]:
        agent_range, means = agents(index=index)
        plt.plot(agent_range, means)
    plt.xlabel('# of agents')
    plt.ylabel('Execution time (s)')
    plt.legend(x)
    plt.show()


if __name__ == '__main__':
    processes()
