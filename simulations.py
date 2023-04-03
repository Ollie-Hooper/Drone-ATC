import multiprocessing
import time

import numpy as np
from matplotlib import pyplot as plt

from drone_atc import drone
from drone_atc.config import ModelConfig, ModelParameters, Analytics
from drone_atc.index import NoIndex, BruteForceIndex, RTree, Quadtree, BallTree
from drone_atc.scheduler import MPModelManager, Model


def run_sim(n_processes=multiprocessing.cpu_count(), n_agents=10000, index=NoIndex):
    params = ModelParameters(
        n_agents=n_agents,
        s=0.1,
        a_max=0.1,
        v_cs=0.1,
        r_com=0.1,
    )
    config = ModelConfig(
        agent='drone',
        spatial_index=index,
        n_processes=n_processes,
        n_steps=3,
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

    step = 0

    for n_processes in p_range:
        analytics = run_sim(n_processes)
        exec_time = analytics[:, step, Analytics.STEP_EXECUTION_TIME.value]
        read = analytics[:, step, Analytics.READ_TIME.value]
        write = analytics[:, step, Analytics.WRITE_TIME.value]

        min_t = min(exec_time)
        mean_t = exec_time.mean()
        max_t = max(exec_time)

        mins.append(min_t)
        means.append(mean_t)
        maxs.append(max_t)

        reads.append(read.mean())
        writes.append(write.mean())

        exec_times.append(exec_time)

    plt.bar(p_range, means)
    plt.xlabel('# of processes')
    plt.ylabel('Execution time of one step (s)')
    plt.show()

    # perf per process

    process_eff = means[0] / (np.array(p_range) * means)
    plt.bar(p_range, process_eff)
    plt.xlabel('# of processes')
    plt.ylabel('Process efficiency (assuming single process as 100%)')
    plt.show()

    # stacked bar
    means = np.array(means)
    reads = np.array(reads)
    writes = np.array(writes)
    misc = means - (reads + writes)

    plt.bar(p_range, reads)
    plt.bar(p_range, writes, bottom=reads)
    plt.bar(p_range, misc, bottom=reads + writes)
    plt.legend(['Read', 'Write', 'Misc.'])
    plt.xlabel('# of processes')
    plt.ylabel('Execution time of one step (s)')
    plt.show()


# 4000 agents: spatial index vs step execution time

def index_plots():
    x = ['None', 'Brute Force', 'RTree', 'Quadtree', 'BallTree']

    exec_times = []
    update_times = []
    for index in [NoIndex, BruteForceIndex, RTree, Quadtree, BallTree]:
        analytics = run_sim(index=index)
        exec_times.append(analytics[0, :, Analytics.STEP_EXECUTION_TIME.value])
        update_times.append(analytics[0, :, Analytics.INDEX_UPDATE_TIME.value])

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
    agent_range = np.logspace(2, 4, 20, dtype=np.int32)# np.linspace(100, 10000, 20, dtype=np.int32)
    exec_times = []

    for n in agent_range:
        analytics = run_sim(n_agents=n, index=index)
        exec_times.append(analytics[0, :, Analytics.STEP_EXECUTION_TIME.value])

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
    x = ['RTree', 'Quadtree', 'BallTree']

    for index in [RTree, Quadtree, BallTree]:
        agent_range, means = agents(index=index)
        plt.plot(agent_range, means)
    plt.xlabel('# of agents')
    plt.ylabel('Execution time (s)')
    plt.legend(x)
    plt.show()


if __name__ == '__main__':
    processes()
