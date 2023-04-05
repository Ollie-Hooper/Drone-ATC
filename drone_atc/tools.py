from matplotlib import pyplot as plt
from numba import njit
from numpy import sqrt


@njit(cache=True)
def mag(x):
    return sqrt(x.dot(x))


import numpy as np
import random


def generate_poisson_disk_samples(square_size, min_distance, num_points, max_attempts=30):
    cell_size = min_distance / np.sqrt(2)
    grid_size = int(np.ceil(square_size / cell_size))
    grid = np.full((grid_size, grid_size), -1, dtype=int)

    def get_grid_coords(point):
        return (point // cell_size).astype(int)

    def is_valid_point(point):
        i, j = get_grid_coords(point)
        if i < 0 or i >= grid_size or j < 0 or j >= grid_size:
            return False

        i_min, i_max = max(0, i - 1), min(grid_size - 1, i + 1)
        j_min, j_max = max(0, j - 1), min(grid_size - 1, j + 1)

        for x in range(i_min, i_max + 1):
            for y in range(j_min, j_max + 1):
                index = grid[x, y]
                if index != -1:
                    if np.linalg.norm(points[index] - point) < min_distance:
                        return False
        return True

    def generate_random_point_around(point):
        radius = random.uniform(min_distance, 2 * min_distance)
        angle = random.uniform(0, 2 * np.pi)
        return point + np.array([radius * np.cos(angle), radius * np.sin(angle)])

    points = []
    sample_queue = []

    init_point = np.array([random.uniform(0, square_size), random.uniform(0, square_size)])
    points.append(init_point)
    grid[tuple(get_grid_coords(init_point))] = 0
    sample_queue.append(0)

    while len(points) < num_points and sample_queue:
        point_index = random.choice(sample_queue)

        for _ in range(max_attempts):
            new_point = generate_random_point_around(points[point_index])
            if is_valid_point(new_point):
                points.append(new_point)
                grid[tuple(get_grid_coords(new_point))] = len(points) - 1
                sample_queue.append(len(points) - 1)
                break
        else:
            sample_queue.remove(point_index)

    return points[:num_points]


# def generate_uniform_points_with_min_distance(square_size, min_distance, num_points):
#     cell_size = square_size / np.sqrt(num_points)
#     num_cells = int(np.ceil(square_size / cell_size))
#     cell_size = square_size/num_cells
#     total_cells = num_cells * num_cells
#
#     if min_distance > cell_size:
#         raise ValueError("Cannot satisfy the minimum distance constraint with the given number of points.")
#
#     def generate_point_in_cell(i, j):
#         x_min = np.max([i * cell_size + min_distance / 2, 0])
#         x_max = np.min([(i + 1) * cell_size - min_distance / 2, square_size])
#         y_min = np.max([j * cell_size + min_distance / 2, 0])
#         y_max = np.min([(j + 1) * cell_size - min_distance / 2, square_size])
#
#         x = np.random.uniform(x_min, x_max)
#         y = np.random.uniform(y_min, y_max)
#         return np.array([x, y])
#
#     points = []
#     cell_indices = np.arange(total_cells)
#     np.random.shuffle(cell_indices)
#
#     for idx in range(num_points):
#         cell_index = cell_indices[idx]
#         i, j = divmod(cell_index, num_cells)
#         point = generate_point_in_cell(i, j)
#         points.append(point)
#
#     return points

@njit(cache=True)
def generate_uniform_points_with_min_distance(square_size, min_distance, num_points):
    cell_size = square_size / np.sqrt(num_points)
    num_cells = int(np.ceil(square_size / cell_size))
    cell_size = square_size / num_cells
    total_cells = num_cells * num_cells

    if min_distance > cell_size:
        raise ValueError("Cannot satisfy the minimum distance constraint with the given number of points.")

    def generate_point_in_cell(i, j):
        x_min = i * cell_size + min_distance / 2
        x_max = (i + 1) * cell_size - min_distance / 2
        y_min = j * cell_size + min_distance / 2
        y_max = (j + 1) * cell_size - min_distance / 2

        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        return x, y

    rx = np.empty(num_points, dtype=np.float64)
    ry = np.empty(num_points, dtype=np.float64)
    cell_indices = np.arange(total_cells)
    np.random.shuffle(cell_indices)

    for idx in range(num_points):
        cell_index = cell_indices[idx]
        i, j = divmod(cell_index, num_cells)
        x, y = generate_point_in_cell(i, j)
        rx[idx] = x
        ry[idx] = y

    return rx, ry


def plot_points(rx, ry, min_radius, square_size):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, square_size)
    ax.set_ylim(0, square_size)

    for point in zip(rx, ry):
        circle = plt.Circle(point, min_radius / 2, color='blue', alpha=0.5)
        ax.add_artist(circle)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
