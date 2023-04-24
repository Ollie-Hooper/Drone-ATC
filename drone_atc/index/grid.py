import numpy as np
from numba import njit
from numba.typed import List

from drone_atc.tools import mag


@njit(cache=True)
def update(grid, secondary_index, initialised, num_rows_cols, gcs, agents):
    if not initialised:
        grid = List(
            [List([List([1, ]) for j in range(num_rows_cols)]) for i in range(num_rows_cols)])
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                grid[i][j].remove(1)
    oid = 0
    for x, y in agents:
        update_agent(grid, secondary_index, initialised, num_rows_cols, gcs, oid, x, y)
        oid += 1
    initialised = True
    return grid, secondary_index, initialised


@njit(cache=True)
def update_agent(grid, secondary_index, initialised, num_rows_cols, gcs, oid, x, y):
    row, col = get_row_col(num_rows_cols, gcs, x, y)
    if not initialised:
        secondary_index[oid] = np.array((x, y, row, col))
        grid[row][col].append(oid)
    _, _, old_row, old_col = secondary_index[oid]
    old_row, old_col = int(old_row), int(old_col)

    if row != old_row or col != old_col:
        grid[old_row][old_col].remove(oid)
        grid[row][col].append(oid)
    secondary_index[oid] = np.array((x, y, row, col))


@njit(cache=True)
def get_row_col(num_rows_cols, gcs, x, y):
    col = min(num_rows_cols - 1, max(0, int(np.floor(x / gcs))))
    row = min(num_rows_cols - 1, max(0, int(np.floor(y / gcs))))
    return row, col


@njit(cache=True)
def agents_in_range(grid, secondary_index, num_rows_cols, gcs, agent: int, r: float) -> np.ndarray:
    cx, cy = secondary_index[agent][:2]
    xqmin, xqmax = cx - r, cx + r
    yqmin, yqmax = cy - r, cy + r

    min_row, min_col = get_row_col(num_rows_cols, gcs, xqmin, yqmin)
    max_row, max_col = get_row_col(num_rows_cols, gcs, xqmax, yqmax)

    in_range = []

    for row in range(min_row, max_row + 1):
        for col in range(min_col, max_col + 1):
            for alt in grid[row][col]:
                r_a = secondary_index[agent][:2]
                r_b = secondary_index[alt][:2]
                r_ab = r_b - r_a
                if mag(r_ab) <= r:
                    in_range.append(alt)
            # in_range.extend(list(grid[row][col]))

    # cells = get_circle_cells(r, cx, cy, gcs, num_rows_cols)

    # cells = np.unique(np.array(cells),axis=0)

    # in_range = []
    # for row, col in cells:
    #     in_range.extend(list(grid[row][col]))

    return np.array(in_range)


@njit(cache=True)
def get_circle_cells(r, center_x, center_y, cell_width, grid_width):
    r = np.ceil(r / cell_width)
    center_x, center_y = get_row_col(grid_width, cell_width, center_x, center_y)

    cells = List([(center_x, center_y),])
    x = 0
    y = r
    d = 3 - 2 * r

    cells = draw(cells, center_x, center_y, x, y)

    while y >= x:
        x += 1

        if d > 0:
            y -= 1
            d = d + 4 * (x - y) + 10
        else:
            d = d + 4 * x + 6

        cells = draw(cells, center_x, center_y, x, y)

    cells = cells[1:]

    cells = [(x, y) for (x, y) in cells if
             0 <= x < grid_width and 0 <= y < grid_width]
    return cells


@njit(cache=True)
def draw(cells, cx, cy, x, y):
    cx, cy, x, y = int(cx), int(cy), int(x), int(y)
    pairs = [((cx - x, cy + y), (cx + x, cy + y)), ((cx - y, cy + x), (cx + y, cy + x)),
             ((cx - x, cy - y), (cx + x, cy - y)), ((cx - y, cy - x), (cx + y, cy - x))]

    for pt1, pt2 in pairs:
        y = pt1[1]
        min_x = pt1[0]
        max_x = pt2[0]

        for i in range(min_x, max_x + 1):
            if (i, y) not in cells:
                cells.append((i, y))

    return cells
