import argparse
import numpy as np
from numba import njit
from .dataset import sudoku_dataset


@njit(fastmath=True, cache=True)
def calc_q(sudoku):
    """Updates unary penalties (Q)."""
    height, width = sudoku.shape
    n_labels = 9
    Q = np.zeros((height, width, n_labels), dtype=np.int32)

    for i in range(height):
        for j in range(width):
            if sudoku[i, j] != 0:
                Q[i, j, sudoku[i, j] - 1] = 1
            else:
                Q[i, j, :] = 1
    return Q


@njit(fastmath=True, cache=True)
def calc_g(sudoku, neighbours_array):
    """Updates binary penalties (G)."""
    height, width = sudoku.shape
    n_labels = 9
    n_neighbours = 20
    G = np.zeros((height, width, n_neighbours, n_labels, n_labels), dtype=np.int32)

    for i in range(height):
        for j in range(width):
            neighbours = neighbours_array[i, j, :, :]
            sudoku_value = sudoku[i, j] - 1
            for n, [n_i, n_j] in enumerate(neighbours):
                neighbour_value = sudoku[n_i, n_j] - 1
                if sudoku[i, j] != 0:
                    if sudoku[n_i, n_j] != 0 and sudoku_value != neighbour_value:
                        G[i, j, n, sudoku_value, neighbour_value] = 1
                    else:
                        G[i, j, n, sudoku[i, j] - 1, :] = 1
                        G[i, j, n, sudoku[i, j] - 1, sudoku[i, j] - 1] = 0
                else:
                    if sudoku[n_i, n_j] != 0:
                        G[i, j, n, :, neighbour_value] = 1
                        G[i, j, n, neighbour_value, neighbour_value] = 0
                    else:
                        G[i, j, n, :, :] = 1
    return G


@njit(fastmath=True, cache=True)
def check_solution(Q_prev, G_prev, neighbours_array):
    """Checks if there is a solution based on Q and G using arc consistency."""
    Q_next = np.copy(Q_prev)
    G_next = np.copy(G_prev)
    height, width, n_neighbours, n_labels, _ = G_next.shape
    while True:
        for i in range(height):
            for j in range(width):
                neighbours = neighbours_array[i, j, :, :]
                for k in range(n_labels):
                    for n, [n_i, n_j] in enumerate(neighbours):
                        binary = (G_prev[i, j, n, k, :] * Q_prev[n_i, n_j, :]).any()
                        if binary == 0:
                            Q_next[i, j, k] = 0
                            G_next[i, j, :, k, :] = 0
                            break
                        else:
                            G_next[i, j, n, k, :] = (
                                G_prev[i, j, n, k, :] * Q_next[n_i, n_j, :]
                            )

        if (G_next == G_prev).all() and (Q_next == Q_prev).all():
            if np.all(G_next == 0):
                return False
            else:
                return True
        else:
            Q_prev = np.copy(Q_next)
            G_prev = np.copy(G_next)


@njit(fastmath=True, cache=True)
def solve_recursive(sudoku, neighbours_array):
    """Recursive backtracking function powered by arc-consistency pruning."""
    height, width = sudoku.shape

    best_i = -1
    best_j = -1
    found = False

    for i in range(height):
        for j in range(width):
            if sudoku[i, j] == 0:
                best_i = i
                best_j = j
                found = True
                break
        if found:
            break

    if not found:
        return True

    for k in range(1, 10):
        sudoku[best_i, best_j] = k

        Q = calc_q(sudoku)
        G = calc_g(sudoku, neighbours_array)
        is_consistent = check_solution(Q, G, neighbours_array)

        if is_consistent:
            if solve_recursive(sudoku, neighbours_array):
                return True

        sudoku[best_i, best_j] = 0

    return False


class SudokuSolver:
    """Sudoku solver using OR-AND and constraint propagation."""

    def __init__(self, height=9, width=9):
        self.height = height
        self.width = width
        self.n_neighbours = 20
        self.neighbours_array = self._precompute_neighbours()

    def _precompute_neighbours(self):
        """Precomputes all neighbour indices to optimize computation."""
        indices_arr = np.indices((self.height, self.width)).transpose(1, 2, 0)
        neighbours_array = np.zeros(
            (self.height, self.width, self.n_neighbours, 2), dtype=np.int64
        )

        for i in range(self.height):
            for j in range(self.width):
                neighbours_array[i, j, :, :] = self._get_neighbours(i, j, indices_arr)

        return neighbours_array

    @staticmethod
    def _get_neighbours(i, j, indices_arr):
        """Gives neighbours of sudoku[i, j] value."""
        horizontal = indices_arr[i, :]
        vertical = indices_arr[:, j]
        cube = indices_arr[
            (i // 3) * 3 : (3 + (i // 3) * 3), (j // 3) * 3 : (3 + (j // 3) * 3)
        ].reshape(-1, 2)

        neighbours = np.vstack((horizontal, vertical, cube))
        neighbours = np.unique(neighbours, axis=0).tolist()
        neighbours.remove([i, j])
        return np.array(neighbours, dtype=np.int64)

    def solve(self, sudoku: np.ndarray) -> np.ndarray:
        """Solves a given sudoku array."""
        sudoku_copy = sudoku.copy()
        success = solve_recursive(sudoku_copy, self.neighbours_array)
        if not success:
            raise Exception("Refuse to recognize")
        return sudoku_copy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sudoku_number", type=int, help="number of sudoku in dataset")
    args = parser.parse_args()

    sudoku = sudoku_dataset[args.sudoku_number]

    print("sudoku")
    print("*" * 50)
    print(sudoku)

    solver = SudokuSolver()
    solved = solver.solve(sudoku)

    print("solution")
    print("*" * 50)
    print(solved)


if __name__ == "__main__":
    main()
