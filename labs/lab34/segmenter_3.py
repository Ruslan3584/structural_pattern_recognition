"""Three-class image segmentation using the shared diffusion engine."""

from __future__ import annotations

from numba import njit
import numpy as np

from .utilities import run_cli


@njit(cache=True)
def check_smooth_border(labelling, q):
    """Apply the notebook's unary penalties to jagged vertical class borders."""
    height, width = labelling.shape
    found_issue = False
    for i in range(1, height - 1):
        for j in range(width):
            current, below, above = (
                labelling[i, j],
                labelling[i + 1, j],
                labelling[i - 1, j],
            )
            if current == 0:
                if below == 1:
                    if above == 1:
                        found_issue = True
                        q[i, j, 1] += 1
                    elif above == 0:
                        found_issue = True
                        q[i + 1, j, 0] += 1
                elif below == 2:
                    if above == 2:
                        found_issue = True
                        q[i, j, 2] += 1
                    elif above == 0:
                        found_issue = True
                        q[i + 1, j, 0] += 1
            if current == 1 and below == 2:
                if above == 2:
                    found_issue = True
                    q[i, j, 2] += 1
                elif above == 1:
                    found_issue = True
                    q[i + 1, j, 1] += 1
    return found_issue, q


@njit(cache=True)
def check_cons(labelling, q):
    """Apply consistency penalties to horizontally crossing regions."""
    height, width = labelling.shape
    found_issue = False
    for i in range(height - 1):
        for j in range(width):
            current, below = labelling[i, j], labelling[i + 1, j]
            if current == 0 and below == 0:
                for neighbor in (j - 1, j + 1):
                    if 0 <= neighbor < width and (
                        (
                            labelling[i, neighbor] == 1
                            and labelling[i + 1, neighbor] == 1
                        )
                        or (
                            labelling[i, neighbor] == 2
                            and labelling[i + 1, neighbor] == 2
                        )
                    ):
                        found_issue = True
                        q[i, j, 1] += 1
            elif current == 1 and below == 1:
                for neighbor in (j - 1, j + 1):
                    if 0 <= neighbor < width and (
                        labelling[i, neighbor] == 2 and labelling[i + 1, neighbor] == 2
                    ):
                        found_issue = True
                        q[i, j, 2] += 1
    return found_issue, q


def notebook_border_refiner(labels, q):
    found_issue, q = check_smooth_border(labels, q)
    if not found_issue:
        found_issue, q = check_cons(labels, q)
    return found_issue, q


def main():
    run_cli(
        lab_number=4,
        labels=3,
        border_refiner=notebook_border_refiner,
        require_vertical_order=True,
    )


if __name__ == "__main__":
    main()
