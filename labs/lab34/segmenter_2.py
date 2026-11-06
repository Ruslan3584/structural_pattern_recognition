"""Two-class image segmentation using the shared diffusion engine."""

from __future__ import annotations

from numba import njit
import numpy as np

from .utilities import run_cli


@njit(cache=True)
def check_smooth_border(mask):
    """Return whether a binary mask has a single, non-crossing smooth border."""
    height, width = mask.shape
    suffix_zero = np.zeros((height + 1, width), dtype=np.bool_)
    for j in range(width):
        seen_zero = False
        for i in range(height - 1, -1, -1):
            if not mask[i, j]:
                seen_zero = True
            suffix_zero[i, j] = seen_zero
    for i in range(height - 1):
        for j in range(width):
            if not mask[i, j] and mask[i + 1, j] and suffix_zero[i + 1, j]:
                return False
            if not mask[i, j] and not mask[i + 1, j]:
                if j > 0 and mask[i, j - 1] and mask[i + 1, j - 1]:
                    return False
                if j + 1 < width and mask[i, j + 1] and mask[i + 1, j + 1]:
                    return False
    return True


def main():
    run_cli(
        lab_number=3,
        labels=2,
        border_check=check_smooth_border,
    )


if __name__ == "__main__":
    main()
