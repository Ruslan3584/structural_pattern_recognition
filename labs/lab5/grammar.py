import argparse
import numpy as np
import matplotlib.pyplot as plt


def xor_columns(col1: np.ndarray, col2: np.ndarray, noise_level: float) -> float:
    """Calculates the log-likelihood cost between two image columns given a noise level.

    Args:
        col1 (np.ndarray): First image column array.
        col2 (np.ndarray): Second image column array.
        noise_level (float): Probability of noise (Bernoulli distribution parameter).

    Returns:
        float: Computed log-likelihood cost score.
    """
    eps = np.finfo(float).eps
    temp_noise = np.clip(noise_level, eps, 1.0 - eps)

    mismatch = col1 ^ col2
    match = 1 ^ mismatch

    return np.sum(mismatch * np.log(temp_noise) + match * np.log(1.0 - temp_noise))


def get_equation(
    input_string: str, dataset: np.ndarray, noise_level: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates the clean equation image, adds Bernoulli noise, and prepares reference symbols.

    Args:
        input_string (str): Equation representation string (use '#' for spaces).
        dataset (np.ndarray): Dataset containing character/symbol templates.
        noise_level (float): Noise level for the Bernoulli distribution.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Clean equation image, noised image,
        and reference symbol images.
    """
    input_string = input_string.replace("#", "2")

    input_array = np.array(list(map(list, input_string.split(" "))), dtype=int).ravel()
    n_rows = 3
    pic = np.hstack(dataset[input_array])

    # Construct equation image
    input_equation = np.vstack(np.hsplit(pic, n_rows))

    # Apply Bernoulli noise
    ksi = np.random.binomial(size=input_equation.size, n=1, p=noise_level).reshape(
        input_equation.shape
    )
    noised_image = ksi ^ input_equation

    # Reference symbols setup (label space size |K| = 8)
    reference_symbols = np.array(
        [0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1]
    )
    n_rows_ref = 3
    n_cols_ref = reference_symbols.shape[0] // n_rows_ref
    pic_ref = np.hstack(dataset[reference_symbols])
    reference_images = np.array(
        np.hsplit(np.vstack(np.hsplit(pic_ref, n_rows_ref)), n_cols_ref)
    )

    return input_equation, noised_image, reference_images


def restore(f: np.ndarray) -> np.ndarray:
    """Backtracks through the dynamic programming table to find the optimal label sequence.

    Args:
        f (np.ndarray): Dynamic programming table of shape (n_cols, 8, 2).

    Returns:
        np.ndarray: Array of optimal label indices for each column.
    """
    n_cols = f.shape[0]
    result = np.full(n_cols, -1, dtype=int)
    result[0] = np.argmax(f[0, :, 0])

    for i in range(1, n_cols):
        prev_state = result[i - 1]
        if prev_state in {0, 4, 5, 6}:
            result[i] = np.argmax(f[i, :, 0])
        else:
            result[i] = np.argmax(f[i, :, 1])

    return result


def get_best_paths(
    reference_images: np.ndarray, noised_image: np.ndarray, noise_level: float
) -> np.ndarray:
    """Computes the optimal path to restore the noisy equation image.

    Args:
        reference_images (np.ndarray): Array of reference symbol images.
        noised_image (np.ndarray): The corrupted image with added noise.
        noise_level (float): Noise probability level.

    Returns:
        np.ndarray: The reconstructed and restored final image.
    """
    n_cols = int((noised_image.shape[1] / noised_image.shape[0]) * 3)
    noised_image_splitted = np.hsplit(noised_image, n_cols)

    # Initialize DP table with -inf (representing impossible state combinations)
    f = np.full((n_cols, 8, 2), -np.inf)

    # Base case: last column initialization
    last_col_costs = np.array(
        [
            xor_columns(ref, noised_image_splitted[-1], noise_level)
            for ref in reference_images
        ]
    )

    for c in [0, 5, 6]:
        f[-1, c, 0] = last_col_costs[c]
    f[-1, 4, 1] = last_col_costs[4]

    # Backward pass
    for column in range(n_cols - 2, -1, -1):
        col_costs = np.array(
            [
                xor_columns(ref, noised_image_splitted[column], noise_level)
                for ref in reference_images
            ]
        )

        max_prev_0 = np.max(f[column + 1, :, 0])
        max_prev_1 = np.max(f[column + 1, :, 1])

        for c in [0, 5, 6]:
            f[column, c, 0] = col_costs[c] + max_prev_0
        f[column, 1, 0] = col_costs[1] + max_prev_1

        f[column, 4, 1] = col_costs[4] + max_prev_0
        for c in [2, 3, 7]:
            f[column, c, 1] = col_costs[c] + max_prev_1

    result = restore(f)
    return np.hstack(reference_images[result])


def main() -> None:
    """Main execution entry point for parsing arguments and running image restoration."""
    parser = argparse.ArgumentParser(
        description="Binary addition using Cocke–Younger–Kasami algorithm for decoding noisy images."
    )
    parser.add_argument(
        "input_string",
        type=str,
        help="Equation string (first summand, second summand, solution). Use '#' for spaces.",
    )
    parser.add_argument(
        "noise_level",
        type=float,
        help="Noise level probability for the Bernoulli distribution.",
    )
    args = parser.parse_args()

    dataset = np.load("labs/lab5/terminal_symbols.npy")
    input_equation_image, noised_image, reference_images = get_equation(
        args.input_string, dataset, args.noise_level
    )

    result_img = get_best_paths(reference_images, noised_image, args.noise_level)

    plt.imsave("input_equation.png", input_equation_image, cmap="binary")
    plt.imsave("noised_image.png", noised_image, cmap="binary")
    plt.imsave("solution.png", result_img, cmap="binary")


if __name__ == "__main__":
    main()
