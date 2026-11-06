import argparse
import json
import string

import cv2
import numpy as np

EPS = 1e-320


class SpaceFinder:
    """
    Find spaces in a noisy binary image of monospaced text.
    """

    def __init__(
        self,
        input_string: str,
        noise: float,
        frequencies_path: str,
        seed: int | None = None,
    ):
        self.input_string = input_string.lower()
        self.noise = float(noise)
        self.frequencies_path = frequencies_path

        self.letter_h = 25
        self.letter_w = 27
        self.alphabet = list(string.ascii_lowercase + " ")

        if not 0.0 <= self.noise <= 1.0:
            raise ValueError("Noise level must be in [0, 1].")

        with open(self.frequencies_path, "r", encoding="utf-8") as f:
            self.frequencies_dict = json.load(f)

        # Generate alphabet character templates
        self.alphabet_dict = {
            char: self.generate_letter_img(char, self.letter_h, self.letter_w)
            for char in self.alphabet
        }

        self.input_string_im = self.string_to_image(self.input_string)

        # Apply Bernoulli noise using XOR
        rng = np.random.default_rng(seed)
        noise = rng.binomial(1, self.noise, size=self.input_string_im.shape)
        self.noised_im = noise ^ self.input_string_im

        self.bigram_probs = self._calculate_bigram_probs()
        self.initial_probs = self._calculate_initial_probs()
        self.noised_with_segmented_spaces = self.noised_im.copy()

    @staticmethod
    def generate_letter_img(letter: str, height: int, width: int) -> np.ndarray:
        """Generate a binary image for a character."""
        img = np.zeros((height, width), dtype=np.uint8)
        if letter == " ":
            return img

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = min(height, width) / 25.0
        thickness = max(1, int(font_scale * 2))
        (text_w, text_h), _ = cv2.getTextSize(letter, font, font_scale, thickness)
        x = max(0, (width - text_w) // 2)
        y = min(height - 1, max(text_h, (height + text_h) // 2))

        cv2.putText(img, letter, (x, y), font, font_scale, 1, thickness, cv2.LINE_AA)
        return (img > 0).astype(np.uint8)

    def string_to_image(self, string_to_convert: str) -> np.ndarray:
        """Concatenate character images horizontally."""
        if not string_to_convert:
            return np.zeros((self.letter_h, 0), dtype=np.uint8)
        return np.hstack([self.alphabet_dict[char] for char in string_to_convert])

    def _calculate_bigram_probs(self) -> np.ndarray:
        """Compute log conditional probabilities of bigrams P(next | current)."""
        k = len(self.alphabet)
        counts = np.zeros((k, k), dtype=np.float64)

        for i, first in enumerate(self.alphabet):
            for j, second in enumerate(self.alphabet):
                counts[i, j] = float(self.frequencies_dict.get(first + second, 0))

        counts += EPS
        probs = counts / counts.sum(axis=1, keepdims=True)

        return np.log(np.maximum(probs, EPS))

    def _calculate_initial_probs(self) -> np.ndarray:
        """Compute initial character probabilities."""
        counts = np.array(
            [float(self.frequencies_dict.get(char, 0)) for char in self.alphabet],
            dtype=np.float64,
        )
        if counts.sum() == 0:
            counts = np.ones(len(self.alphabet), dtype=np.float64)
        counts += EPS
        return counts / counts.sum()

    def find_spaces(self) -> list[int]:
        """
        Find space positions using the Viterbi algorithm (MAP estimation).
        """
        height, width = self.noised_im.shape
        char_width = self.letter_w
        N = width // char_width
        num_states = len(self.alphabet)

        # Bernoulli noise model: compute emission log-probabilities
        p_safe = max(min(self.noise, 1.0 - 1e-15), 1e-15)
        log_1_p, log_p = np.log(1.0 - p_safe), np.log(p_safe)

        emissions = np.zeros((N, num_states), dtype=np.float64)
        for t in range(N):
            Y_t = self.noised_im[:, t * char_width : (t + 1) * char_width]
            for c_idx, char in enumerate(self.alphabet):
                xor_diff = Y_t ^ self.alphabet_dict[char]
                emissions[t, c_idx] = np.sum(
                    (1 - xor_diff) * log_1_p + xor_diff * log_p
                )

        # Viterbi dynamic programming
        V = np.full((N, num_states), -np.inf, dtype=np.float64)
        backpointer = np.zeros((N, num_states), dtype=int)

        V[0, :] = np.log(self.initial_probs) + emissions[0, :]

        for t in range(1, N):
            for c in range(num_states):
                scores = V[t - 1] + self.bigram_probs[:, c]
                best_prev = np.argmax(scores)
                V[t, c] = emissions[t, c] + scores[best_prev]
                backpointer[t, c] = best_prev

        # Backtracking
        best_state = np.argmax(V[N - 1])
        path = [best_state]
        for t in range(N - 1, 0, -1):
            best_state = backpointer[t, best_state]
            path.append(best_state)
        path.reverse()

        # Extract space slot indices
        spaces = [i for i, idx in enumerate(path) if self.alphabet[idx] == " "]
        self.noised_with_segmented_spaces = self.paint_spaces_in_red(
            self.noised_im, spaces
        )
        return spaces

    def paint_spaces_in_red(self, image: np.ndarray, spaces: list[int]) -> np.ndarray:
        """Mark found spaces in red on a BGR image with black text on white background."""
        # Convert binary 0/1 to 0-255 grayscale with black text (0) and white background (255)
        gray = ((1 - image) * 255).astype(np.uint8)
        # Use COLOR_GRAY2BGR so OpenCV saves colors correctly (BGR format: [0, 0, 255] is Red)
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        height, width = image.shape

        for index in spaces:
            if 0 <= index < width // self.letter_w:
                x1 = index * self.letter_w
                x2 = min((index + 1) * self.letter_w - 1, width - 1)
                bgr[:, x1 : x2 + 1, :] = [0, 0, 255]  # Red in BGR

        return bgr


def main():
    parser = argparse.ArgumentParser(description="Find spaces in a noisy text image.")
    parser.add_argument("--input_string", type=str, required=True, help="Input string.")
    parser.add_argument(
        "--noise_level", type=float, required=True, help="Noise level [0, 1]."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()

    space_finder = SpaceFinder(
        args.input_string, args.noise_level, "labs/lab1/frequencies.json", args.seed
    )
    ground_truth = [i for i, c in enumerate(space_finder.input_string) if c == " "]
    predicted_spaces = space_finder.find_spaces()

    print("Ground-truth spaces:", ground_truth)
    print("Predicted spaces:", predicted_spaces)

    # Save debug images with consistent black text on white background style
    debug_input_image = space_finder.paint_spaces_in_red(
        space_finder.input_string_im, ground_truth
    )
    debug_noised_image = space_finder.paint_spaces_in_red(space_finder.noised_im, [])

    cv2.imwrite("image.png", debug_input_image)
    cv2.imwrite("noised_image.png", debug_noised_image)
    cv2.imwrite("noised_image_result.png", space_finder.noised_with_segmented_spaces)
    print("Debug images have been saved successfully.")


if __name__ == "__main__":
    main()
