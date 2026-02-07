"""
Simple ML model: predict next color (and optionally number) from sequence of spins.
Uses sliding window + sklearn classifier (MultinomialNB or MLPClassifier).
"""
from typing import Optional
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

from roulette import number_to_color, RED_NUMBERS, TOTAL_NUMBERS

WINDOW_SIZE = 5
MIN_SAMPLES = WINDOW_SIZE + 10  # need enough to train


def _build_sequences(numbers: list[int], window: int) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, y) for next-color prediction. X = last `window` numbers (as indices 0-36), y = next color."""
    X_list = []
    y_list = []
    for i in range(len(numbers) - window):
        X_list.append(numbers[i : i + window])
        y_list.append(number_to_color(numbers[i + window]))
    return np.array(X_list), np.array(y_list)


def _build_sequences_number(numbers: list[int], window: int) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, y) for next-number prediction. y = next number 0-36."""
    X_list = []
    y_list = []
    for i in range(len(numbers) - window):
        X_list.append(numbers[i : i + window])
        y_list.append(numbers[i + window])
    return np.array(X_list), np.array(y_list)


class RoulettePredictor:
    """Train on spin sequence, predict next color and number probabilities."""

    def __init__(self, window_size: int = WINDOW_SIZE):
        self.window_size = window_size
        self.color_encoder = LabelEncoder()
        self.color_encoder.fit(["red", "black", "green"])
        self.color_model: Optional[MLPClassifier] = None
        self.number_model: Optional[MLPClassifier] = None
        self._trained = False

    def fit(self, numbers: list[int]) -> bool:
        """Train on full history. Returns True if trained."""
        if len(numbers) < MIN_SAMPLES:
            return False
        X_color, y_color = _build_sequences(numbers, self.window_size)
        y_color_enc = self.color_encoder.transform(y_color)
        self.color_model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
        self.color_model.fit(X_color, y_color_enc)

        X_num, y_num = _build_sequences_number(numbers, self.window_size)
        self.number_model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
        self.number_model.fit(X_num, y_num)
        self._trained = True
        return True

    def predict_color_probs(self, numbers: list[int]) -> Optional[dict[str, float]]:
        """Predict P(next color) from last window. Returns None if not enough data or not trained."""
        if not self._trained or self.color_model is None or len(numbers) < self.window_size:
            return None
        last = numbers[-self.window_size :]
        X = np.array([last])
        probs = self.color_model.predict_proba(X)[0]
        labels = self.color_encoder.classes_
        return dict(zip(labels, probs.tolist()))

    def predict_number_probs(self, numbers: list[int], top_n: int = 10) -> Optional[list[tuple[int, float]]]:
        """Predict P(next number) from last window. Returns top_n (number, prob) or None."""
        if not self._trained or self.number_model is None or len(numbers) < self.window_size:
            return None
        last = numbers[-self.window_size :]
        X = np.array([last])
        probs = self.number_model.predict_proba(X)[0]
        classes = self.number_model.classes_
        pairs = list(zip(classes.tolist(), probs.tolist()))
        pairs.sort(key=lambda x: -x[1])
        return pairs[:top_n]

    def ensure_trained(self, numbers: list[int]) -> None:
        """Re-train if we have enough data."""
        if len(numbers) >= MIN_SAMPLES:
            self.fit(numbers)


# Singleton used by API
_predictor: Optional[RoulettePredictor] = None


def get_predictor() -> RoulettePredictor:
    global _predictor
    if _predictor is None:
        _predictor = RoulettePredictor()
    return _predictor
