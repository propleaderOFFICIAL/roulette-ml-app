"""
European roulette: number -> color mapping and probability calculations.
37 numbers: 0, 1-36. Red, Black, Green (0 only).
"""
from typing import Literal

Color = Literal["red", "black", "green"]

# European roulette: red numbers
RED_NUMBERS = {1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36}
TOTAL_NUMBERS = 37


def number_to_color(number: int) -> Color:
    """Map number 0-36 to color (european roulette)."""
    if number < 0 or number > 36:
        raise ValueError("Number must be between 0 and 36")
    if number == 0:
        return "green"
    return "red" if number in RED_NUMBERS else "black"


def theoretical_probabilities() -> dict[str, float]:
    """Return theoretical probabilities for red, black, green."""
    return {
        "red": 18 / TOTAL_NUMBERS,
        "black": 18 / TOTAL_NUMBERS,
        "green": 1 / TOTAL_NUMBERS,
    }


def theoretical_number_probability() -> float:
    """Probability for any single number (1/37)."""
    return 1.0 / TOTAL_NUMBERS


def empirical_color_probabilities(numbers: list[int]) -> dict[str, float]:
    """Compute empirical (frequency) probabilities per color from a list of spins."""
    if not numbers:
        return {"red": 0.0, "black": 0.0, "green": 0.0}
    counts: dict[str, int] = {"red": 0, "black": 0, "green": 0}
    for n in numbers:
        counts[number_to_color(n)] += 1
    total = len(numbers)
    return {c: count / total for c, count in counts.items()}


def empirical_number_probabilities(numbers: list[int]) -> dict[int, float]:
    """Compute empirical frequency per number (0-36). Returns only numbers that appeared."""
    if not numbers:
        return {}
    counts: dict[int, int] = {}
    for n in numbers:
        if 0 <= n <= 36:
            counts[n] = counts.get(n, 0) + 1
    total = len(numbers)
    return {n: count / total for n, count in counts.items()}


def empirical_top_numbers(numbers: list[int], top_n: int = 10) -> list[tuple[int, float]]:
    """Return top N numbers by frequency as (number, probability) sorted descending."""
    probs = empirical_number_probabilities(numbers)
    sorted_items = sorted(probs.items(), key=lambda x: -x[1])
    return sorted_items[:top_n]
