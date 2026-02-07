"""
Advanced Feature Engineering for Roulette Prediction.

Extracts sophisticated features from spin sequences including:
- Temporal patterns (time-based analysis)
- Sequence patterns (streaks, gaps, frequencies)
- Statistical features (rolling stats, Z-scores)
- Wheel sector analysis (Voisins, Orphelins, Tiers)
- Mathematical patterns (Fibonacci, Dozen/Column distributions)
"""

import numpy as np
from typing import Optional
from datetime import datetime
from collections import Counter, deque
from roulette import number_to_color, RED_NUMBERS, TOTAL_NUMBERS

# European roulette wheel order (physical positions)
WHEEL_ORDER = [
    0, 32, 15, 19, 4, 21, 2, 25, 17, 34, 6, 27, 13, 36, 11, 30, 8, 23, 10,
    5, 24, 16, 33, 1, 20, 14, 31, 9, 22, 18, 29, 7, 28, 12, 35, 3, 26
]

# Wheel sectors (standard French bets)
VOISINS_DU_ZERO = [0, 2, 3, 4, 7, 12, 15, 18, 19, 21, 22, 25, 26, 28, 29, 32, 35]
TIERS_DU_CYLINDRE = [5, 8, 10, 11, 13, 16, 23, 24, 27, 30, 33, 36]
ORPHELINS = [1, 6, 9, 14, 17, 20, 31, 34]

# Dozens and columns
DOZEN_1 = list(range(1, 13))
DOZEN_2 = list(range(13, 25))
DOZEN_3 = list(range(25, 37))
COLUMN_1 = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34]
COLUMN_2 = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]
COLUMN_3 = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36]

# Low/High
LOW_NUMBERS = list(range(1, 19))
HIGH_NUMBERS = list(range(19, 37))

# Even/Odd
EVEN_NUMBERS = [n for n in range(1, 37) if n % 2 == 0]
ODD_NUMBERS = [n for n in range(1, 37) if n % 2 == 1]


class AdvancedFeatureExtractor:
    """
    Extract comprehensive features from roulette spin sequences.
    
    Features include:
    - Hot/cold number analysis
    - Streak detection
    - Sector bias analysis
    - Gap analysis (sleepers)
    - Statistical deviations
    - Pattern sequences
    """
    
    def __init__(self, window_sizes: list[int] = None):
        """
        Initialize the feature extractor.
        
        Args:
            window_sizes: List of lookback windows for different analyses
        """
        self.window_sizes = window_sizes or [5, 10, 20, 50, 100]
        
    def extract_basic_features(self, numbers: list[int], window: int = 10) -> dict:
        """Extract basic frequency and distribution features."""
        if len(numbers) < window:
            recent = numbers
        else:
            recent = numbers[-window:]
        
        counts = Counter(recent)
        total = len(recent)
        
        # Color distribution
        colors = [number_to_color(n) for n in recent]
        color_counts = Counter(colors)
        
        return {
            'red_ratio': color_counts.get('red', 0) / max(total, 1),
            'black_ratio': color_counts.get('black', 0) / max(total, 1),
            'green_ratio': color_counts.get('green', 0) / max(total, 1),
            'unique_numbers': len(counts),
            'most_common': counts.most_common(1)[0][0] if counts else 0,
            'most_common_freq': counts.most_common(1)[0][1] / max(total, 1) if counts else 0,
        }
    
    def extract_streak_features(self, numbers: list[int]) -> dict:
        """Detect consecutive patterns and streaks."""
        if len(numbers) < 2:
            return {
                'current_color_streak': 0,
                'current_streak_color': 'none',
                'max_color_streak': 0,
                'alternating_count': 0,
                'same_dozen_streak': 0,
                'same_column_streak': 0,
            }
        
        colors = [number_to_color(n) for n in numbers]
        
        # Current color streak
        current_streak = 1
        current_color = colors[-1]
        for i in range(len(colors) - 2, -1, -1):
            if colors[i] == current_color:
                current_streak += 1
            else:
                break
        
        # Max color streak
        max_streak = 1
        temp_streak = 1
        for i in range(1, len(colors)):
            if colors[i] == colors[i-1]:
                temp_streak += 1
                max_streak = max(max_streak, temp_streak)
            else:
                temp_streak = 1
        
        # Alternating pattern detection
        alternating = 0
        for i in range(1, len(colors)):
            if colors[i] != colors[i-1]:
                alternating += 1
            else:
                alternating = 0
        
        # Dozen streak
        def get_dozen(n):
            if n == 0: return 0
            if n <= 12: return 1
            if n <= 24: return 2
            return 3
        
        dozens = [get_dozen(n) for n in numbers]
        dozen_streak = 1
        for i in range(len(dozens) - 2, -1, -1):
            if dozens[i] == dozens[-1] and dozens[-1] != 0:
                dozen_streak += 1
            else:
                break
        
        # Column streak
        def get_column(n):
            if n == 0: return 0
            return ((n - 1) % 3) + 1
        
        columns = [get_column(n) for n in numbers]
        column_streak = 1
        for i in range(len(columns) - 2, -1, -1):
            if columns[i] == columns[-1] and columns[-1] != 0:
                column_streak += 1
            else:
                break
        
        return {
            'current_color_streak': current_streak,
            'current_streak_color': current_color,
            'max_color_streak': max_streak,
            'alternating_count': alternating,
            'same_dozen_streak': dozen_streak,
            'same_column_streak': column_streak,
        }
    
    def extract_hot_cold_features(self, numbers: list[int], window: int = 50) -> dict:
        """Identify hot and cold numbers based on frequency."""
        if len(numbers) < window:
            recent = numbers
        else:
            recent = numbers[-window:]
        
        counts = Counter(recent)
        total = len(recent)
        expected_freq = total / TOTAL_NUMBERS
        
        # Calculate deviation from expected
        deviations = {}
        for n in range(TOTAL_NUMBERS):
            actual = counts.get(n, 0)
            deviation = (actual - expected_freq) / max(expected_freq, 0.1)
            deviations[n] = deviation
        
        # Hot numbers (appearing more than expected)
        hot = sorted([n for n, dev in deviations.items() if dev > 1], 
                     key=lambda x: deviations[x], reverse=True)[:5]
        
        # Cold numbers (appearing less than expected)
        cold = sorted([n for n, dev in deviations.items() if dev < -0.5], 
                      key=lambda x: deviations[x])[:5]
        
        # Sleeper numbers (not appeared recently)
        all_numbers = set(range(TOTAL_NUMBERS))
        appeared = set(recent)
        sleepers = list(all_numbers - appeared)
        
        return {
            'hot_numbers': hot,
            'cold_numbers': cold,
            'sleepers': sleepers[:10],
            'hottest_number': hot[0] if hot else -1,
            'coldest_number': cold[0] if cold else -1,
            'num_sleepers': len(sleepers),
        }
    
    def extract_sector_features(self, numbers: list[int], window: int = 50) -> dict:
        """Analyze wheel sector distribution (physical wheel position)."""
        if len(numbers) < window:
            recent = numbers
        else:
            recent = numbers[-window:]
        
        total = len(recent)
        
        # Sector counts
        voisins_count = sum(1 for n in recent if n in VOISINS_DU_ZERO)
        tiers_count = sum(1 for n in recent if n in TIERS_DU_CYLINDRE)
        orphelins_count = sum(1 for n in recent if n in ORPHELINS)
        
        # Expected ratios for fair wheel
        voisins_expected = len(VOISINS_DU_ZERO) / TOTAL_NUMBERS
        tiers_expected = len(TIERS_DU_CYLINDRE) / TOTAL_NUMBERS
        orphelins_expected = len(ORPHELINS) / TOTAL_NUMBERS
        
        # Actual ratios
        voisins_ratio = voisins_count / max(total, 1)
        tiers_ratio = tiers_count / max(total, 1)
        orphelins_ratio = orphelins_count / max(total, 1)
        
        # Bias scores (deviation from expected)
        voisins_bias = (voisins_ratio - voisins_expected) / max(voisins_expected, 0.01)
        tiers_bias = (tiers_ratio - tiers_expected) / max(tiers_expected, 0.01)
        orphelins_bias = (orphelins_ratio - orphelins_expected) / max(orphelins_expected, 0.01)
        
        # Physical wheel position analysis
        wheel_positions = []
        for n in recent:
            try:
                pos = WHEEL_ORDER.index(n)
                wheel_positions.append(pos)
            except ValueError:
                pass
        
        # Wheel position clustering (are spins clustering in wheel areas?)
        if wheel_positions:
            pos_std = np.std(wheel_positions)
            pos_mean = np.mean(wheel_positions)
        else:
            pos_std = 0
            pos_mean = 0
        
        return {
            'voisins_ratio': voisins_ratio,
            'tiers_ratio': tiers_ratio,
            'orphelins_ratio': orphelins_ratio,
            'voisins_bias': voisins_bias,
            'tiers_bias': tiers_bias,
            'orphelins_bias': orphelins_bias,
            'wheel_position_std': pos_std,
            'wheel_position_mean': pos_mean,
            'dominant_sector': max([('voisins', voisins_ratio), 
                                    ('tiers', tiers_ratio), 
                                    ('orphelins', orphelins_ratio)],
                                   key=lambda x: x[1])[0],
        }
    
    def extract_gap_features(self, numbers: list[int]) -> dict:
        """Analyze gaps (sleeper patterns) for each number."""
        if not numbers:
            return {
                'avg_gap': 0,
                'max_current_gap': 0,
                'numbers_over_expected_gap': [],
            }
        
        # Calculate current gap for each number
        current_gaps = {}
        for n in range(TOTAL_NUMBERS):
            try:
                last_idx = len(numbers) - 1 - numbers[::-1].index(n)
                current_gaps[n] = len(numbers) - 1 - last_idx
            except ValueError:
                current_gaps[n] = len(numbers)  # Never appeared
        
        # Expected gap (on average, number appears every 37 spins)
        expected_gap = TOTAL_NUMBERS
        
        # Numbers overdue
        overdue = [n for n, gap in current_gaps.items() if gap > expected_gap * 1.5]
        
        return {
            'avg_gap': np.mean(list(current_gaps.values())),
            'max_current_gap': max(current_gaps.values()),
            'max_gap_number': max(current_gaps.items(), key=lambda x: x[1])[0],
            'numbers_over_expected_gap': overdue[:5],
            'overdue_count': len(overdue),
        }
    
    def extract_statistical_features(self, numbers: list[int], window: int = 100) -> dict:
        """Extract advanced statistical features."""
        if len(numbers) < 3:
            return {
                'mean': 0, 'std': 0, 'skewness': 0, 'kurtosis': 0,
                'entropy': 0, 'chi_squared': 0,
            }
        
        if len(numbers) < window:
            recent = numbers
        else:
            recent = numbers[-window:]
        
        arr = np.array(recent)
        
        # Basic stats
        mean = np.mean(arr)
        std = np.std(arr)
        
        # Skewness and Kurtosis
        n = len(arr)
        if std > 0:
            skewness = np.mean(((arr - mean) / std) ** 3)
            kurtosis = np.mean(((arr - mean) / std) ** 4) - 3
        else:
            skewness = 0
            kurtosis = 0
        
        # Entropy (measure of randomness)
        counts = Counter(recent)
        probs = [c / n for c in counts.values()]
        entropy = -sum(p * np.log2(p + 1e-10) for p in probs)
        max_entropy = np.log2(TOTAL_NUMBERS)
        normalized_entropy = entropy / max_entropy
        
        # Chi-squared statistic (deviation from uniform)
        expected = n / TOTAL_NUMBERS
        chi_squared = sum((counts.get(i, 0) - expected) ** 2 / expected 
                          for i in range(TOTAL_NUMBERS))
        
        return {
            'mean': mean,
            'std': std,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'entropy': normalized_entropy,
            'chi_squared': chi_squared,
        }
    
    def extract_transition_features(self, numbers: list[int]) -> dict:
        """Analyze number-to-number transitions (Markov-like)."""
        if len(numbers) < 2:
            return {
                'same_color_prob': 0,
                'color_change_prob': 0,
                'same_parity_prob': 0,
                'same_dozen_prob': 0,
            }
        
        same_color = 0
        same_parity = 0
        same_dozen = 0
        
        for i in range(1, len(numbers)):
            prev, curr = numbers[i-1], numbers[i]
            
            if number_to_color(prev) == number_to_color(curr):
                same_color += 1
            
            if prev != 0 and curr != 0:
                if prev % 2 == curr % 2:
                    same_parity += 1
                
                # Dozen comparison
                prev_dozen = (prev - 1) // 12
                curr_dozen = (curr - 1) // 12
                if prev_dozen == curr_dozen:
                    same_dozen += 1
        
        n = len(numbers) - 1
        
        return {
            'same_color_prob': same_color / max(n, 1),
            'color_change_prob': 1 - (same_color / max(n, 1)),
            'same_parity_prob': same_parity / max(n, 1),
            'same_dozen_prob': same_dozen / max(n, 1),
        }
    
    def extract_all_features(self, numbers: list[int], 
                            timestamps: list[str] = None) -> dict:
        """
        Extract all features for model input.
        
        Args:
            numbers: List of spin outcomes (0-36)
            timestamps: Optional list of ISO timestamps for temporal features
            
        Returns:
            Dictionary of all extracted features
        """
        features = {}
        
        # Basic features at different windows
        for w in [10, 20, 50]:
            basic = self.extract_basic_features(numbers, w)
            for k, v in basic.items():
                features[f'{k}_w{w}'] = v
        
        # Streak features
        streaks = self.extract_streak_features(numbers)
        features.update(streaks)
        
        # Hot/cold features
        hot_cold = self.extract_hot_cold_features(numbers)
        # Extract numeric features only
        features['hottest_number'] = hot_cold['hottest_number']
        features['coldest_number'] = hot_cold['coldest_number']
        features['num_sleepers'] = hot_cold['num_sleepers']
        
        # Sector features
        sectors = self.extract_sector_features(numbers)
        features['voisins_ratio'] = sectors['voisins_ratio']
        features['tiers_ratio'] = sectors['tiers_ratio']
        features['orphelins_ratio'] = sectors['orphelins_ratio']
        features['voisins_bias'] = sectors['voisins_bias']
        features['tiers_bias'] = sectors['tiers_bias']
        features['orphelins_bias'] = sectors['orphelins_bias']
        features['wheel_position_std'] = sectors['wheel_position_std']
        
        # Gap features
        gaps = self.extract_gap_features(numbers)
        features['avg_gap'] = gaps['avg_gap']
        features['max_current_gap'] = gaps['max_current_gap']
        features['overdue_count'] = gaps['overdue_count']
        
        # Statistical features
        stats = self.extract_statistical_features(numbers)
        features.update(stats)
        
        # Transition features
        transitions = self.extract_transition_features(numbers)
        features.update(transitions)
        
        return features
    
    def features_to_vector(self, features: dict) -> np.ndarray:
        """Convert feature dictionary to numpy vector for model input."""
        # Define feature order for consistency
        feature_order = [
            # Basic features at different windows
            'red_ratio_w10', 'black_ratio_w10', 'green_ratio_w10',
            'unique_numbers_w10', 'most_common_freq_w10',
            'red_ratio_w20', 'black_ratio_w20', 'green_ratio_w20',
            'unique_numbers_w20', 'most_common_freq_w20',
            'red_ratio_w50', 'black_ratio_w50', 'green_ratio_w50',
            'unique_numbers_w50', 'most_common_freq_w50',
            # Streak features
            'current_color_streak', 'max_color_streak', 'alternating_count',
            'same_dozen_streak', 'same_column_streak',
            # Hot/cold
            'hottest_number', 'coldest_number', 'num_sleepers',
            # Sector
            'voisins_ratio', 'tiers_ratio', 'orphelins_ratio',
            'voisins_bias', 'tiers_bias', 'orphelins_bias',
            'wheel_position_std',
            # Gap
            'avg_gap', 'max_current_gap', 'overdue_count',
            # Statistical
            'mean', 'std', 'skewness', 'kurtosis', 'entropy', 'chi_squared',
            # Transition
            'same_color_prob', 'color_change_prob', 'same_parity_prob', 'same_dozen_prob',
        ]
        
        vector = []
        for f in feature_order:
            val = features.get(f, 0)
            if isinstance(val, str):
                val = 0  # Skip string features
            vector.append(float(val) if val is not None else 0.0)
        
        return np.array(vector)


# Singleton instance
_feature_extractor: Optional[AdvancedFeatureExtractor] = None


def get_feature_extractor() -> AdvancedFeatureExtractor:
    global _feature_extractor
    if _feature_extractor is None:
        _feature_extractor = AdvancedFeatureExtractor()
    return _feature_extractor
