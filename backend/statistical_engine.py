"""
Statistical Analysis Engine for Roulette.

Advanced statistical analysis including:
- Chi-squared tests for bias detection
- Monte Carlo simulations
- Markov chain analysis
- Bayesian probability updates
- Entropy and randomness metrics
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
from scipy import stats
from datetime import datetime

from roulette import number_to_color, TOTAL_NUMBERS, RED_NUMBERS


class StatisticalEngine:
    """
    Advanced statistical analysis for roulette outcomes.
    
    Provides:
    - Chi-squared goodness-of-fit tests
    - Monte Carlo simulations for probability estimation
    - Markov chain transition analysis
    - Bayesian inference for probability updating
    - Entropy and randomness metrics
    """
    
    def __init__(self):
        self.monte_carlo_iterations = 10000
        
    def chi_squared_test(self, numbers: list[int]) -> Dict[str, Any]:
        """
        Perform chi-squared goodness-of-fit test.
        
        Tests whether the observed distribution differs significantly
        from the expected uniform distribution.
        
        Returns:
            Test statistic, p-value, and interpretation
        """
        if len(numbers) < 50:
            return {
                'statistic': 0,
                'p_value': 1.0,
                'degrees_of_freedom': TOTAL_NUMBERS - 1,
                'significant': False,
                'interpretation': 'Insufficient data (need 50+ spins)',
            }
        
        # Observed frequencies
        counts = Counter(numbers)
        observed = [counts.get(n, 0) for n in range(TOTAL_NUMBERS)]
        
        # Expected frequencies (uniform distribution)
        total = len(numbers)
        expected = [total / TOTAL_NUMBERS] * TOTAL_NUMBERS
        
        # Chi-squared test
        chi2, p_value = stats.chisquare(observed, expected)
        
        # Interpretation
        significant = p_value < 0.05
        if p_value < 0.01:
            interpretation = "Highly significant deviation from random (p < 0.01)"
        elif p_value < 0.05:
            interpretation = "Significant deviation from random (p < 0.05)"
        elif p_value < 0.10:
            interpretation = "Marginally significant (p < 0.10)"
        else:
            interpretation = "Distribution appears random (p >= 0.10)"
        
        return {
            'statistic': float(chi2),
            'p_value': float(p_value),
            'degrees_of_freedom': TOTAL_NUMBERS - 1,
            'significant': significant,
            'interpretation': interpretation,
            'total_spins': total,
        }
    
    def chi_squared_color_test(self, numbers: list[int]) -> Dict[str, Any]:
        """
        Chi-squared test specifically for color distribution.
        """
        if len(numbers) < 20:
            return {
                'statistic': 0,
                'p_value': 1.0,
                'significant': False,
                'interpretation': 'Insufficient data',
            }
        
        colors = [number_to_color(n) for n in numbers]
        counts = Counter(colors)
        total = len(numbers)
        
        # Expected: 18/37 red, 18/37 black, 1/37 green
        observed = [
            counts.get('red', 0),
            counts.get('black', 0),
            counts.get('green', 0),
        ]
        expected = [
            total * 18 / 37,
            total * 18 / 37,
            total * 1 / 37,
        ]
        
        chi2, p_value = stats.chisquare(observed, expected)
        
        return {
            'statistic': float(chi2),
            'p_value': float(p_value),
            'observed': {'red': observed[0], 'black': observed[1], 'green': observed[2]},
            'expected': {'red': expected[0], 'black': expected[1], 'green': expected[2]},
            'significant': p_value < 0.05,
            'interpretation': 'Color distribution appears biased' if p_value < 0.05 else 'Color distribution appears random',
        }
    
    def monte_carlo_simulation(self, numbers: list[int], 
                               target_type: str = 'color') -> Dict[str, Any]:
        """
        Monte Carlo simulation to estimate future probabilities.
        
        Args:
            numbers: Historical spin data
            target_type: 'color' or 'number'
            
        Returns:
            Simulated probability distribution
        """
        if len(numbers) < 10:
            return {'error': 'Insufficient data'}
        
        # Build empirical distribution
        if target_type == 'color':
            colors = [number_to_color(n) for n in numbers]
            counts = Counter(colors)
            total = len(colors)
            probs = {c: count / total for c, count in counts.items()}
            
            # Ensure all colors present
            for c in ['red', 'black', 'green']:
                if c not in probs:
                    probs[c] = 1 / TOTAL_NUMBERS
            
            # Simulate
            simulated = {'red': 0, 'black': 0, 'green': 0}
            for _ in range(self.monte_carlo_iterations):
                r = np.random.random()
                cumulative = 0
                for color, prob in probs.items():
                    cumulative += prob
                    if r < cumulative:
                        simulated[color] += 1
                        break
            
            # Normalize
            for c in simulated:
                simulated[c] /= self.monte_carlo_iterations
            
            return {
                'type': 'color',
                'simulated_probabilities': simulated,
                'empirical_probabilities': probs,
                'theoretical_probabilities': {'red': 18/37, 'black': 18/37, 'green': 1/37},
                'iterations': self.monte_carlo_iterations,
            }
        else:
            # Number simulation
            counts = Counter(numbers)
            total = len(numbers)
            
            # Laplace smoothing to avoid zero probabilities
            probs = {n: (counts.get(n, 0) + 1) / (total + TOTAL_NUMBERS) 
                     for n in range(TOTAL_NUMBERS)}
            
            # Simulate next spin
            simulated_counts = Counter()
            prob_values = list(probs.values())
            prob_sum = sum(prob_values)
            normalized_probs = [p / prob_sum for p in prob_values]
            
            for _ in range(self.monte_carlo_iterations):
                result = np.random.choice(TOTAL_NUMBERS, p=normalized_probs)
                simulated_counts[result] += 1
            
            # Top 10 simulated
            top_simulated = simulated_counts.most_common(10)
            top_simulated = [(n, c / self.monte_carlo_iterations) for n, c in top_simulated]
            
            return {
                'type': 'number',
                'top_simulated': [{'number': n, 'probability': p} for n, p in top_simulated],
                'iterations': self.monte_carlo_iterations,
            }
    
    def markov_chain_analysis(self, numbers: list[int]) -> Dict[str, Any]:
        """
        Analyze transition probabilities using Markov chain analysis.
        
        Models the probability of transitioning from one outcome to another.
        """
        if len(numbers) < 20:
            return {'error': 'Insufficient data for Markov analysis'}
        
        # Color transitions
        colors = [number_to_color(n) for n in numbers]
        color_transitions = {}
        
        for i in range(len(colors) - 1):
            from_color = colors[i]
            to_color = colors[i + 1]
            
            if from_color not in color_transitions:
                color_transitions[from_color] = Counter()
            color_transitions[from_color][to_color] += 1
        
        # Normalize to probabilities
        color_matrix = {}
        for from_color, to_counts in color_transitions.items():
            total = sum(to_counts.values())
            color_matrix[from_color] = {
                to_color: count / total 
                for to_color, count in to_counts.items()
            }
        
        # Dozens transitions
        def get_dozen(n):
            if n == 0: return 'zero'
            if n <= 12: return 'first'
            if n <= 24: return 'second'
            return 'third'
        
        dozens = [get_dozen(n) for n in numbers]
        dozen_transitions = {}
        
        for i in range(len(dozens) - 1):
            from_dozen = dozens[i]
            to_dozen = dozens[i + 1]
            
            if from_dozen not in dozen_transitions:
                dozen_transitions[from_dozen] = Counter()
            dozen_transitions[from_dozen][to_dozen] += 1
        
        dozen_matrix = {}
        for from_dozen, to_counts in dozen_transitions.items():
            total = sum(to_counts.values())
            dozen_matrix[from_dozen] = {
                to_dozen: count / total 
                for to_dozen, count in to_counts.items()
            }
        
        # Steady state (for analysis)
        last_color = colors[-1] if colors else 'red'
        if last_color in color_matrix:
            next_color_probs = color_matrix[last_color]
        else:
            next_color_probs = {'red': 18/37, 'black': 18/37, 'green': 1/37}
        
        return {
            'color_transition_matrix': color_matrix,
            'dozen_transition_matrix': dozen_matrix,
            'current_state': {
                'last_color': last_color,
                'last_dozen': get_dozen(numbers[-1]) if numbers else None,
            },
            'next_color_prediction': next_color_probs,
            'chain_length': len(numbers),
        }
    
    def bayesian_update(self, numbers: list[int], 
                        prior: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Bayesian probability update for color prediction.
        
        Uses observed data to update prior beliefs about color probabilities.
        
        Args:
            numbers: Observed spins
            prior: Prior probabilities (default: uniform)
        """
        if prior is None:
            # Use theoretical as prior
            prior = {'red': 18/37, 'black': 18/37, 'green': 1/37}
        
        if not numbers:
            return {
                'prior': prior,
                'posterior': prior,
                'update_factor': {'red': 1.0, 'black': 1.0, 'green': 1.0},
            }
        
        # Count observations
        colors = [number_to_color(n) for n in numbers]
        counts = Counter(colors)
        total = len(colors)
        
        # Bayesian update (using Beta-Binomial conjugate)
        # For simplicity, use pseudo-counts
        alpha_prior = 10  # Strength of prior
        
        posterior = {}
        update_factor = {}
        
        for color in ['red', 'black', 'green']:
            observed = counts.get(color, 0)
            prior_count = prior[color] * alpha_prior
            
            # Posterior = (prior_count + observed) / (alpha_prior + total)
            posterior[color] = (prior_count + observed) / (alpha_prior + total)
            update_factor[color] = posterior[color] / prior[color]
        
        # Normalize
        total_posterior = sum(posterior.values())
        posterior = {c: p / total_posterior for c, p in posterior.items()}
        
        # Confidence (based on sample size)
        confidence = min(1.0, total / 100)  # Max confidence at 100 spins
        
        return {
            'prior': prior,
            'posterior': posterior,
            'update_factor': update_factor,
            'confidence': confidence,
            'observations': total,
            'predicted_color': max(posterior, key=posterior.get),
        }
    
    def calculate_entropy(self, numbers: list[int]) -> Dict[str, Any]:
        """
        Calculate entropy to measure randomness.
        
        Higher entropy = more random
        Lower entropy = possible patterns or biases
        """
        if len(numbers) < 10:
            return {
                'number_entropy': 0,
                'color_entropy': 0,
                'normalized_entropy': 0,
                'randomness_score': 0,
            }
        
        # Number entropy
        counts = Counter(numbers)
        total = len(numbers)
        probs = [c / total for c in counts.values()]
        number_entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in probs)
        max_number_entropy = np.log2(TOTAL_NUMBERS)
        
        # Color entropy
        colors = [number_to_color(n) for n in numbers]
        color_counts = Counter(colors)
        color_probs = [c / total for c in color_counts.values()]
        color_entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in color_probs)
        max_color_entropy = np.log2(3)  # 3 colors
        
        # Normalized (0-1, where 1 = maximum randomness)
        normalized_number = number_entropy / max_number_entropy
        normalized_color = color_entropy / max_color_entropy
        
        # Overall randomness score (weighted average)
        randomness_score = 0.7 * normalized_number + 0.3 * normalized_color
        
        return {
            'number_entropy': float(number_entropy),
            'color_entropy': float(color_entropy),
            'max_number_entropy': float(max_number_entropy),
            'max_color_entropy': float(max_color_entropy),
            'normalized_number_entropy': float(normalized_number),
            'normalized_color_entropy': float(normalized_color),
            'randomness_score': float(randomness_score),
            'interpretation': self._interpret_randomness(randomness_score),
        }
    
    def _interpret_randomness(self, score: float) -> str:
        if score > 0.95:
            return "Highly random - no detectable patterns"
        elif score > 0.85:
            return "Random - slight variations from uniform"
        elif score > 0.70:
            return "Somewhat random - minor non-uniform patterns"
        elif score > 0.50:
            return "Low randomness - possible biases detected"
        else:
            return "Very low randomness - significant biases present"
    
    def runs_test(self, numbers: list[int]) -> Dict[str, Any]:
        """
        Runs test for randomness (Wald-Wolfowitz).
        
        Tests whether the sequence has too many or too few runs
        (consecutive sequences of same category).
        """
        if len(numbers) < 20:
            return {'error': 'Insufficient data for runs test'}
        
        # Convert to binary: above/below median
        median = np.median(numbers)
        binary = [1 if n > median else 0 for n in numbers]
        
        # Count runs
        runs = 1
        for i in range(1, len(binary)):
            if binary[i] != binary[i-1]:
                runs += 1
        
        # Expected runs and variance
        n1 = sum(binary)
        n2 = len(binary) - n1
        n = n1 + n2
        
        if n1 == 0 or n2 == 0:
            return {'error': 'All values on one side of median'}
        
        expected_runs = (2 * n1 * n2 / n) + 1
        variance = (2 * n1 * n2 * (2 * n1 * n2 - n)) / (n * n * (n - 1))
        
        if variance <= 0:
            return {'error': 'Invalid variance calculation'}
        
        z_score = (runs - expected_runs) / np.sqrt(variance)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        return {
            'runs_observed': runs,
            'runs_expected': float(expected_runs),
            'z_score': float(z_score),
            'p_value': float(p_value),
            'random': p_value > 0.05,
            'interpretation': 'Sequence appears random' if p_value > 0.05 else 'Sequence shows non-random patterns',
        }
    
    def get_full_analysis(self, numbers: list[int]) -> Dict[str, Any]:
        """
        Run all statistical analyses and return comprehensive results.
        """
        chi2_number = self.chi_squared_test(numbers)
        chi2_color = self.chi_squared_color_test(numbers)
        monte_carlo_color = self.monte_carlo_simulation(numbers, 'color')
        monte_carlo_number = self.monte_carlo_simulation(numbers, 'number')
        markov = self.markov_chain_analysis(numbers)
        bayesian = self.bayesian_update(numbers)
        entropy = self.calculate_entropy(numbers)
        runs = self.runs_test(numbers)
        
        # Overall assessment
        bias_indicators = 0
        if chi2_number.get('significant', False):
            bias_indicators += 1
        if chi2_color.get('significant', False):
            bias_indicators += 1
        if not runs.get('random', True):
            bias_indicators += 1
        if entropy.get('randomness_score', 1) < 0.7:
            bias_indicators += 1
        
        if bias_indicators >= 3:
            overall = 'Strong evidence of non-random behavior'
        elif bias_indicators >= 2:
            overall = 'Some evidence of patterns or biases'
        elif bias_indicators >= 1:
            overall = 'Minor deviations from pure randomness'
        else:
            overall = 'Outcomes appear random as expected'
        
        return {
            'chi_squared': {
                'number': chi2_number,
                'color': chi2_color,
            },
            'monte_carlo': {
                'color': monte_carlo_color,
                'number': monte_carlo_number,
            },
            'markov': markov,
            'bayesian': bayesian,
            'entropy': entropy,
            'runs_test': runs,
            'overall_assessment': overall,
            'bias_indicators': bias_indicators,
            'total_spins': len(numbers),
        }


class WheelClusteringAnalyzer:
    """
    Clustering-based analysis to detect wheel biases and anomalies.
    
    Uses unsupervised learning to:
    - Detect numbers that appear together (sector clustering)
    - Identify anomalous patterns suggesting wheel defects
    - Find number groupings that deviate from random
    """
    
    # European roulette wheel layout (physical order on wheel)
    WHEEL_ORDER = [
        0, 32, 15, 19, 4, 21, 2, 25, 17, 34, 6, 27, 13, 36, 11, 30, 8, 23, 10,
        5, 24, 16, 33, 1, 20, 14, 31, 9, 22, 18, 29, 7, 28, 12, 35, 3, 26
    ]
    
    def __init__(self):
        # Build position lookup: number -> position on wheel
        self.number_to_position = {
            num: idx for idx, num in enumerate(self.WHEEL_ORDER)
        }
        
    def get_wheel_distance(self, num1: int, num2: int) -> int:
        """Calculate the distance between two numbers on the physical wheel."""
        pos1 = self.number_to_position.get(num1, 0)
        pos2 = self.number_to_position.get(num2, 0)
        
        # Wheel is circular, take minimum distance
        forward = abs(pos2 - pos1)
        backward = 37 - forward
        return min(forward, backward)
    
    def analyze_sector_clustering(self, numbers: List[int]) -> Dict[str, Any]:
        """
        Analyze if numbers cluster in specific wheel sectors.
        
        A biased wheel might produce numbers from the same physical
        sector more often than expected.
        """
        if len(numbers) < 30:
            return {'status': 'insufficient_data', 'min_required': 30}
        
        # Calculate consecutive wheel distances
        distances = []
        for i in range(1, len(numbers)):
            dist = self.get_wheel_distance(numbers[i-1], numbers[i])
            distances.append(dist)
        
        # If consecutive numbers are often close on wheel = possible bias
        avg_distance = np.mean(distances)
        expected_avg = 37 / 4  # ~9.25 for random distribution
        
        # Standard deviation of distances
        std_distance = np.std(distances)
        
        # Cluster score: how much do numbers cluster vs random
        # Lower than expected = clustering (bias)
        cluster_score = avg_distance / expected_avg
        
        # Find "hot zones" on wheel (sectors with many hits)
        position_counts = Counter()
        for num in numbers:
            pos = self.number_to_position.get(num, 0)
            # Group into 6 sectors of ~6 numbers each
            sector = pos // 6
            position_counts[sector] += 1
        
        # Check if any sector is significantly overrepresented
        expected_per_sector = len(numbers) / 6
        hot_sectors = []
        cold_sectors = []
        
        for sector in range(6):
            count = position_counts.get(sector, 0)
            deviation = (count - expected_per_sector) / expected_per_sector
            if deviation > 0.3:  # 30% above expected
                hot_sectors.append({
                    'sector': sector,
                    'count': count,
                    'deviation': f"+{deviation*100:.1f}%"
                })
            elif deviation < -0.3:  # 30% below expected
                cold_sectors.append({
                    'sector': sector,
                    'count': count,
                    'deviation': f"{deviation*100:.1f}%"
                })
        
        # Interpretation
        if cluster_score < 0.7:
            interpretation = "Strong clustering detected - possible wheel bias"
            bias_likely = True
        elif cluster_score < 0.85:
            interpretation = "Moderate clustering - monitor for patterns"
            bias_likely = False
        else:
            interpretation = "Normal distribution - no significant clustering"
            bias_likely = False
        
        return {
            'cluster_score': float(cluster_score),
            'avg_wheel_distance': float(avg_distance),
            'expected_distance': float(expected_avg),
            'distance_std': float(std_distance),
            'hot_sectors': hot_sectors,
            'cold_sectors': cold_sectors,
            'bias_likely': bias_likely,
            'interpretation': interpretation,
            'sample_size': len(numbers)
        }
    
    def detect_number_pairs(self, numbers: List[int], 
                           min_occurrences: int = 5) -> Dict[str, Any]:
        """
        Detect pairs of numbers that appear consecutively more often than expected.
        
        This can indicate wheel defects where certain number transitions
        are more likely due to physical imperfections.
        """
        if len(numbers) < 50:
            return {'status': 'insufficient_data'}
        
        # Count pair occurrences
        pair_counts = Counter()
        for i in range(1, len(numbers)):
            pair = (numbers[i-1], numbers[i])
            pair_counts[pair] += 1
        
        # Expected frequency for any pair
        expected = len(numbers) / (37 * 37)
        
        # Find overrepresented pairs
        suspicious_pairs = []
        for pair, count in pair_counts.most_common(20):
            if count >= min_occurrences:
                wheel_dist = self.get_wheel_distance(pair[0], pair[1])
                significance = count / max(expected, 0.1)
                suspicious_pairs.append({
                    'from': pair[0],
                    'to': pair[1],
                    'count': count,
                    'wheel_distance': wheel_dist,
                    'significance': f"{significance:.1f}x expected"
                })
        
        return {
            'suspicious_pairs': suspicious_pairs[:10],
            'total_pairs_analyzed': len(pair_counts),
            'sample_size': len(numbers)
        }
    
    def detect_sleeper_anomalies(self, numbers: List[int]) -> Dict[str, Any]:
        """
        Detect anomalies in sleeper patterns (numbers not appearing).
        
        A biased wheel might have some numbers that rarely appear.
        """
        if len(numbers) < 100:
            return {'status': 'insufficient_data', 'min_required': 100}
        
        counts = Counter(numbers)
        
        # Expected count for each number
        expected = len(numbers) / 37
        
        # Find significantly underrepresented numbers
        cold_numbers = []
        hot_numbers = []
        
        for num in range(37):
            count = counts.get(num, 0)
            deviation = (count - expected) / max(expected, 1)
            
            if count < expected * 0.3:  # Less than 30% of expected
                cold_numbers.append({
                    'number': num,
                    'count': count,
                    'expected': round(expected, 1),
                    'wheel_position': self.number_to_position.get(num, 0)
                })
            elif count > expected * 2:  # More than 200% of expected
                hot_numbers.append({
                    'number': num,
                    'count': count,
                    'expected': round(expected, 1),
                    'wheel_position': self.number_to_position.get(num, 0)
                })
        
        # Check if cold/hot numbers are clustered on wheel
        cold_positions = [self.number_to_position.get(n['number'], 0) for n in cold_numbers]
        hot_positions = [self.number_to_position.get(n['number'], 0) for n in hot_numbers]
        
        wheel_bias_indicator = False
        if len(cold_positions) >= 2 or len(hot_positions) >= 2:
            # Check if they're close on wheel
            if len(cold_positions) >= 2:
                cold_spread = max(cold_positions) - min(cold_positions)
                if cold_spread < 10:  # Clustered cold zone
                    wheel_bias_indicator = True
            if len(hot_positions) >= 2:
                hot_spread = max(hot_positions) - min(hot_positions)
                if hot_spread < 10:  # Clustered hot zone
                    wheel_bias_indicator = True
        
        return {
            'cold_numbers': cold_numbers,
            'hot_numbers': hot_numbers,
            'wheel_bias_indicator': wheel_bias_indicator,
            'sample_size': len(numbers),
            'expected_per_number': round(expected, 1)
        }
    
    def get_full_clustering_analysis(self, numbers: List[int]) -> Dict[str, Any]:
        """Run all clustering analyses and return comprehensive results."""
        sector_analysis = self.analyze_sector_clustering(numbers)
        pair_analysis = self.detect_number_pairs(numbers)
        sleeper_analysis = self.detect_sleeper_anomalies(numbers)
        
        # Overall wheel health assessment
        bias_indicators = 0
        if sector_analysis.get('bias_likely', False):
            bias_indicators += 2
        if len(sector_analysis.get('hot_sectors', [])) >= 2:
            bias_indicators += 1
        if sleeper_analysis.get('wheel_bias_indicator', False):
            bias_indicators += 2
        if len(sleeper_analysis.get('hot_numbers', [])) >= 3:
            bias_indicators += 1
        
        if bias_indicators >= 4:
            wheel_assessment = "High probability of wheel bias - exploit patterns"
            exploitable = True
        elif bias_indicators >= 2:
            wheel_assessment = "Possible minor bias - continue monitoring"
            exploitable = False
        else:
            wheel_assessment = "Wheel appears fair and random"
            exploitable = False
        
        return {
            'sector_clustering': sector_analysis,
            'pair_analysis': pair_analysis,
            'sleeper_anomalies': sleeper_analysis,
            'wheel_assessment': wheel_assessment,
            'bias_indicators': bias_indicators,
            'exploitable': exploitable,
            'total_spins_analyzed': len(numbers)
        }


# Singletons
_statistical_engine: Optional[StatisticalEngine] = None
_wheel_clustering: Optional[WheelClusteringAnalyzer] = None


def get_statistical_engine() -> StatisticalEngine:
    global _statistical_engine
    if _statistical_engine is None:
        _statistical_engine = StatisticalEngine()
    return _statistical_engine


def get_wheel_clustering_analyzer() -> WheelClusteringAnalyzer:
    global _wheel_clustering
    if _wheel_clustering is None:
        _wheel_clustering = WheelClusteringAnalyzer()
    return _wheel_clustering
