"""
Pattern Detection Algorithms for Roulette Analysis.

Detects various patterns in spin sequences:
- Streak patterns (hot/cold runs)
- Wheel bias analysis
- Gap analysis (sleeper numbers)
- Cluster detection
- Martingale pattern detection
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter, deque
from datetime import datetime

from roulette import number_to_color, RED_NUMBERS, TOTAL_NUMBERS
from advanced_features import (
    WHEEL_ORDER, VOISINS_DU_ZERO, TIERS_DU_CYLINDRE, ORPHELINS,
    DOZEN_1, DOZEN_2, DOZEN_3, COLUMN_1, COLUMN_2, COLUMN_3,
    LOW_NUMBERS, HIGH_NUMBERS, EVEN_NUMBERS, ODD_NUMBERS
)


class PatternDetector:
    """
    Comprehensive pattern detection for roulette analysis.
    
    Analyzes spin sequences to identify:
    - Hot numbers (appearing more than expected)
    - Cold numbers (not appearing as expected)
    - Sleeper numbers (overdue)
    - Streak patterns
    - Wheel sector biases
    - Cluster patterns
    """
    
    def __init__(self):
        self.alert_threshold = 1.5  # Standard deviations for alerts
        
    def detect_hot_cold_numbers(self, numbers: list[int], 
                                 window: int = 100) -> Dict[str, Any]:
        """
        Identify hot and cold numbers based on frequency analysis.
        
        Returns:
            Dictionary with hot numbers, cold numbers, and detailed stats
        """
        if len(numbers) < 10:
            return {
                'hot': [], 'cold': [], 'neutral': list(range(TOTAL_NUMBERS)),
                'frequency_map': {}, 'total_spins': 0
            }
        
        recent = numbers[-window:] if len(numbers) > window else numbers
        total = len(recent)
        expected = total / TOTAL_NUMBERS
        
        counts = Counter(recent)
        frequency_map = {}
        hot = []
        cold = []
        neutral = []
        
        for n in range(TOTAL_NUMBERS):
            actual = counts.get(n, 0)
            frequency = actual / total
            deviation = (actual - expected) / max(np.sqrt(expected), 0.1)
            
            frequency_map[n] = {
                'count': actual,
                'frequency': frequency,
                'expected': expected / total,
                'deviation': deviation,
                'status': 'neutral'
            }
            
            if deviation > self.alert_threshold:
                hot.append((n, deviation, actual))
                frequency_map[n]['status'] = 'hot'
            elif deviation < -self.alert_threshold:
                cold.append((n, deviation, actual))
                frequency_map[n]['status'] = 'cold'
            else:
                neutral.append(n)
        
        # Sort by deviation
        hot.sort(key=lambda x: -x[1])
        cold.sort(key=lambda x: x[1])
        
        return {
            'hot': [{'number': n, 'deviation': d, 'count': c} for n, d, c in hot[:10]],
            'cold': [{'number': n, 'deviation': d, 'count': c} for n, d, c in cold[:10]],
            'neutral_count': len(neutral),
            'frequency_map': frequency_map,
            'total_spins': total,
            'expected_frequency': expected / total,
        }
    
    def detect_sleepers(self, numbers: list[int]) -> Dict[str, Any]:
        """
        Find "sleeper" numbers - those that haven't appeared for a long time.
        
        A number is considered a sleeper if its gap exceeds 1.5x expected gap (37 spins).
        """
        if not numbers:
            return {'sleepers': [], 'gaps': {}, 'max_gap': 0}
        
        expected_gap = TOTAL_NUMBERS
        gaps = {}
        sleepers = []
        
        for n in range(TOTAL_NUMBERS):
            # Find last occurrence
            try:
                last_idx = len(numbers) - 1 - numbers[::-1].index(n)
                gap = len(numbers) - 1 - last_idx
            except ValueError:
                gap = len(numbers)  # Never appeared
            
            gaps[n] = {
                'gap': gap,
                'percentage_of_expected': gap / expected_gap * 100,
                'is_sleeper': gap > expected_gap * 1.5,
                'is_overdue': gap > expected_gap * 2,
            }
            
            if gaps[n]['is_sleeper']:
                sleepers.append({
                    'number': n,
                    'gap': gap,
                    'overdue_by': gap - expected_gap,
                    'urgency': 'high' if gap > expected_gap * 2 else 'medium'
                })
        
        sleepers.sort(key=lambda x: -x['gap'])
        
        return {
            'sleepers': sleepers[:10],
            'gaps': gaps,
            'max_gap': max(g['gap'] for g in gaps.values()) if gaps else 0,
            'max_gap_number': max(gaps.items(), key=lambda x: x[1]['gap'])[0] if gaps else None,
            'total_sleepers': len(sleepers),
            'expected_gap': expected_gap,
        }
    
    def detect_streaks(self, numbers: list[int]) -> Dict[str, Any]:
        """
        Detect streak patterns in the spin sequence.
        
        Analyzes:
        - Color streaks
        - Dozen streaks
        - Column streaks
        - Even/Odd streaks
        - High/Low streaks
        """
        if len(numbers) < 2:
            return {'current_streaks': {}, 'max_streaks': {}, 'pattern_alerts': []}
        
        # Current streaks
        current = {
            'color': self._calculate_current_streak(numbers, 'color'),
            'dozen': self._calculate_current_streak(numbers, 'dozen'),
            'column': self._calculate_current_streak(numbers, 'column'),
            'parity': self._calculate_current_streak(numbers, 'parity'),
            'high_low': self._calculate_current_streak(numbers, 'high_low'),
        }
        
        # Maximum historical streaks
        max_streaks = {
            'color': self._calculate_max_streak(numbers, 'color'),
            'dozen': self._calculate_max_streak(numbers, 'dozen'),
            'column': self._calculate_max_streak(numbers, 'column'),
            'parity': self._calculate_max_streak(numbers, 'parity'),
            'high_low': self._calculate_max_streak(numbers, 'high_low'),
        }
        
        # Pattern alerts
        alerts = []
        for pattern_type, streak_info in current.items():
            if streak_info['length'] >= 5:
                alerts.append({
                    'type': pattern_type,
                    'value': streak_info['value'],
                    'length': streak_info['length'],
                    'severity': 'high' if streak_info['length'] >= 8 else 'medium',
                    'message': f"{streak_info['length']} consecutive {streak_info['value']} in {pattern_type}"
                })
        
        return {
            'current_streaks': current,
            'max_streaks': max_streaks,
            'pattern_alerts': alerts,
        }
    
    def _calculate_current_streak(self, numbers: list[int], 
                                   pattern_type: str) -> Dict[str, Any]:
        """Calculate current streak for a pattern type."""
        if not numbers:
            return {'length': 0, 'value': None}
        
        values = self._get_pattern_values(numbers, pattern_type)
        if not values:
            return {'length': 0, 'value': None}
        
        current_value = values[-1]
        streak = 1
        
        for i in range(len(values) - 2, -1, -1):
            if values[i] == current_value:
                streak += 1
            else:
                break
        
        return {'length': streak, 'value': current_value}
    
    def _calculate_max_streak(self, numbers: list[int], 
                               pattern_type: str) -> Dict[str, Any]:
        """Calculate maximum streak for a pattern type."""
        values = self._get_pattern_values(numbers, pattern_type)
        if len(values) < 2:
            return {'length': len(values), 'value': values[0] if values else None}
        
        max_streak = 1
        max_value = values[0]
        current_streak = 1
        
        for i in range(1, len(values)):
            if values[i] == values[i-1]:
                current_streak += 1
                if current_streak > max_streak:
                    max_streak = current_streak
                    max_value = values[i]
            else:
                current_streak = 1
        
        return {'length': max_streak, 'value': max_value}
    
    def _get_pattern_values(self, numbers: list[int], pattern_type: str) -> list:
        """Convert numbers to pattern values."""
        if pattern_type == 'color':
            return [number_to_color(n) for n in numbers]
        elif pattern_type == 'dozen':
            return [self._get_dozen(n) for n in numbers]
        elif pattern_type == 'column':
            return [self._get_column(n) for n in numbers]
        elif pattern_type == 'parity':
            return [self._get_parity(n) for n in numbers]
        elif pattern_type == 'high_low':
            return [self._get_high_low(n) for n in numbers]
        return []
    
    def _get_dozen(self, n: int) -> str:
        if n == 0: return 'zero'
        if n <= 12: return 'first'
        if n <= 24: return 'second'
        return 'third'
    
    def _get_column(self, n: int) -> str:
        if n == 0: return 'zero'
        return ['first', 'second', 'third'][(n - 1) % 3]
    
    def _get_parity(self, n: int) -> str:
        if n == 0: return 'zero'
        return 'even' if n % 2 == 0 else 'odd'
    
    def _get_high_low(self, n: int) -> str:
        if n == 0: return 'zero'
        return 'high' if n >= 19 else 'low'
    
    def detect_sector_bias(self, numbers: list[int], 
                           window: int = 200) -> Dict[str, Any]:
        """
        Detect wheel sector biases that might indicate physical wheel defects.
        
        Analyzes distribution across:
        - Voisins du Zero (numbers around 0 on wheel)
        - Tiers du Cylindre (third of wheel opposite zero)
        - Orphelins (remaining numbers)
        """
        if len(numbers) < 50:
            return {'bias_detected': False, 'sectors': {}, 'message': 'Need more data'}
        
        recent = numbers[-window:] if len(numbers) > window else numbers
        total = len(recent)
        
        sectors = {
            'voisins': {
                'numbers': VOISINS_DU_ZERO,
                'expected': len(VOISINS_DU_ZERO) / TOTAL_NUMBERS,
                'count': sum(1 for n in recent if n in VOISINS_DU_ZERO),
            },
            'tiers': {
                'numbers': TIERS_DU_CYLINDRE,
                'expected': len(TIERS_DU_CYLINDRE) / TOTAL_NUMBERS,
                'count': sum(1 for n in recent if n in TIERS_DU_CYLINDRE),
            },
            'orphelins': {
                'numbers': ORPHELINS,
                'expected': len(ORPHELINS) / TOTAL_NUMBERS,
                'count': sum(1 for n in recent if n in ORPHELINS),
            },
        }
        
        bias_alerts = []
        bias_detected = False
        
        for sector_name, data in sectors.items():
            actual = data['count'] / total
            expected = data['expected']
            deviation = (actual - expected) / max(np.sqrt(expected * (1 - expected) / total), 0.01)
            
            data['actual'] = actual
            data['deviation'] = deviation
            data['bias_level'] = 'none'
            
            if abs(deviation) > 2:
                bias_detected = True
                data['bias_level'] = 'significant' if abs(deviation) > 3 else 'moderate'
                direction = 'high' if deviation > 0 else 'low'
                bias_alerts.append({
                    'sector': sector_name,
                    'deviation': deviation,
                    'direction': direction,
                    'message': f"{sector_name} sector showing {data['bias_level']} {direction} frequency"
                })
        
        # Physical wheel position clustering
        wheel_positions = [WHEEL_ORDER.index(n) for n in recent if n in WHEEL_ORDER]
        if wheel_positions:
            position_std = np.std(wheel_positions)
            position_mean = np.mean(wheel_positions)
            # Low std might indicate wheel bias toward a sector
            expected_std = len(WHEEL_ORDER) / np.sqrt(12)  # Uniform distribution std
            clustering = 1 - (position_std / expected_std)
        else:
            clustering = 0
        
        return {
            'bias_detected': bias_detected,
            'sectors': sectors,
            'alerts': bias_alerts,
            'wheel_clustering': clustering,
            'total_analyzed': total,
        }
    
    def detect_all_patterns(self, numbers: list[int]) -> Dict[str, Any]:
        """
        Run all pattern detection algorithms and return comprehensive results.
        """
        hot_cold = self.detect_hot_cold_numbers(numbers)
        sleepers = self.detect_sleepers(numbers)
        streaks = self.detect_streaks(numbers)
        sector_bias = self.detect_sector_bias(numbers)
        
        # Combine all alerts
        all_alerts = []
        all_alerts.extend(streaks.get('pattern_alerts', []))
        all_alerts.extend(sector_bias.get('alerts', []))
        
        # Add hot number alerts
        for hot in hot_cold.get('hot', [])[:3]:
            all_alerts.append({
                'type': 'hot_number',
                'number': hot['number'],
                'severity': 'medium',
                'message': f"Number {hot['number']} is hot (deviation: {hot['deviation']:.2f})"
            })
        
        # Add sleeper alerts
        for sleeper in sleepers.get('sleepers', [])[:3]:
            if sleeper.get('urgency') == 'high':
                all_alerts.append({
                    'type': 'sleeper',
                    'number': sleeper['number'],
                    'severity': 'high',
                    'message': f"Number {sleeper['number']} hasn't appeared in {sleeper['gap']} spins"
                })
        
        # Generate pattern-based predictions with agreement
        pattern_predictions = self._generate_pattern_predictions(
            numbers, hot_cold, sleepers, streaks, sector_bias
        )
        
        return {
            'hot_cold': hot_cold,
            'sleepers': sleepers,
            'streaks': streaks,
            'sector_bias': sector_bias,
            'alerts': all_alerts,
            'alert_count': len(all_alerts),
            'analyzed_spins': len(numbers),
            'pattern_predictions': pattern_predictions,
        }
    
    def _generate_pattern_predictions(
        self,
        numbers: list[int],
        hot_cold: Dict,
        sleepers: Dict,
        streaks: Dict,
        sector_bias: Dict,
    ) -> list[Dict[str, Any]]:
        """
        Generate concrete predictions from all pattern sources.
        
        Each prediction has:
        - type: 'number', 'color', 'sector', 'area'
        - value: the predicted value
        - probability: base probability from pattern analysis
        - overdue_boost: extra probability from gap analysis
        - final_probability: probability + overdue_boost
        - sources: list of pattern sources that agree
        - agreement: len(sources) / total_sources
        - description: human-readable description
        """
        # Collect all number-level predictions from different sources
        number_scores: Dict[int, Dict] = {}
        TOTAL_SOURCES = 4  # hot, sleeper, streak_pattern, sector
        
        # --- Source 1: Hot Numbers ---
        for hot in hot_cold.get('hot', []):
            n = hot['number']
            if n not in number_scores:
                number_scores[n] = {'sources': [], 'base_prob': 0, 'overdue_boost': 0}
            # Probability based on deviation (higher deviation = more likely to continue)
            prob = min(0.15 + hot['deviation'] * 0.02, 0.5)
            number_scores[n]['base_prob'] = max(number_scores[n]['base_prob'], prob)
            number_scores[n]['sources'].append('🔥 Numero Caldo')
        
        # --- Source 2: Sleepers (with overdue boost!) ---
        expected_gap = TOTAL_NUMBERS  # 37 spins expected gap
        for sleeper in sleepers.get('sleepers', []):
            n = sleeper['number']
            if n not in number_scores:
                number_scores[n] = {'sources': [], 'base_prob': 0, 'overdue_boost': 0}
            
            gap = sleeper['gap']
            # Base probability for sleeper
            base = 1.0 / TOTAL_NUMBERS  # ~2.7%
            
            # OVERDUE BOOST: the longer a number is absent, the higher the boost
            # Using a logarithmic curve so it doesn't go to infinity
            overdue_ratio = gap / expected_gap
            if overdue_ratio > 1.5:
                # Boost increases with gap but caps out
                boost = min(0.08 * np.log(overdue_ratio), 0.25)
                number_scores[n]['overdue_boost'] = boost
            
            number_scores[n]['base_prob'] = max(number_scores[n]['base_prob'], base)
            number_scores[n]['sources'].append(f'💤 Ritardatario ({gap} giri)')
        
        # --- Source 3: Sector Bias ---
        sector_numbers_map = {
            'voisins': VOISINS_DU_ZERO,
            'tiers': TIERS_DU_CYLINDRE,
            'orphelins': ORPHELINS,
        }
        sector_labels = {
            'voisins': '🎡 Voisins du Zero',
            'tiers': '🎡 Tiers du Cylindre',
            'orphelins': '🎡 Orphelins',
        }
        
        for sector_name, data in sector_bias.get('sectors', {}).items():
            if data.get('deviation', 0) > 1.0:  # Hot sector
                sector_nums = sector_numbers_map.get(sector_name, [])
                sector_label = sector_labels.get(sector_name, sector_name)
                for n in sector_nums:
                    if n not in number_scores:
                        number_scores[n] = {'sources': [], 'base_prob': 0, 'overdue_boost': 0}
                    prob = data.get('actual', 0) * 0.5  # Scale sector probability
                    number_scores[n]['base_prob'] = max(number_scores[n]['base_prob'], prob)
                    # Avoid duplicate sector source
                    if not any(sector_name in s for s in number_scores[n]['sources']):
                        number_scores[n]['sources'].append(sector_label)
        
        # --- Source 4: Check if numbers are also in cold/sleeper AND hot sector (cross-pattern) ---
        # Numbers that are both sleeper AND in a hot sector get an extra source
        for n, data in number_scores.items():
            has_sleeper = any('Ritardatario' in s for s in data['sources'])
            has_sector = any('🎡' in s for s in data['sources'])
            if has_sleeper and has_sector:
                data['sources'].append('⚡ Cross-Pattern')
        
        # --- Build final predictions ---
        predictions = []
        for n, data in number_scores.items():
            if not data['sources']:
                continue
            
            final_prob = data['base_prob'] + data['overdue_boost']
            final_prob = min(final_prob, 0.5)  # Cap at 50%
            
            agreement = len(data['sources']) / TOTAL_SOURCES
            
            predictions.append({
                'type': 'number',
                'value': n,
                'label': f'Numero {n}',
                'probability': round(final_prob, 4),
                'overdue_boost': round(data['overdue_boost'], 4),
                'base_probability': round(data['base_prob'], 4),
                'sources': data['sources'],
                'source_count': len(data['sources']),
                'agreement': round(min(agreement, 1.0), 4),
                'description': ' + '.join(data['sources']),
            })
        
        # --- Area/Streak Predictions ---
        for pattern_type, streak in streaks.get('current_streaks', {}).items():
            if streak.get('length', 0) >= 3:
                # Active streak = predict continuation
                prob = min(0.35 + streak['length'] * 0.03, 0.65)
                
                label_map = {
                    'color': 'Colore',
                    'dozen': 'Dozzina',
                    'column': 'Colonna',
                    'parity': 'Pari/Dispari',
                    'high_low': 'Alto/Basso',
                }
                
                value_map = {
                    'red': 'Rosso', 'black': 'Nero',
                    'first': '1ª', 'second': '2ª', 'third': '3ª',
                    'even': 'Pari', 'odd': 'Dispari',
                    'high': 'Alto', 'low': 'Basso',
                }
                
                friendly_value = value_map.get(streak['value'], streak['value'])
                
                predictions.append({
                    'type': 'streak',
                    'value': friendly_value,
                    'label': label_map.get(pattern_type, pattern_type),
                    'probability': round(prob, 4),
                    'overdue_boost': 0,
                    'base_probability': round(prob, 4),
                    'sources': [f'🌊 Serie di {streak["length"]}'],
                    'source_count': 1,
                    'agreement': round(1 / TOTAL_SOURCES, 4),
                    'description': f'{friendly_value} - {streak["length"]} di fila',
                })
        
        # --- Classic Category Overdue Boost ---
        # For each category type, check how many spins since each value last appeared.
        # If overdue (gap > threshold), create a prediction with boost.
        category_configs = {
            'color': {
                'values': {'red': RED_NUMBERS, 'black': [n for n in range(1, TOTAL_NUMBERS) if n not in RED_NUMBERS and n != 0]},
                'expected_gap': TOTAL_NUMBERS / 18,   # ~2.06 spins
                'overdue_mult': 3.0,                  # overdue if gap > 6 spins
                'label': 'Colore',
                'friendly': {'red': 'Rosso 🔴', 'black': 'Nero ⚫'},
                'icon': '🎨',
            },
            'dozen': {
                'values': {'1ª Dozzina': DOZEN_1, '2ª Dozzina': DOZEN_2, '3ª Dozzina': DOZEN_3},
                'expected_gap': TOTAL_NUMBERS / 12,   # ~3.08 spins
                'overdue_mult': 3.0,                  # overdue if gap > 9 spins
                'label': 'Dozzina',
                'friendly': {'1ª Dozzina': '1ª (1-12)', '2ª Dozzina': '2ª (13-24)', '3ª Dozzina': '3ª (25-36)'},
                'icon': '📊',
            },
            'column': {
                'values': {'1ª Colonna': COLUMN_1, '2ª Colonna': COLUMN_2, '3ª Colonna': COLUMN_3},
                'expected_gap': TOTAL_NUMBERS / 12,
                'overdue_mult': 3.0,
                'label': 'Colonna',
                'friendly': {'1ª Colonna': '1ª Col.', '2ª Colonna': '2ª Col.', '3ª Colonna': '3ª Col.'},
                'icon': '📊',
            },
            'parity': {
                'values': {'even': EVEN_NUMBERS, 'odd': ODD_NUMBERS},
                'expected_gap': TOTAL_NUMBERS / 18,
                'overdue_mult': 3.0,
                'label': 'Pari/Dispari',
                'friendly': {'even': 'Pari', 'odd': 'Dispari'},
                'icon': '🔢',
            },
            'high_low': {
                'values': {'high': HIGH_NUMBERS, 'low': LOW_NUMBERS},
                'expected_gap': TOTAL_NUMBERS / 18,
                'overdue_mult': 3.0,
                'label': 'Alto/Basso',
                'friendly': {'high': 'Alto (19-36)', 'low': 'Basso (1-18)'},
                'icon': '↕️',
            },
        }
        
        for cat_name, cfg in category_configs.items():
            for val_name, val_numbers in cfg['values'].items():
                # Calculate gap: how many spins since any number in this category appeared
                gap = 0
                for spin in reversed(numbers):
                    if spin in val_numbers:
                        break
                    gap += 1
                else:
                    gap = len(numbers)  # Never appeared
                
                expected = cfg['expected_gap']
                threshold = expected * cfg['overdue_mult']
                
                if gap >= threshold:
                    # Base probability for this category
                    category_coverage = len(val_numbers) / TOTAL_NUMBERS
                    base_prob = category_coverage
                    
                    # Overdue boost (logarithmic)
                    overdue_ratio = gap / expected
                    boost = min(0.10 * np.log(overdue_ratio), 0.30)
                    
                    final_prob = min(base_prob + boost, 0.85)
                    friendly = cfg['friendly'].get(val_name, val_name)
                    
                    sources = [f'⏰ Ritardo {gap} giri']
                    
                    # Check if there's also a streak of the opposite value
                    for pattern_type, streak in streaks.get('current_streaks', {}).items():
                        if pattern_type == cat_name and streak.get('value') != val_name and streak.get('length', 0) >= 3:
                            sources.append(f'🌊 Opposto in serie ({streak["length"]})')
                    
                    predictions.append({
                        'type': 'category',
                        'value': friendly,
                        'label': cfg['label'],
                        'probability': round(final_prob, 4),
                        'overdue_boost': round(boost, 4),
                        'base_probability': round(base_prob, 4),
                        'sources': sources,
                        'source_count': len(sources),
                        'agreement': round(len(sources) / TOTAL_SOURCES, 4),
                        'description': f'{friendly} — non esce da {gap} giri',
                    })
        
        # Sort by final probability descending
        predictions.sort(key=lambda x: -x['probability'])
        
        return predictions[:20]  # Top 20


# Singleton
_pattern_detector: Optional[PatternDetector] = None


def get_pattern_detector() -> PatternDetector:
    global _pattern_detector
    if _pattern_detector is None:
        _pattern_detector = PatternDetector()
    return _pattern_detector
