"""
Ensemble Model System for Advanced Roulette Prediction.

Multi-model ensemble combining:
- Deep Neural Networks (MLP with attention-like weighting)
- Random Forest Classifier
- Gradient Boosting Classifier
- XGBoost Classifier (if available)
- Weighted voting ensemble with dynamic weight adjustment

Uses soft voting with confidence scoring and model performance tracking.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple, Any
from collections import deque
from datetime import datetime

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.calibration import CalibratedClassifierCV

# XGBoost is optional (richiede libomp su macOS: brew install libomp)
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False
    xgb = None

from roulette import number_to_color, TOTAL_NUMBERS
from advanced_features import get_feature_extractor, AdvancedFeatureExtractor


# Configuration
WINDOW_SIZE = 10  # Sequence window for feature extraction
MIN_SAMPLES = 30  # Minimum samples to train
RETRAIN_INTERVAL = 10  # Retrain after N new spins
PERFORMANCE_WINDOW = 50  # Track last N predictions for performance


class ModelPerformanceTracker:
    """Track prediction performance for dynamic weight adjustment."""
    
    def __init__(self, window_size: int = PERFORMANCE_WINDOW):
        self.window_size = window_size
        self.predictions: Dict[str, deque] = {}
        self.actual: deque = deque(maxlen=window_size)
        
    def add_prediction(self, model_name: str, prediction: int, confidence: float):
        """Record a prediction from a model."""
        if model_name not in self.predictions:
            self.predictions[model_name] = deque(maxlen=self.window_size)
        self.predictions[model_name].append({
            'prediction': prediction,
            'confidence': confidence,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    def add_actual(self, actual: int):
        """Record actual outcome."""
        self.actual.append(actual)
    
    def get_accuracy(self, model_name: str) -> float:
        """Calculate recent accuracy for a model."""
        if model_name not in self.predictions:
            return 0.5  # Default
        
        preds = list(self.predictions[model_name])
        actuals = list(self.actual)
        
        # Align predictions with actuals (offset by 1)
        if len(preds) < 2 or len(actuals) < 1:
            return 0.5
        
        correct = 0
        total = min(len(preds) - 1, len(actuals))
        
        for i in range(total):
            if preds[i]['prediction'] == actuals[i]:
                correct += 1
        
        return correct / max(total, 1)
    
    def get_all_accuracies(self) -> Dict[str, float]:
        """Get accuracy for all tracked models."""
        return {name: self.get_accuracy(name) for name in self.predictions}


class BasePredictor:
    """Base class for individual predictors."""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.scaler = StandardScaler()
        self.color_encoder = LabelEncoder()
        self.color_encoder.fit(['red', 'black', 'green'])
        self._trained = False
        
    def fit(self, X: np.ndarray, y_color: np.ndarray, y_number: np.ndarray) -> bool:
        """Train the model. Override in subclasses."""
        raise NotImplementedError
    
    def predict_color_probs(self, X: np.ndarray) -> Optional[Dict[str, float]]:
        """Predict color probabilities."""
        raise NotImplementedError
    
    def predict_number_probs(self, X: np.ndarray, top_n: int = 10) -> Optional[List[Tuple[int, float]]]:
        """Predict number probabilities."""
        raise NotImplementedError


class DeepNeuralNetworkPredictor(BasePredictor):
    """
    Deep MLP with multiple hidden layers.
    Architecture: Input -> 128 -> 64 -> 32 -> Output
    Uses calibrated probabilities for better confidence estimates.
    """
    
    def __init__(self):
        super().__init__("DeepMLP")
        self.color_model = None
        self.number_model = None
        
    def fit(self, X: np.ndarray, y_color: np.ndarray, y_number: np.ndarray) -> bool:
        try:
            X_scaled = self.scaler.fit_transform(X)
            y_color_enc = self.color_encoder.transform(y_color)
            n_samples = len(X_scaled)
            # Con ~100 campioni usiamo reti più piccole e più iterazioni per far convergere
            if n_samples < 80:
                hidden_color, hidden_num = (64, 32), (128, 64, 32)
            else:
                hidden_color, hidden_num = (128, 64, 32), (256, 128, 64, 32)

            # Color model: solo MLP (niente calibrazione per evitare errori con pochi dati)
            self.color_model = MLPClassifier(
                hidden_layer_sizes=hidden_color,
                activation='relu',
                solver='adam',
                max_iter=1500,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=42,
                alpha=0.001,
            )
            self.color_model.fit(X_scaled, y_color_enc)

            # Number model
            self.number_model = MLPClassifier(
                hidden_layer_sizes=hidden_num,
                activation='relu',
                solver='adam',
                max_iter=1500,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=42,
                alpha=0.001,
            )
            self.number_model.fit(X_scaled, y_number)

            self._trained = True
            return True
        except Exception as e:
            print(f"DeepMLP training error: {e}")
            return False
    
    def predict_color_probs(self, X: np.ndarray) -> Optional[Dict[str, float]]:
        if not self._trained or self.color_model is None:
            return None
        try:
            X_scaled = self.scaler.transform(X)
            probs = self.color_model.predict_proba(X_scaled)[0]
            labels = self.color_encoder.classes_
            return dict(zip(labels, probs.tolist()))
        except:
            return None
    
    def predict_number_probs(self, X: np.ndarray, top_n: int = 10) -> Optional[List[Tuple[int, float]]]:
        if not self._trained or self.number_model is None:
            return None
        try:
            X_scaled = self.scaler.transform(X)
            probs = self.number_model.predict_proba(X_scaled)[0]
            classes = self.number_model.classes_
            pairs = list(zip(classes.tolist(), probs.tolist()))
            pairs.sort(key=lambda x: -x[1])
            return pairs[:top_n]
        except:
            return None


class RandomForestPredictor(BasePredictor):
    """
    Random Forest ensemble classifier.
    Good for capturing non-linear patterns with feature importance.
    """
    
    def __init__(self):
        super().__init__("RandomForest")
        self.color_model = None
        self.number_model = None
        
    def fit(self, X: np.ndarray, y_color: np.ndarray, y_number: np.ndarray) -> bool:
        try:
            X_scaled = self.scaler.fit_transform(X)
            y_color_enc = self.color_encoder.transform(y_color)
            
            # Color model
            self.color_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            self.color_model.fit(X_scaled, y_color_enc)
            
            # Number model
            self.number_model = RandomForestClassifier(
                n_estimators=300,
                max_depth=20,
                min_samples_split=3,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1
            )
            self.number_model.fit(X_scaled, y_number)
            
            self._trained = True
            return True
        except Exception as e:
            print(f"RandomForest training error: {e}")
            return False
    
    def predict_color_probs(self, X: np.ndarray) -> Optional[Dict[str, float]]:
        if not self._trained or self.color_model is None:
            return None
        try:
            X_scaled = self.scaler.transform(X)
            probs = self.color_model.predict_proba(X_scaled)[0]
            labels = self.color_encoder.classes_
            return dict(zip(labels, probs.tolist()))
        except:
            return None
    
    def predict_number_probs(self, X: np.ndarray, top_n: int = 10) -> Optional[List[Tuple[int, float]]]:
        if not self._trained or self.number_model is None:
            return None
        try:
            X_scaled = self.scaler.transform(X)
            probs = self.number_model.predict_proba(X_scaled)[0]
            classes = self.number_model.classes_
            pairs = list(zip(classes.tolist(), probs.tolist()))
            pairs.sort(key=lambda x: -x[1])
            return pairs[:top_n]
        except:
            return None
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores from the color model."""
        if self.color_model is None:
            return {}
        return dict(enumerate(self.color_model.feature_importances_))


class GradientBoostingPredictor(BasePredictor):
    """
    Gradient Boosting classifier.
    Sequential ensemble good for capturing complex patterns.
    """
    
    def __init__(self):
        super().__init__("GradientBoosting")
        self.color_model = None
        self.number_model = None
        
    def fit(self, X: np.ndarray, y_color: np.ndarray, y_number: np.ndarray) -> bool:
        try:
            X_scaled = self.scaler.fit_transform(X)
            y_color_enc = self.color_encoder.transform(y_color)
            
            # Color model
            self.color_model = GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=5,
                min_samples_leaf=2,
                subsample=0.8,
                random_state=42
            )
            self.color_model.fit(X_scaled, y_color_enc)
            
            # Number model (lighter due to 37 classes)
            self.number_model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=4,
                min_samples_split=5,
                subsample=0.8,
                random_state=42
            )
            self.number_model.fit(X_scaled, y_number)
            
            self._trained = True
            return True
        except Exception as e:
            print(f"GradientBoosting training error: {e}")
            return False
    
    def predict_color_probs(self, X: np.ndarray) -> Optional[Dict[str, float]]:
        if not self._trained or self.color_model is None:
            return None
        try:
            X_scaled = self.scaler.transform(X)
            probs = self.color_model.predict_proba(X_scaled)[0]
            labels = self.color_encoder.classes_
            return dict(zip(labels, probs.tolist()))
        except:
            return None
    
    def predict_number_probs(self, X: np.ndarray, top_n: int = 10) -> Optional[List[Tuple[int, float]]]:
        if not self._trained or self.number_model is None:
            return None
        try:
            X_scaled = self.scaler.transform(X)
            probs = self.number_model.predict_proba(X_scaled)[0]
            classes = self.number_model.classes_
            pairs = list(zip(classes.tolist(), probs.tolist()))
            pairs.sort(key=lambda x: -x[1])
            return pairs[:top_n]
        except:
            return None


class XGBoostPredictor(BasePredictor):
    """
    XGBoost classifier (state-of-the-art gradient boosting).
    Only available if xgboost is installed.
    """
    
    def __init__(self):
        super().__init__("XGBoost")
        self.color_model = None
        self.number_model = None
        self.available = HAS_XGBOOST
        
    def fit(self, X: np.ndarray, y_color: np.ndarray, y_number: np.ndarray) -> bool:
        if not self.available:
            return False
        try:
            X_scaled = self.scaler.fit_transform(X)
            y_color_enc = self.color_encoder.transform(y_color)
            
            # Color model
            self.color_model = xgb.XGBClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                min_child_weight=2,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric='mlogloss'
            )
            self.color_model.fit(X_scaled, y_color_enc)
            
            # Number model
            self.number_model = xgb.XGBClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=5,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric='mlogloss'
            )
            self.number_model.fit(X_scaled, y_number)
            
            self._trained = True
            return True
        except Exception as e:
            print(f"XGBoost training error: {e}")
            return False
    
    def predict_color_probs(self, X: np.ndarray) -> Optional[Dict[str, float]]:
        if not self._trained or self.color_model is None:
            return None
        try:
            X_scaled = self.scaler.transform(X)
            probs = self.color_model.predict_proba(X_scaled)[0]
            labels = self.color_encoder.classes_
            return dict(zip(labels, probs.tolist()))
        except:
            return None
    
    def predict_number_probs(self, X: np.ndarray, top_n: int = 10) -> Optional[List[Tuple[int, float]]]:
        if not self._trained or self.number_model is None:
            return None
        try:
            X_scaled = self.scaler.transform(X)
            probs = self.number_model.predict_proba(X_scaled)[0]
            classes = self.number_model.classes_
            pairs = list(zip(classes.tolist(), probs.tolist()))
            pairs.sort(key=lambda x: -x[1])
            return pairs[:top_n]
        except:
            return None


class EnsembleRoulettePredictor:
    """
    Main ensemble predictor combining multiple models.
    
    Uses weighted soft voting with dynamic weight adjustment
    based on recent prediction performance.
    """
    
    def __init__(self):
        self.feature_extractor = get_feature_extractor()
        self.models: List[BasePredictor] = [
            DeepNeuralNetworkPredictor(),
            RandomForestPredictor(),
            GradientBoostingPredictor(),
            XGBoostPredictor(),
        ]
        self.model_weights: Dict[str, float] = {
            'DeepMLP': 0.30,
            'RandomForest': 0.25,
            'GradientBoosting': 0.25,
            'XGBoost': 0.20,
        }
        self.performance_tracker = ModelPerformanceTracker()
        self._trained = False
        self._last_train_size = 0
        
    def _build_training_data(self, numbers: list[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build training dataset with advanced features."""
        X_list = []
        y_color_list = []
        y_number_list = []
        
        # Use sliding window
        for i in range(WINDOW_SIZE, len(numbers)):
            # Get sequence up to position i
            sequence = numbers[:i]
            
            # Extract features
            features = self.feature_extractor.extract_all_features(sequence)
            X_list.append(self.feature_extractor.features_to_vector(features))
            
            # Target: next number/color
            y_number_list.append(numbers[i])
            y_color_list.append(number_to_color(numbers[i]))
        
        return (
            np.array(X_list),
            np.array(y_color_list),
            np.array(y_number_list)
        )
    
    def fit(self, numbers: list[int]) -> bool:
        """Train all ensemble models."""
        if len(numbers) < MIN_SAMPLES:
            return False
        
        X, y_color, y_number = self._build_training_data(numbers)
        
        if len(X) < 10:  # Need enough samples
            return False
        
        # Train each model
        for model in self.models:
            success = model.fit(X, y_color, y_number)
            if not success:
                print(f"Model {model.name} failed to train")
        
        self._trained = True
        self._last_train_size = len(numbers)
        return True
    
    def ensure_trained(self, numbers: list[int]) -> None:
        """Retrain if needed based on new data."""
        if len(numbers) < MIN_SAMPLES:
            return
        
        if not self._trained:
            self.fit(numbers)
        elif len(numbers) - self._last_train_size >= RETRAIN_INTERVAL:
            self.fit(numbers)
    
    def _update_weights(self):
        """Dynamically adjust model weights based on performance."""
        accuracies = self.performance_tracker.get_all_accuracies()
        
        if not accuracies:
            return
        
        # Compute softmax weights from accuracies
        values = [accuracies.get(m.name, 0.5) for m in self.models if m._trained]
        names = [m.name for m in self.models if m._trained]
        
        if not values:
            return
        
        # Softmax with temperature
        temperature = 2.0
        exp_values = np.exp(np.array(values) / temperature)
        softmax = exp_values / np.sum(exp_values)
        
        for name, weight in zip(names, softmax):
            self.model_weights[name] = float(weight)
    
    def predict_color_probs(self, numbers: list[int]) -> Optional[Dict[str, Any]]:
        """
        Get ensemble color prediction with individual model contributions.
        
        Returns:
            {
                'ensemble': {'red': 0.45, 'black': 0.52, 'green': 0.03},
                'confidence': 0.72,
                'models': {
                    'DeepMLP': {'red': 0.44, ...},
                    'RandomForest': {'red': 0.46, ...},
                    ...
                },
                'agreement': 0.8
            }
        """
        if not self._trained or len(numbers) < WINDOW_SIZE:
            return None
        
        # Extract features for current state
        features = self.feature_extractor.extract_all_features(numbers)
        X = np.array([self.feature_extractor.features_to_vector(features)])
        
        # Get predictions from each model
        model_predictions = {}
        valid_predictions = []
        
        for model in self.models:
            if not model._trained:
                continue
            probs = model.predict_color_probs(X)
            if probs:
                model_predictions[model.name] = probs
                valid_predictions.append((model.name, probs))
        
        if not valid_predictions:
            return None
        
        # Weighted ensemble
        ensemble = {'red': 0.0, 'black': 0.0, 'green': 0.0}
        total_weight = 0.0
        
        for name, probs in valid_predictions:
            weight = self.model_weights.get(name, 0.25)
            for color in ensemble:
                ensemble[color] += probs.get(color, 0) * weight
            total_weight += weight
        
        # Normalize
        for color in ensemble:
            ensemble[color] /= max(total_weight, 1e-6)
        
        # Calculate confidence (max probability)
        confidence = max(ensemble.values())
        
        # Calculate agreement (how many models agree on top prediction)
        top_color = max(ensemble, key=ensemble.get)
        agreements = sum(
            1 for _, probs in valid_predictions 
            if max(probs, key=probs.get) == top_color
        )
        agreement = agreements / len(valid_predictions)
        
        return {
            'ensemble': ensemble,
            'confidence': confidence,
            'models': model_predictions,
            'agreement': agreement,
            'weights': {k: v for k, v in self.model_weights.items() 
                       if k in model_predictions}
        }
    
    def predict_number_probs(self, numbers: list[int], 
                            top_n: int = 10) -> Optional[Dict[str, Any]]:
        """
        Get ensemble number prediction with individual model contributions.
        """
        if not self._trained or len(numbers) < WINDOW_SIZE:
            return None
        
        # Extract features
        features = self.feature_extractor.extract_all_features(numbers)
        X = np.array([self.feature_extractor.features_to_vector(features)])
        
        # Get predictions from each model
        model_predictions = {}
        all_number_probs = {n: 0.0 for n in range(TOTAL_NUMBERS)}
        total_weight = 0.0
        
        for model in self.models:
            if not model._trained:
                continue
            
            probs = model.predict_number_probs(X, top_n=TOTAL_NUMBERS)
            if probs:
                model_predictions[model.name] = probs[:top_n]
                weight = self.model_weights.get(model.name, 0.25)
                
                for num, prob in probs:
                    all_number_probs[num] += prob * weight
                total_weight += weight
        
        if not model_predictions:
            return None
        
        # Normalize and sort
        for num in all_number_probs:
            all_number_probs[num] /= max(total_weight, 1e-6)
        
        sorted_probs = sorted(all_number_probs.items(), key=lambda x: -x[1])
        top_numbers = [(int(n), float(p)) for n, p in sorted_probs[:top_n]]
        
        # Confidence
        confidence = top_numbers[0][1] if top_numbers else 0.0
        
        return {
            'ensemble': top_numbers,
            'confidence': confidence,
            'models': model_predictions,
            'weights': {k: v for k, v in self.model_weights.items() 
                       if k in model_predictions}
        }
    
    def predict_betting_areas(self, numbers: list[int]) -> Optional[Dict[str, Any]]:
        """
        Get AI predictions for betting areas based on number probability distributions.
        
        Returns predictions for:
        - Dozens (1-12, 13-24, 25-36)
        - Columns (1st, 2nd, 3rd)
        - High/Low (1-18, 19-36)
        - Parity (Even/Odd)
        
        Each with probabilities derived from the ensemble number predictions.
        """
        # Get full number probability distribution
        number_preds = self.predict_number_probs(numbers, top_n=37)
        if not number_preds:
            return None
        
        # Convert to full probability dict (all 37 numbers)
        num_probs = {n: 0.0 for n in range(37)}
        for num, prob in number_preds['ensemble']:
            num_probs[num] = prob
        
        # Normalize (ensure sums to ~1)
        total = sum(num_probs.values())
        if total > 0:
            for n in num_probs:
                num_probs[n] /= total
        
        # Helper to get number's properties
        def get_dozen(n: int) -> int:
            if n == 0: return 0
            if n <= 12: return 1
            if n <= 24: return 2
            return 3
        
        def get_column(n: int) -> int:
            if n == 0: return 0
            return ((n - 1) % 3) + 1
        
        # Calculate betting area probabilities
        dozens = {'1ª (1-12)': 0.0, '2ª (13-24)': 0.0, '3ª (25-36)': 0.0}
        columns = {'1ª col': 0.0, '2ª col': 0.0, '3ª col': 0.0}
        high_low = {'Basso (1-18)': 0.0, 'Alto (19-36)': 0.0}
        parity = {'Pari': 0.0, 'Dispari': 0.0}
        zero_prob = num_probs[0]
        
        for n in range(1, 37):
            prob = num_probs[n]
            
            # Dozens
            d = get_dozen(n)
            if d == 1: dozens['1ª (1-12)'] += prob
            elif d == 2: dozens['2ª (13-24)'] += prob
            else: dozens['3ª (25-36)'] += prob
            
            # Columns
            c = get_column(n)
            if c == 1: columns['1ª col'] += prob
            elif c == 2: columns['2ª col'] += prob
            else: columns['3ª col'] += prob
            
            # High/Low
            if n <= 18:
                high_low['Basso (1-18)'] += prob
            else:
                high_low['Alto (19-36)'] += prob
            
            # Parity
            if n % 2 == 0:
                parity['Pari'] += prob
            else:
                parity['Dispari'] += prob
        
        # Find best predictions
        def get_prediction(probs: Dict[str, float]) -> Dict[str, Any]:
            best = max(probs, key=probs.get)
            return {
                'probabilities': probs,
                'prediction': best,
                'confidence': probs[best]
            }
        
        return {
            'dozen': get_prediction(dozens),
            'column': get_prediction(columns),
            'high_low': get_prediction(high_low),
            'parity': get_prediction(parity),
            'zero_probability': zero_prob,
            'source': 'ensemble_number_distribution'
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return information about model status and weights."""
        return {
            'trained': self._trained,
            'total_samples': self._last_train_size,
            'models': {
                m.name: {
                    'trained': m._trained,
                    'weight': self.model_weights.get(m.name, 0),
                    'available': getattr(m, 'available', True)
                }
                for m in self.models
            },
            'min_samples_required': MIN_SAMPLES,
            'retrain_interval': RETRAIN_INTERVAL,
        }


# Singleton
_ensemble_predictor: Optional[EnsembleRoulettePredictor] = None


def get_ensemble_predictor() -> EnsembleRoulettePredictor:
    global _ensemble_predictor
    if _ensemble_predictor is None:
        _ensemble_predictor = EnsembleRoulettePredictor()
    return _ensemble_predictor
