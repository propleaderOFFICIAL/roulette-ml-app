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

import warnings
import numpy as np
from typing import Optional, Dict, List, Tuple, Any
from collections import deque
from datetime import datetime

# Sopprime il warning sklearn "unique classes > 50% of samples": roulette = 37 classi, pochi spin.
warnings.filterwarnings(
    "ignore",
    message="The number of unique classes is greater than 50% of the number of samples",
    category=UserWarning,
)

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
    
    def get_error_analysis(self) -> Dict[str, Any]:
        """Analyze prediction errors for learning insights."""
        if len(self.actual) < 5:
            return {'status': 'insufficient_data'}
        
        # Analyze which models are consistently wrong
        errors_by_type = {
            'color_errors': 0,
            'number_near_misses': 0,  # Predicted close number (on wheel)
            'total_predictions': len(self.actual)
        }
        
        for model_name, preds in self.predictions.items():
            preds_list = list(preds)
            actuals_list = list(self.actual)
            
            for i in range(min(len(preds_list) - 1, len(actuals_list))):
                pred = preds_list[i]['prediction']
                actual = actuals_list[i]
                
                # Check if prediction was in same wheel sector
                if abs(pred - actual) <= 3 or abs(pred - actual) >= 34:
                    errors_by_type['number_near_misses'] += 1
        
        return errors_by_type


class OnlineLearningManager:
    """
    Manages online/incremental learning for the ensemble.
    
    Monitors prediction performance and triggers retraining when:
    - Accuracy drops below threshold
    - Enough new samples accumulated
    - Performance diverges significantly from baseline
    
    This enables the system to "learn from errors" over time.
    """
    
    def __init__(self, 
                 accuracy_threshold: float = 0.02,  # Below random (1/37 ≈ 2.7%)
                 retrain_window: int = 30,
                 min_samples_for_retrain: int = 50):
        self.accuracy_threshold = accuracy_threshold
        self.retrain_window = retrain_window
        self.min_samples_for_retrain = min_samples_for_retrain
        self.retrain_history: List[Dict[str, Any]] = []
        self.last_retrain_sample_count = 0
        self.baseline_accuracy: Dict[str, float] = {}
        
    def should_retrain(self, 
                       performance_tracker: ModelPerformanceTracker,
                       current_sample_count: int) -> Tuple[bool, str]:
        """
        Determine if retraining should be triggered.
        
        Returns:
            (should_retrain: bool, reason: str)
        """
        # Check if enough new samples since last retrain
        samples_since_retrain = current_sample_count - self.last_retrain_sample_count
        if samples_since_retrain < self.retrain_window:
            return False, "not_enough_new_samples"
        
        # Check if we have minimum samples
        if current_sample_count < self.min_samples_for_retrain:
            return False, "below_minimum_samples"
        
        # Check model accuracies
        accuracies = performance_tracker.get_all_accuracies()
        if not accuracies:
            return False, "no_accuracy_data"
        
        # Calculate average accuracy
        avg_accuracy = sum(accuracies.values()) / len(accuracies)
        
        # Trigger retrain if accuracy dropped significantly
        if avg_accuracy < self.accuracy_threshold:
            return True, "accuracy_below_threshold"
        
        # Check for accuracy degradation vs baseline
        if self.baseline_accuracy:
            for model_name, current_acc in accuracies.items():
                baseline = self.baseline_accuracy.get(model_name, 0.5)
                if current_acc < baseline * 0.5:  # 50% drop from baseline
                    return True, f"model_{model_name}_degraded"
        
        # Periodic retrain after enough samples
        if samples_since_retrain >= self.retrain_window * 2:
            return True, "periodic_retrain"
        
        return False, "no_retrain_needed"
    
    def record_retrain(self, 
                       sample_count: int, 
                       reason: str,
                       accuracies_before: Dict[str, float],
                       accuracies_after: Dict[str, float] = None):
        """Record a retraining event for analysis."""
        self.last_retrain_sample_count = sample_count
        
        # Update baseline with post-retrain accuracy
        if accuracies_after:
            self.baseline_accuracy = accuracies_after.copy()
        
        self.retrain_history.append({
            'timestamp': datetime.utcnow().isoformat(),
            'sample_count': sample_count,
            'reason': reason,
            'accuracies_before': accuracies_before,
            'accuracies_after': accuracies_after
        })
        
        # Keep only last 20 retrain events
        if len(self.retrain_history) > 20:
            self.retrain_history = self.retrain_history[-20:]
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about the online learning process."""
        if not self.retrain_history:
            return {'status': 'no_retrains_yet'}
        
        return {
            'total_retrains': len(self.retrain_history),
            'last_retrain': self.retrain_history[-1],
            'current_baseline_accuracy': self.baseline_accuracy,
            'retrain_reasons': [r['reason'] for r in self.retrain_history[-5:]]
        }


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

            # Architettura completa (128, 64, 32) e (256, 128, 64, 32) per massima precisione
            # max_iter alto e validation_fraction moderata per convergere anche con ~100 campioni
            self.color_model = MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                max_iter=2000,
                early_stopping=True,
                validation_fraction=0.12,
                random_state=42,
                alpha=0.001,
            )
            self.color_model.fit(X_scaled, y_color_enc)

            self.number_model = MLPClassifier(
                hidden_layer_sizes=(256, 128, 64, 32),
                activation='relu',
                solver='adam',
                max_iter=2000,
                early_stopping=True,
                validation_fraction=0.12,
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


class SequenceAwarePredictor(BasePredictor):
    """
    Sequence-aware predictor using temporal feature engineering.
    
    Approximates recurrent/LSTM behavior without heavy dependencies by:
    - Using position-weighted sequence features
    - Capturing decay patterns (recent spins weigh more)
    - Learning transition patterns
    
    This is a lightweight alternative to LSTM that works with sklearn.
    """
    
    def __init__(self):
        super().__init__("SequenceAware")
        self.color_model = None
        self.number_model = None
        self.sequence_length = 20  # Last N spins for sequence features
        
    def _extract_sequence_features(self, X: np.ndarray) -> np.ndarray:
        """Add temporal weighting to features (exponential decay)."""
        # Apply exponential decay weighting to simulate recency bias
        n_features = X.shape[1]
        
        # Create decay weights for feature importance
        decay_factor = 0.95
        weights = np.array([decay_factor ** i for i in range(min(n_features, 50))])
        weights = np.tile(weights, (n_features // 50) + 1)[:n_features]
        
        # Apply weighted transformation
        X_weighted = X * weights
        
        # Consistent shape for training and prediction (removed X_diff to avoid mismatch)
        X_enhanced = np.hstack([X, X_weighted])
        
        return X_enhanced
        
    def fit(self, X: np.ndarray, y_color: np.ndarray, y_number: np.ndarray) -> bool:
        try:
            X_enhanced = self._extract_sequence_features(X)
            X_scaled = self.scaler.fit_transform(X_enhanced)
            y_color_enc = self.color_encoder.transform(y_color)
            
            # Deeper network for sequence patterns
            self.color_model = MLPClassifier(
                hidden_layer_sizes=(256, 128, 64, 32),
                activation='tanh',  # Better for temporal patterns
                solver='adam',
                max_iter=2000,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=42,
                alpha=0.0005,  # Less regularization for pattern learning
            )
            self.color_model.fit(X_scaled, y_color_enc)
            
            self.number_model = MLPClassifier(
                hidden_layer_sizes=(512, 256, 128, 64),
                activation='tanh',
                solver='adam',
                max_iter=2000,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=42,
                alpha=0.0005,
            )
            self.number_model.fit(X_scaled, y_number)
            
            self._trained = True
            return True
        except Exception as e:
            print(f"SequenceAware training error: {e}")
            return False
    
    def predict_color_probs(self, X: np.ndarray) -> Optional[Dict[str, float]]:
        if not self._trained or self.color_model is None:
            return None
        try:
            X_enhanced = self._extract_sequence_features(X)
            X_scaled = self.scaler.transform(X_enhanced)
            probs = self.color_model.predict_proba(X_scaled)[0]
            labels = self.color_encoder.classes_
            return dict(zip(labels, probs.tolist()))
        except:
            return None
    
    def predict_number_probs(self, X: np.ndarray, top_n: int = 10) -> Optional[List[Tuple[int, float]]]:
        if not self._trained or self.number_model is None:
            return None
        try:
            X_enhanced = self._extract_sequence_features(X)
            X_scaled = self.scaler.transform(X_enhanced)
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
                eval_metric='mlogloss',
            )
            self.color_model.fit(X_scaled, y_color_enc)
            
            # Number model: 37 classi fisse (roulette europea 0-36).
            # Lo sklearn wrapper inferisce num_class da np.unique(y): se nei dati mancano
            # alcuni numeri, XGBoost va in errore. Aggiungiamo 37 righe "prior" (una per
            # ogni classe) con peso minimo così tutte le classi sono presenti in y.
            last_row = X_scaled[-1 :].copy()
            X_prior = np.tile(last_row, (TOTAL_NUMBERS, 1))
            y_prior = np.arange(TOTAL_NUMBERS, dtype=np.int64)
            X_num = np.vstack([X_scaled, X_prior])
            y_num_full = np.concatenate([y_number, y_prior])
            w_real = np.ones(len(y_number), dtype=np.float64)
            w_prior = np.full(TOTAL_NUMBERS, 1e-6, dtype=np.float64)
            sample_weight_num = np.concatenate([w_real, w_prior])
            self.number_model = xgb.XGBClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=5,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='mlogloss',
                num_class=TOTAL_NUMBERS,
            )
            self.number_model.fit(X_num, y_num_full, sample_weight=sample_weight_num)
            
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
            SequenceAwarePredictor(),  # New sequence-aware model
        ]
        self.model_weights: Dict[str, float] = {
            'DeepMLP': 0.25,
            'RandomForest': 0.20,
            'GradientBoosting': 0.20,
            'XGBoost': 0.15,
            'SequenceAware': 0.20,  # Give sequence model good weight
        }
        self.performance_tracker = ModelPerformanceTracker()
        self.online_learning = OnlineLearningManager()
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
        
        # Train each model (evita messaggio "failed" per modelli non disponibili, es. XGBoost senza libomp)
        for model in self.models:
            success = model.fit(X, y_color, y_number)
            if not success and getattr(model, "available", True):
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
    
    def smart_retrain(self, numbers: list[int]) -> Dict[str, Any]:
        """
        Intelligent retraining that learns from errors.
        
        Uses OnlineLearningManager to decide when to retrain based on:
        - Prediction accuracy degradation
        - Performance vs baseline
        - New data accumulation
        
        Returns:
            Dict with retrain decision and stats
        """
        if len(numbers) < MIN_SAMPLES:
            return {'action': 'skipped', 'reason': 'insufficient_samples'}
        
        # Initial training
        if not self._trained:
            self.fit(numbers)
            return {'action': 'initial_training', 'samples': len(numbers)}
        
        # Check if retraining needed
        should_retrain, reason = self.online_learning.should_retrain(
            self.performance_tracker,
            len(numbers)
        )
        
        if should_retrain:
            # Save pre-retrain accuracy
            accuracies_before = self.performance_tracker.get_all_accuracies()
            
            # Retrain
            self.fit(numbers)
            
            # Update weights based on new performance
            self._update_weights()
            
            # Get post-retrain accuracy (will be same until new predictions)
            accuracies_after = self.performance_tracker.get_all_accuracies()
            
            # Record the retrain event
            self.online_learning.record_retrain(
                len(numbers), reason, accuracies_before, accuracies_after
            )
            
            return {
                'action': 'retrained',
                'reason': reason,
                'samples': len(numbers),
                'accuracies_before': accuracies_before
            }
        
        return {'action': 'no_retrain_needed', 'reason': reason}
    
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
                'ensemble': {'red': 0.47, 'black': 0.53},
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
        
        # Weighted ensemble (only red/black — green/0 is a number, not a color bet)
        raw_ensemble = {'red': 0.0, 'black': 0.0, 'green': 0.0}
        total_weight = 0.0
        
        for name, probs in valid_predictions:
            weight = self.model_weights.get(name, 0.25)
            for color in raw_ensemble:
                raw_ensemble[color] += probs.get(color, 0) * weight
            total_weight += weight
        
        # Normalize raw
        for color in raw_ensemble:
            raw_ensemble[color] /= max(total_weight, 1e-6)
        
        # Redistribute green probability proportionally to red/black
        green_prob = raw_ensemble.get('green', 0.0)
        rb_total = raw_ensemble['red'] + raw_ensemble['black']
        if rb_total > 0:
            ensemble = {
                'red': raw_ensemble['red'] + green_prob * (raw_ensemble['red'] / rb_total),
                'black': raw_ensemble['black'] + green_prob * (raw_ensemble['black'] / rb_total),
            }
        else:
            ensemble = {'red': 0.5, 'black': 0.5}
        
        # Calculate confidence (max probability)
        confidence = max(ensemble.values())
        
        # Calculate agreement (how many models agree on top prediction)
        # For agreement, compare only red vs black per model
        top_color = max(ensemble, key=ensemble.get)
        agreements = sum(
            1 for _, probs in valid_predictions
            if max({'red': probs.get('red', 0), 'black': probs.get('black', 0)},
                   key=lambda c: {'red': probs.get('red', 0), 'black': probs.get('black', 0)}[c]) == top_color
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

        # Calculate agreement (how many models agree on top prediction)
        top_num = top_numbers[0][0] if top_numbers else -1
        
        # Helper to get top number for a model
        def get_model_top_num(probs_list):
            if not probs_list: return -1
            # probs_list is [(num, prob), ...] sorted or not
            # We assume it's valid. Find max prob.
            return max(probs_list, key=lambda x: x[1])[0]

        agreements = sum(
            1 for name, probs in model_predictions.items()
            if get_model_top_num(probs) == top_num
        )
        agreement = agreements / len(model_predictions) if model_predictions else 0.0
        
        return {
            'ensemble': top_numbers,
            'confidence': confidence,
            'agreement': agreement,
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
        Includes agreement percentage among models.
        """
        # Get full number probability distribution
        number_preds = self.predict_number_probs(numbers, top_n=37)
        if not number_preds:
            return None
        
        # Helper to find best prediction
        def get_prediction(probs: Dict[str, float]) -> Dict[str, Any]:
            best = max(probs, key=probs.get)
            return {
                'probabilities': probs,
                'prediction': best,
                'confidence': probs[best]
            }

        # Helper to calculate area probabilities
        def calculate_area_probs(probs_list):
            # Convert to full probability dict (all 37 numbers)
            num_probs = {n: 0.0 for n in range(37)}
            for num, prob in probs_list:
                num_probs[num] = prob
            
            # Normalize (ensure sums to ~1)
            total = sum(num_probs.values())
            if total > 0:
                for n in num_probs:
                    num_probs[n] /= total
            
            # Helper to get number's properties
            def get_dozen(n): return 0 if n == 0 else 1 + (n-1)//12
            def get_column(n): return 0 if n == 0 else 1 + (n-1)%3
            
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
                elif d == 3: dozens['3ª (25-36)'] += prob
                
                # Columns
                c = get_column(n)
                if c == 1: columns['1ª col'] += prob
                elif c == 2: columns['2ª col'] += prob
                elif c == 3: columns['3ª col'] += prob
                
                # High/Low
                if n <= 18: high_low['Basso (1-18)'] += prob
                else: high_low['Alto (19-36)'] += prob
                
                # Parity
                if n % 2 == 0: parity['Pari'] += prob
                else: parity['Dispari'] += prob
            
            # Sectors (European wheel layout)
            voisins_nums = {0, 2, 3, 4, 7, 12, 15, 18, 19, 21, 22, 25, 26, 28, 29, 32, 35}
            tiers_nums = {5, 8, 10, 11, 13, 16, 23, 24, 27, 30, 33, 36}
            orphelins_nums = {1, 6, 9, 14, 17, 20, 31, 34}
            
            sectors = {'Voisins': 0.0, 'Tiers': 0.0, 'Orphelins': 0.0}
            for n in range(0, 37):
                p = num_probs[n]
                if n in voisins_nums: sectors['Voisins'] += p
                elif n in tiers_nums: sectors['Tiers'] += p
                elif n in orphelins_nums: sectors['Orphelins'] += p
            
            return {
                'dozen': get_prediction(dozens),
                'column': get_prediction(columns),
                'high_low': get_prediction(high_low),
                'parity': get_prediction(parity),
                'sector': get_prediction(sectors),
                'zero_probability': zero_prob
            }

        # 1. Ensemble prediction
        ensemble_area_predictions = calculate_area_probs(number_preds['ensemble'])
        ensemble_area_predictions['source'] = 'ensemble_number_distribution'

        # 2. Individual model predictions for agreement calculation
        model_area_predictions = {}
        # Avoid computing if models are not present (should be there if ensemble is trained)
        for model_name, model_probs_list in number_preds.get('models', {}).items():
            model_area_predictions[model_name] = calculate_area_probs(model_probs_list)

        # 3. Calculate agreement for each area type
        area_types = ['dozen', 'column', 'high_low', 'parity', 'sector']
        for area_type in area_types:
            top_pred = ensemble_area_predictions[area_type]['prediction']
            
            agreements = 0
            total_models = len(model_area_predictions)
            
            for model_preds in model_area_predictions.values():
                if model_preds[area_type]['prediction'] == top_pred:
                    agreements += 1
            
            ensemble_area_predictions[area_type]['agreement'] = (
                agreements / total_models if total_models > 0 else 0.0
            )

        return ensemble_area_predictions
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return information about model status, weights, and learning stats."""
        return {
            'trained': self._trained,
            'total_samples': self._last_train_size,
            'models': {
                m.name: {
                    'trained': m._trained,
                    'weight': self.model_weights.get(m.name, 0),
                    'available': getattr(m, 'available', True),
                    'accuracy': self.performance_tracker.get_accuracy(m.name)
                }
                for m in self.models
            },
            'min_samples_required': MIN_SAMPLES,
            'retrain_interval': RETRAIN_INTERVAL,
            'learning_stats': self.online_learning.get_learning_stats(),
            'performance': {
                'accuracies': self.performance_tracker.get_all_accuracies(),
                'error_analysis': self.performance_tracker.get_error_analysis()
            }
        }


# Singleton
_ensemble_predictor: Optional[EnsembleRoulettePredictor] = None


def get_ensemble_predictor() -> EnsembleRoulettePredictor:
    global _ensemble_predictor
    if _ensemble_predictor is None:
        _ensemble_predictor = EnsembleRoulettePredictor()
    return _ensemble_predictor
