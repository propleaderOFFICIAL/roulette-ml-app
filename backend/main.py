"""
FastAPI backend for Roulette ML App.
Records spins and returns theoretical, empirical, and advanced ML predictions.

Advanced features:
- Multi-model ensemble predictions
- Pattern detection and analysis
- Statistical analysis (chi-squared, Monte Carlo, Markov chains)
- Model performance tracking
"""
import os
from datetime import datetime
from pathlib import Path
import json
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import roulette
from model import get_predictor
from ensemble_model import get_ensemble_predictor
from pattern_detector import get_pattern_detector
from statistical_engine import get_statistical_engine
from advanced_features import get_feature_extractor

app = FastAPI(
    title="Roulette ML API",
    description="Advanced API for recording roulette spins and getting AI/ML predictions",
    version="2.0.0",
)

# CORS: in produzione usa CORS_ORIGINS (es. https://tuo-frontend.onrender.com)
_origins = os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173,*")
_origins_list = [o.strip() for o in _origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage; optionally persist to data/spins.json
DATA_DIR = Path(__file__).parent / "data"
SPINS_FILE = DATA_DIR / "spins.json"
spins_store: list[dict] = []


def load_spins() -> None:
    global spins_store
    if SPINS_FILE.exists():
        try:
            with open(SPINS_FILE) as f:
                spins_store = json.load(f)
        except Exception:
            spins_store = []


def save_spins() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(SPINS_FILE, "w") as f:
        json.dump(spins_store, f, indent=2)


@app.on_event("startup")
def startup():
    load_spins()
    numbers = [s["number"] for s in spins_store]
    
    # Train legacy predictor
    predictor = get_predictor()
    if len(numbers) >= 15:
        predictor.ensure_trained(numbers)
    
    # Train advanced ensemble predictor
    ensemble = get_ensemble_predictor()
    if len(numbers) >= 30:
        ensemble.ensure_trained(numbers)


class SpinRequest(BaseModel):
    number: int = Field(..., ge=0, le=36, description="Roulette number 0-36")


class SpinResponse(BaseModel):
    number: int
    color: str
    timestamp: str
    total_spins: int


@app.get("/")
async def root():
    return {
        "message": "Roulette ML API - Advanced Edition",
        "status": "running",
        "version": "2.0.0",
        "endpoints": {
            "predictions": "/predictions",
            "advanced_predictions": "/predictions/advanced",
            "patterns": "/analysis/patterns",
            "statistics": "/analysis/statistics",
            "model_info": "/models/info",
            "features": "/analysis/features",
        }
    }


@app.post("/spins", response_model=SpinResponse)
async def add_spin(req: SpinRequest):
    """Record a spin (number 0-36)."""
    color = roulette.number_to_color(req.number)
    entry = {
        "number": req.number,
        "color": color,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    spins_store.append(entry)
    save_spins()
    
    # Train predictors
    numbers = [s["number"] for s in spins_store]
    
    predictor = get_predictor()
    predictor.ensure_trained(numbers)
    
    ensemble = get_ensemble_predictor()
    ensemble.ensure_trained(numbers)
    
    return SpinResponse(
        number=req.number,
        color=color,
        timestamp=entry["timestamp"],
        total_spins=len(spins_store),
    )


@app.get("/spins")
async def get_spins(limit: Optional[int] = 100):
    """Return last N spins (newest first)."""
    if limit is None or limit <= 0:
        limit = 100
    out = list(reversed(spins_store[-limit:]))
    return {"spins": out, "total": len(spins_store)}


@app.delete("/spins")
async def clear_spins():
    """Clear all spin history and reset predictors."""
    global spins_store
    count = len(spins_store)
    spins_store = []
    save_spins()
    
    # Reset predictors
    predictor = get_predictor()
    predictor._trained = False
    predictor._last_train_size = 0
    
    ensemble = get_ensemble_predictor()
    ensemble._trained = False
    ensemble._last_train_size = 0
    for model in ensemble.models:
        model._trained = False
    
    return {
        "message": "Cronologia cancellata",
        "cleared_count": count,
    }


@app.get("/predictions")
async def get_predictions():
    """Return theoretical, empirical, and basic ML predictions."""
    numbers = [s["number"] for s in spins_store]

    theoretical = {
        "color": roulette.theoretical_probabilities(),
        "number_probability": roulette.theoretical_number_probability(),
    }

    empirical_color = roulette.empirical_color_probabilities(numbers)
    empirical_top = roulette.empirical_top_numbers(numbers, top_n=10)
    empirical = {
        "color": empirical_color,
        "top_numbers": [{"number": n, "probability": p} for n, p in empirical_top],
    }

    predictor = get_predictor()
    model_color = predictor.predict_color_probs(numbers)
    model_numbers = predictor.predict_number_probs(numbers, top_n=10)
    model = None
    if model_color is not None or model_numbers is not None:
        model = {}
        if model_color is not None:
            model["color"] = model_color
        if model_numbers is not None:
            model["top_numbers"] = [{"number": int(n), "probability": float(p)} for n, p in model_numbers]

    # Get AI betting area predictions if ensemble is trained
    betting_areas = None
    ensemble = get_ensemble_predictor()
    if len(numbers) >= 30:
        ensemble.ensure_trained(numbers)
        betting_areas = ensemble.predict_betting_areas(numbers)

    return {
        "theoretical": theoretical,
        "empirical": empirical,
        "model": model,
        "betting_areas": betting_areas,
        "total_spins": len(spins_store),
    }


@app.get("/predictions/advanced")
async def get_advanced_predictions():
    """
    Return comprehensive multi-model ensemble predictions.
    
    Includes:
    - Ensemble predictions with confidence scores
    - Individual model predictions
    - Model weights and agreement metrics
    """
    numbers = [s["number"] for s in spins_store]
    ensemble = get_ensemble_predictor()
    
    if len(numbers) < 30:
        return {
            "error": "Need at least 30 spins for advanced predictions",
            "current_spins": len(numbers),
            "required_spins": 30,
        }
    
    ensemble.ensure_trained(numbers)
    
    color_predictions = ensemble.predict_color_probs(numbers)
    number_predictions = ensemble.predict_number_probs(numbers, top_n=15)
    betting_areas = ensemble.predict_betting_areas(numbers)
    model_info = ensemble.get_model_info()
    
    # Format number predictions for JSON
    if number_predictions and 'ensemble' in number_predictions:
        number_predictions['ensemble'] = [
            {"number": int(n), "probability": float(p)} 
            for n, p in number_predictions['ensemble']
        ]
        
        # Format individual model predictions
        if 'models' in number_predictions:
            for model_name, preds in number_predictions['models'].items():
                number_predictions['models'][model_name] = [
                    {"number": int(n), "probability": float(p)} 
                    for n, p in preds
                ]
    
    return {
        "color": color_predictions,
        "number": number_predictions,
        "betting_areas": betting_areas,
        "model_info": model_info,
        "total_spins": len(numbers),
    }


@app.get("/analysis/patterns")
async def get_pattern_analysis():
    """
    Get comprehensive pattern analysis.
    
    Includes:
    - Hot/cold number analysis
    - Sleeper (overdue) numbers
    - Streak patterns
    - Wheel sector bias analysis
    """
    numbers = [s["number"] for s in spins_store]
    detector = get_pattern_detector()
    
    if len(numbers) < 10:
        return {
            "error": "Need at least 10 spins for pattern analysis",
            "current_spins": len(numbers),
        }
    
    patterns = detector.detect_all_patterns(numbers)
    
    return {
        "patterns": patterns,
        "total_spins": len(numbers),
    }


@app.get("/analysis/statistics")
async def get_statistical_analysis():
    """
    Get comprehensive statistical analysis.
    
    Includes:
    - Chi-squared tests
    - Monte Carlo simulations
    - Markov chain analysis
    - Bayesian probability updates
    - Entropy and randomness metrics
    """
    numbers = [s["number"] for s in spins_store]
    engine = get_statistical_engine()
    
    if len(numbers) < 20:
        return {
            "error": "Need at least 20 spins for statistical analysis",
            "current_spins": len(numbers),
        }
    
    analysis = engine.get_full_analysis(numbers)
    
    return {
        "statistics": analysis,
        "total_spins": len(numbers),
    }


@app.get("/analysis/features")
async def get_feature_analysis():
    """
    Get detailed feature extraction for current state.
    
    Shows all features being fed to the ML models.
    """
    numbers = [s["number"] for s in spins_store]
    extractor = get_feature_extractor()
    
    if len(numbers) < 10:
        return {
            "error": "Need at least 10 spins for feature analysis",
            "current_spins": len(numbers),
        }
    
    features = extractor.extract_all_features(numbers)
    
    # Add additional info
    hot_cold = extractor.extract_hot_cold_features(numbers)
    sectors = extractor.extract_sector_features(numbers)
    streaks = extractor.extract_streak_features(numbers)
    
    return {
        "features": features,
        "hot_cold": hot_cold,
        "sectors": sectors,
        "streaks": streaks,
        "total_spins": len(numbers),
    }


@app.get("/models/info")
async def get_model_info():
    """
    Get information about available models and their status.
    """
    ensemble = get_ensemble_predictor()
    model_info = ensemble.get_model_info()
    
    return {
        "ensemble": model_info,
        "legacy_predictor": {
            "trained": get_predictor()._trained,
            "window_size": 5,
        },
        "total_spins": len(spins_store),
    }


@app.delete("/spins")
async def clear_spins():
    """Clear all recorded spins (for testing)."""
    global spins_store
    spins_store = []
    save_spins()
    return {"message": "All spins cleared", "total_spins": 0}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "total_spins": len(spins_store),
        "models_trained": get_ensemble_predictor()._trained,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
