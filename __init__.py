"""
Portfolio Demo: Simplified Goal Prediction Pipeline

A simplified, educational implementation of a football match goal prediction
system using Poisson and Dixon-Coles models. This demo showcases:

- Data loading and validation
- Feature engineering for goal models
- Hyperparameter optimization with Optuna
- Model evaluation with proper scoring rules (RPS, Brier)
- Match prediction and market extraction

This is a standalone demo intended for portfolio purposes.
It does not expose any proprietary business logic or sensitive endpoints.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .data_loader import MatchDataLoader
from .feature_engineering import FeatureEngineer
from .model_training import ModelTrainer
from .model_evaluation import ModelEvaluator
from .predict import MatchPredictor

__all__ = [
    "MatchDataLoader",
    "FeatureEngineer",
    "ModelTrainer",
    "ModelEvaluator",
    "MatchPredictor",
]
