"""
Model Training Module

Handles training of Poisson and Dixon-Coles goal prediction models.
Uses the penaltyblog library for the statistical model implementations.

Models supported:
- Poisson: Basic independent Poisson model for home/away goals
- Dixon-Coles: Poisson with low-score correlation adjustment

Training includes:
- Hyperparameter optimization via Optuna
- Temporal decay weighting
- Model persistence with versioning
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import joblib
import numpy as np
import optuna
import pandas as pd
from penaltyblog.models import DixonColesGoalModel, PoissonGoalsModel


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    model_type: Literal["poisson", "dixon_coles"] = "dixon_coles"
    xi: float = 0.005  # Temporal decay rate
    window_years: int = 4  # Training window in years
    max_goals: int = 10  # Maximum goals for probability grid


@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization."""

    n_trials: int = 50
    metric: Literal["rps", "brier"] = "rps"
    xi_range: tuple[float, float] = (0.0001, 0.01)
    window_years_range: tuple[int, int] = (2, 6)
    test_season: Optional[str] = None  # If None, uses last season


@dataclass
class TrainingResult:
    """Result of model training."""

    model_type: str
    xi: float
    window_years: int
    train_samples: int
    teams: list[str]
    trained_at: str
    artifact_path: Optional[str] = None


class ModelTrainer:
    """
    Train goal prediction models with hyperparameter optimization.

    This class provides:
    - Single model training with fixed hyperparameters
    - Hyperparameter optimization via Optuna
    - Model persistence with metadata

    Example:
        >>> trainer = ModelTrainer()
        >>> # Train with fixed parameters
        >>> model, result = trainer.train(df, config=TrainingConfig())
        >>>
        >>> # Or optimize hyperparameters
        >>> best_model, best_result = trainer.optimize(
        ...     df,
        ...     config=OptimizationConfig(n_trials=30)
        ... )
    """

    MODEL_CLASSES = {
        "poisson": PoissonGoalsModel,
        "dixon_coles": DixonColesGoalModel,
    }

    def __init__(self, output_dir: str | Path = "models"):
        """
        Initialize the model trainer.

        Args:
            output_dir: Directory to save trained models.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        df: pd.DataFrame,
        config: Optional[TrainingConfig] = None,
    ) -> tuple[object, TrainingResult]:
        """
        Train a single model with fixed configuration.

        Args:
            df: Prepared match DataFrame.
            config: Training configuration (uses defaults if not provided).

        Returns:
            Tuple of (trained model, training result).
        """
        if config is None:
            config = TrainingConfig()

        # Apply training window
        df_train = self._apply_window(df, config.window_years)

        # Compute temporal weights
        weights = self._compute_weights(df_train, config.xi)

        # Get model class
        model_class = self.MODEL_CLASSES[config.model_type]

        # Instantiate model (penaltyblog models take data in constructor)
        model = model_class(
            goals_home=df_train["home_goals"].values,
            goals_away=df_train["away_goals"].values,
            teams_home=df_train["home_team"].values,
            teams_away=df_train["away_team"].values,
            weights=weights,
        )

        # Fit the model (estimates parameters)
        model.fit()

        # Extract team list
        teams = self._extract_teams(model)

        result = TrainingResult(
            model_type=config.model_type,
            xi=config.xi,
            window_years=config.window_years,
            train_samples=len(df_train),
            teams=teams,
            trained_at=datetime.now().isoformat(),
        )

        return model, result

    def optimize(
        self,
        df: pd.DataFrame,
        config: Optional[OptimizationConfig] = None,
        model_type: Literal["poisson", "dixon_coles"] = "dixon_coles",
        verbose: bool = True,
    ) -> tuple[object, TrainingResult, dict]:
        """
        Optimize hyperparameters and train the best model.

        Uses Optuna to find optimal xi (decay rate) and window_years
        via rolling evaluation on the test season.

        Args:
            df: Prepared match DataFrame.
            config: Optimization configuration.
            model_type: Type of model to optimize.
            verbose: Whether to show optimization progress.

        Returns:
            Tuple of (best model, training result, optimization stats).
        """
        if config is None:
            config = OptimizationConfig()

        # Determine test season
        test_season = config.test_season
        if test_season is None:
            test_season = df["season"].iloc[-1]

        # Split data
        df_available = df[df["season"] != test_season].copy()
        df_test = df[df["season"] == test_season].copy()

        if len(df_test) == 0:
            raise ValueError(f"No test data found for season {test_season}")

        if verbose:
            print(f"Optimizing {model_type} model")
            print(f"Training seasons: {df_available['season'].unique().tolist()}")
            print(f"Test season: {test_season} ({len(df_test)} matches)")

        # Create Optuna study
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction="minimize", sampler=sampler)

        def objective(trial: optuna.Trial) -> float:
            # Sample hyperparameters
            xi = trial.suggest_float(
                "xi",
                config.xi_range[0],
                config.xi_range[1],
                log=True,
            )
            window_years = trial.suggest_int(
                "window_years",
                config.window_years_range[0],
                config.window_years_range[1],
            )

            # Evaluate on test season with rolling window
            scores = self._evaluate_rolling(
                df_available,
                df_test,
                model_type,
                xi,
                window_years,
                metric=config.metric,
            )

            return np.mean(scores)

        # Run optimization
        optuna.logging.set_verbosity(
            optuna.logging.INFO if verbose else optuna.logging.WARNING
        )
        study.optimize(objective, n_trials=config.n_trials, show_progress_bar=verbose)

        # Train final model with best params
        best_params = study.best_params
        if verbose:
            print(f"\nBest parameters: {best_params}")
            print(f"Best score ({config.metric}): {study.best_value:.4f}")

        best_config = TrainingConfig(
            model_type=model_type,
            xi=best_params["xi"],
            window_years=best_params["window_years"],
        )

        model, result = self.train(df, best_config)

        opt_stats = {
            "n_trials": config.n_trials,
            "best_params": best_params,
            "best_score": study.best_value,
            "metric": config.metric,
            "test_season": test_season,
        }

        return model, result, opt_stats

    def save(
        self,
        model: object,
        result: TrainingResult,
        name: str = "model",
    ) -> str:
        """
        Save trained model and metadata.

        Args:
            model: Trained model object.
            result: Training result metadata.
            name: Base name for the model files.

        Returns:
            Path to saved model file.
        """
        # Find next version number
        existing = list(self.output_dir.glob(f"{name}_v*.joblib"))
        if existing:
            versions = [int(p.stem.split("_v")[-1]) for p in existing]
            next_version = max(versions) + 1
        else:
            next_version = 1

        # Save model
        model_path = self.output_dir / f"{name}_v{next_version:03d}.joblib"
        joblib.dump(model, model_path)

        # Save metadata
        result.artifact_path = str(model_path)
        meta_path = self.output_dir / f"{name}_v{next_version:03d}.meta.json"
        with open(meta_path, "w") as f:
            json.dump(asdict(result), f, indent=2)

        return str(model_path)

    def load(self, filepath: str | Path) -> tuple[object, dict]:
        """
        Load a saved model and its metadata.

        Args:
            filepath: Path to the .joblib model file.

        Returns:
            Tuple of (model, metadata dict).
        """
        filepath = Path(filepath)
        model = joblib.load(filepath)

        # Load metadata
        meta_path = filepath.with_suffix(".meta.json")
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)
        else:
            metadata = {}

        return model, metadata

    def _apply_window(self, df: pd.DataFrame, window_years: int) -> pd.DataFrame:
        """Apply training window to filter recent data."""
        max_date = df["date"].max()
        cutoff = max_date - pd.DateOffset(years=window_years)
        return df[df["date"] >= cutoff].copy()

    def _compute_weights(self, df: pd.DataFrame, xi: float) -> np.ndarray:
        """Compute temporal decay weights."""
        max_date = df["date"].max()
        days_diff = (max_date - df["date"]).dt.days.values
        return np.exp(-xi * days_diff)

    def _extract_teams(self, model: object) -> list[str]:
        """Extract team list from fitted model."""
        # Try different attributes based on model type
        if hasattr(model, "teams_"):
            return list(model.teams_)
        if hasattr(model, "teams"):
            return list(model.teams)

        # Fallback: extract from parameter names
        params = model.get_params() if hasattr(model, "get_params") else {}
        teams = set()
        for key in params:
            if key.startswith("attack_") or key.startswith("defence_"):
                team = key.split("_", 1)[1]
                teams.add(team)
        return sorted(teams)

    def _evaluate_rolling(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        model_type: str,
        xi: float,
        window_years: int,
        metric: str,
    ) -> list[float]:
        """
        Evaluate model with rolling window on test data.

        For each match in df_test:
        1. Train on all available data before that match
        2. Predict the match
        3. Score the prediction

        Returns list of scores (one per test match).
        """
        from .model_evaluation import compute_rps, compute_brier

        metric_fn = compute_rps if metric == "rps" else compute_brier
        scores = []

        # Sort test matches chronologically
        df_test = df_test.sort_values("date")

        model_class = self.MODEL_CLASSES[model_type]

        for idx, row in df_test.iterrows():
            match_date = row["date"]

            # Get training data up to this point
            df_up_to = pd.concat([
                df_train,
                df_test[df_test["date"] < match_date]
            ])

            # Apply window
            cutoff = match_date - pd.DateOffset(years=window_years)
            df_window = df_up_to[df_up_to["date"] >= cutoff]

            if len(df_window) < 20:
                continue

            # Compute weights
            days_diff = (match_date - df_window["date"]).dt.days.values
            weights = np.exp(-xi * days_diff)

            # Check if teams are in training data
            if row["home_team"] not in df_window["home_team"].values:
                continue
            if row["away_team"] not in df_window["away_team"].values:
                continue

            try:
                # Train model (penaltyblog models take data in constructor)
                model = model_class(
                    goals_home=df_window["home_goals"].values,
                    goals_away=df_window["away_goals"].values,
                    teams_home=df_window["home_team"].values,
                    teams_away=df_window["away_team"].values,
                    weights=weights,
                )
                model.fit()

                # Predict
                probs = model.predict(row["home_team"], row["away_team"])
                prob_home = probs.home_win
                prob_draw = probs.draw
                prob_away = probs.away_win

                # True outcome
                actual_result = row["result"]  # 0=H, 1=D, 2=A

                # Score
                score = metric_fn(
                    [prob_home, prob_draw, prob_away],
                    actual_result,
                )
                scores.append(score)

            except Exception:
                # Skip matches that fail (e.g., unseen teams)
                continue

        return scores
