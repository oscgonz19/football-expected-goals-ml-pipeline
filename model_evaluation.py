"""
Model Evaluation Module

Provides proper scoring rules for evaluating probabilistic predictions.
Includes RPS (Ranked Probability Score) and Brier Score implementations.

Proper scoring rules incentivize honest probability reporting - the expected
score is maximized when the forecaster reports their true beliefs.

Lower scores are better for all metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class EvaluationResult:
    """Container for evaluation results."""

    metric: str
    score: float
    n_samples: int
    score_std: Optional[float] = None
    scores_by_outcome: Optional[dict[str, float]] = None


def compute_rps(probs: list[float], actual_outcome: int) -> float:
    """
    Compute Ranked Probability Score for a single prediction.

    RPS measures the accuracy of probabilistic forecasts for ordered
    categorical outcomes. It penalizes predictions that place probability
    mass far from the actual outcome more heavily than those that are
    close but wrong.

    Formula:
        RPS = (1/K) * Σ (cumsum(probs) - cumsum(actual))^2

    where K is the number of categories minus 1.

    Args:
        probs: Predicted probabilities [P(home), P(draw), P(away)].
        actual_outcome: True outcome (0=home, 1=draw, 2=away).

    Returns:
        RPS score (lower is better, range [0, 1]).

    Example:
        >>> # Confident correct prediction
        >>> compute_rps([0.8, 0.15, 0.05], 0)
        0.0125
        >>> # Wrong prediction
        >>> compute_rps([0.8, 0.15, 0.05], 2)
        0.9025
    """
    probs = np.array(probs)
    n_classes = len(probs)

    # Create one-hot encoding of actual outcome
    actual = np.zeros(n_classes)
    actual[actual_outcome] = 1.0

    # Cumulative sums
    cum_probs = np.cumsum(probs)
    cum_actual = np.cumsum(actual)

    # RPS formula
    rps = np.sum((cum_probs - cum_actual) ** 2) / (n_classes - 1)

    return float(rps)


def compute_brier(probs: list[float], actual_outcome: int) -> float:
    """
    Compute Brier Score for a single prediction.

    The Brier score measures the mean squared error between predicted
    probabilities and the actual outcome. It's a strictly proper scoring
    rule commonly used in probability forecasting.

    Formula:
        BS = Σ (prob_i - actual_i)^2

    Args:
        probs: Predicted probabilities [P(home), P(draw), P(away)].
        actual_outcome: True outcome (0=home, 1=draw, 2=away).

    Returns:
        Brier score (lower is better, range [0, 2] for 3 outcomes).

    Example:
        >>> # Perfect prediction
        >>> compute_brier([1.0, 0.0, 0.0], 0)
        0.0
        >>> # Uniform prediction
        >>> compute_brier([0.333, 0.334, 0.333], 0)
        0.667
    """
    probs = np.array(probs)
    n_classes = len(probs)

    # Create one-hot encoding of actual outcome
    actual = np.zeros(n_classes)
    actual[actual_outcome] = 1.0

    # Brier score
    brier = np.sum((probs - actual) ** 2)

    return float(brier)


def compute_log_loss(probs: list[float], actual_outcome: int, eps: float = 1e-15) -> float:
    """
    Compute Log Loss (Ignorance Score) for a single prediction.

    Log loss measures the negative log-likelihood of the true outcome
    under the predicted distribution. It heavily penalizes confident
    wrong predictions.

    Formula:
        LL = -log(prob_actual)

    Args:
        probs: Predicted probabilities [P(home), P(draw), P(away)].
        actual_outcome: True outcome (0=home, 1=draw, 2=away).
        eps: Small value to avoid log(0).

    Returns:
        Log loss (lower is better, range [0, inf)).

    Example:
        >>> # Confident correct prediction
        >>> compute_log_loss([0.9, 0.05, 0.05], 0)
        0.105...
        >>> # Confident wrong prediction
        >>> compute_log_loss([0.9, 0.05, 0.05], 2)
        2.996...
    """
    probs = np.clip(probs, eps, 1 - eps)
    return float(-np.log(probs[actual_outcome]))


class ModelEvaluator:
    """
    Evaluate goal prediction models on test data.

    Provides methods to:
    - Compute various proper scoring rules
    - Generate calibration plots
    - Analyze prediction quality by outcome type
    - Compare multiple models

    Example:
        >>> evaluator = ModelEvaluator()
        >>> results = evaluator.evaluate(model, df_test)
        >>> print(f"RPS: {results['rps'].score:.4f}")
        >>> print(f"Brier: {results['brier'].score:.4f}")
    """

    METRICS = {
        "rps": compute_rps,
        "brier": compute_brier,
        "log_loss": compute_log_loss,
    }

    def __init__(self):
        """Initialize the evaluator."""
        pass

    def evaluate(
        self,
        model: object,
        df_test: pd.DataFrame,
        metrics: Optional[list[str]] = None,
    ) -> dict[str, EvaluationResult]:
        """
        Evaluate model on test dataset.

        For each match in df_test, generates a prediction and scores it
        against the actual outcome.

        Args:
            model: Trained model with predict(home, away) method.
            df_test: Test DataFrame with match data.
            metrics: List of metrics to compute (default: all).

        Returns:
            Dictionary mapping metric names to EvaluationResult.
        """
        if metrics is None:
            metrics = list(self.METRICS.keys())

        # Collect predictions
        predictions = []
        actuals = []
        skipped = 0

        for _, row in df_test.iterrows():
            try:
                probs = model.predict(row["home_team"], row["away_team"])
                predictions.append([probs.home_win, probs.draw, probs.away_win])
                actuals.append(row["result"])
            except Exception:
                skipped += 1
                continue

        if not predictions:
            raise ValueError("No valid predictions generated")

        if skipped > 0:
            print(f"Warning: Skipped {skipped} matches due to unknown teams")

        # Compute metrics
        results = {}
        for metric_name in metrics:
            metric_fn = self.METRICS[metric_name]
            scores = [
                metric_fn(pred, actual)
                for pred, actual in zip(predictions, actuals)
            ]

            # Breakdown by outcome
            scores_array = np.array(scores)
            actuals_array = np.array(actuals)
            scores_by_outcome = {
                "home_win": float(np.mean(scores_array[actuals_array == 0])) if (actuals_array == 0).any() else None,
                "draw": float(np.mean(scores_array[actuals_array == 1])) if (actuals_array == 1).any() else None,
                "away_win": float(np.mean(scores_array[actuals_array == 2])) if (actuals_array == 2).any() else None,
            }

            results[metric_name] = EvaluationResult(
                metric=metric_name,
                score=float(np.mean(scores)),
                n_samples=len(scores),
                score_std=float(np.std(scores)),
                scores_by_outcome=scores_by_outcome,
            )

        return results

    def evaluate_calibration(
        self,
        model: object,
        df_test: pd.DataFrame,
        n_bins: int = 10,
    ) -> pd.DataFrame:
        """
        Compute calibration statistics.

        Bins predictions by confidence and compares predicted vs actual
        frequencies. Well-calibrated models should have predicted and
        actual frequencies that match.

        Args:
            model: Trained model.
            df_test: Test DataFrame.
            n_bins: Number of probability bins.

        Returns:
            DataFrame with calibration statistics per bin.
        """
        predictions = []
        actuals = []

        for _, row in df_test.iterrows():
            try:
                probs = model.predict(row["home_team"], row["away_team"])
                predictions.append([probs.home_win, probs.draw, probs.away_win])
                actuals.append(row["result"])
            except Exception:
                continue

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        # Focus on home win probability for calibration
        home_probs = predictions[:, 0]
        home_actual = (actuals == 0).astype(float)

        # Create bins
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(home_probs, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        calibration_data = []
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() == 0:
                continue

            bin_probs = home_probs[mask]
            bin_actuals = home_actual[mask]

            calibration_data.append({
                "bin_center": (bins[i] + bins[i + 1]) / 2,
                "predicted_prob": bin_probs.mean(),
                "actual_freq": bin_actuals.mean(),
                "count": mask.sum(),
            })

        return pd.DataFrame(calibration_data)

    def compare_models(
        self,
        models: dict[str, object],
        df_test: pd.DataFrame,
        metric: str = "rps",
    ) -> pd.DataFrame:
        """
        Compare multiple models on the same test set.

        Args:
            models: Dictionary mapping model names to model objects.
            df_test: Test DataFrame.
            metric: Metric to use for comparison.

        Returns:
            DataFrame with comparison results.
        """
        results = []

        for name, model in models.items():
            try:
                eval_result = self.evaluate(model, df_test, metrics=[metric])
                results.append({
                    "model": name,
                    "score": eval_result[metric].score,
                    "score_std": eval_result[metric].score_std,
                    "n_samples": eval_result[metric].n_samples,
                })
            except Exception as e:
                results.append({
                    "model": name,
                    "score": None,
                    "score_std": None,
                    "n_samples": 0,
                    "error": str(e),
                })

        df_comparison = pd.DataFrame(results)
        return df_comparison.sort_values("score")


def baseline_uniform() -> list[float]:
    """
    Generate baseline uniform prediction.

    Returns:
        [1/3, 1/3, 1/3] probabilities.
    """
    return [1 / 3, 1 / 3, 1 / 3]


def baseline_historical(df: pd.DataFrame) -> list[float]:
    """
    Generate baseline prediction from historical frequencies.

    Args:
        df: DataFrame with historical match results.

    Returns:
        [P(home), P(draw), P(away)] from historical data.
    """
    total = len(df)
    if total == 0:
        return baseline_uniform()

    home_wins = (df["result"] == 0).sum() / total
    draws = (df["result"] == 1).sum() / total
    away_wins = (df["result"] == 2).sum() / total

    return [home_wins, draws, away_wins]
