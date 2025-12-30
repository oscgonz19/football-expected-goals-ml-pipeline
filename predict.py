"""
Prediction Module

Handles match predictions and market extraction from trained models.
Provides a clean interface for generating predictions for new matches.

Markets supported:
- 1X2: Home win / Draw / Away win probabilities
- Expected Goals: Mean goals for each team
- BTTS: Both Teams To Score (yes/no)
- Totals: Over/Under for various goal lines
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

import numpy as np


@dataclass
class MarketProbabilities:
    """Container for match outcome probabilities."""

    home_win: float
    draw: float
    away_win: float


@dataclass
class ExpectedGoals:
    """Expected goals for each team."""

    home: float
    away: float
    total: float


@dataclass
class TotalsLine:
    """Over/Under probabilities for a goal line."""

    line: float
    over: float
    under: float
    push: Optional[float] = None  # For integer lines


@dataclass
class MatchPrediction:
    """Complete prediction for a match."""

    home_team: str
    away_team: str
    match_date: date
    generated_at: datetime

    # Core markets
    probabilities: MarketProbabilities
    expected_goals: ExpectedGoals

    # Optional markets
    btts_yes: Optional[float] = None
    btts_no: Optional[float] = None
    totals: list[TotalsLine] = field(default_factory=list)

    # Metadata
    model_type: Optional[str] = None
    confidence_note: Optional[str] = None


class MatchPredictor:
    """
    Generate match predictions from trained models.

    This class provides:
    - Match outcome predictions (1X2)
    - Expected goals computation
    - Derived market probabilities (BTTS, Totals)
    - Probability grid extraction

    Example:
        >>> predictor = MatchPredictor(model, model_type="dixon_coles")
        >>> prediction = predictor.predict("arsenal", "chelsea", date.today())
        >>> print(f"Home win: {prediction.probabilities.home_win:.1%}")
        >>> print(f"Expected goals: {prediction.expected_goals.total:.2f}")
    """

    DEFAULT_TOTAL_LINES = [0.5, 1.5, 2.5, 3.5, 4.5]

    def __init__(
        self,
        model: object,
        model_type: str = "dixon_coles",
        max_goals: int = 10,
    ):
        """
        Initialize the predictor.

        Args:
            model: Trained goal model with predict() method.
            model_type: Type of model ("poisson" or "dixon_coles").
            max_goals: Maximum goals to consider in probability grid.
        """
        self.model = model
        self.model_type = model_type
        self.max_goals = max_goals
        self._teams = self._extract_teams()

    def predict(
        self,
        home_team: str,
        away_team: str,
        match_date: Optional[date] = None,
        total_lines: Optional[list[float]] = None,
        include_btts: bool = True,
    ) -> MatchPrediction:
        """
        Generate prediction for a match.

        Args:
            home_team: Home team name (normalized).
            away_team: Away team name (normalized).
            match_date: Date of the match (default: today).
            total_lines: Goal lines for totals market (default: standard lines).
            include_btts: Whether to include BTTS market.

        Returns:
            MatchPrediction with all computed markets.

        Raises:
            ValueError: If teams are not in the model.
        """
        # Normalize team names
        home_team = home_team.lower().strip()
        away_team = away_team.lower().strip()

        # Validate teams
        self._validate_teams(home_team, away_team)

        if match_date is None:
            match_date = date.today()

        if total_lines is None:
            total_lines = self.DEFAULT_TOTAL_LINES

        # Get base prediction from model
        result = self.model.predict(home_team, away_team)

        # Extract probabilities
        probs = MarketProbabilities(
            home_win=result.home_win,
            draw=result.draw,
            away_win=result.away_win,
        )

        # Extract expected goals directly from result
        xg = ExpectedGoals(
            home=result.home_goal_expectation,
            away=result.away_goal_expectation,
            total=result.home_goal_expectation + result.away_goal_expectation,
        )

        # Store grid for later use
        self._last_grid = result.grid

        # Build prediction object
        prediction = MatchPrediction(
            home_team=home_team,
            away_team=away_team,
            match_date=match_date,
            generated_at=datetime.now(),
            probabilities=probs,
            expected_goals=xg,
            model_type=self.model_type,
        )

        # Add BTTS market (use grid from result)
        if include_btts:
            # BTTS = sum of grid[i,j] where i>0 and j>0
            btts = float(self._last_grid[1:, 1:].sum())
            prediction.btts_yes = btts
            prediction.btts_no = 1 - btts

        # Add totals markets (use grid from result)
        for line in total_lines:
            totals_line = self._compute_totals_from_grid(self._last_grid, line)
            prediction.totals.append(totals_line)

        # Add confidence note for edge cases
        prediction.confidence_note = self._get_confidence_note(home_team, away_team)

        return prediction

    def get_probability_grid(
        self,
        home_team: str,
        away_team: str,
    ) -> np.ndarray:
        """
        Get the full probability grid for exact scores.

        Returns a 2D array where grid[i, j] = P(home=i, away=j).

        Args:
            home_team: Home team name.
            away_team: Away team name.

        Returns:
            numpy array of shape (max_goals+1, max_goals+1).
        """
        home_team = home_team.lower().strip()
        away_team = away_team.lower().strip()

        # Get probabilities from model
        result = self.model.predict(home_team, away_team)

        # Extract probability matrix
        # penaltyblog models provide this via score_matrix or similar
        if hasattr(result, "score_matrix"):
            return result.score_matrix[: self.max_goals + 1, : self.max_goals + 1]

        # Fallback: reconstruct from marginals if available
        return self._reconstruct_grid(home_team, away_team)

    def get_teams(self) -> list[str]:
        """
        Get list of teams known to the model.

        Returns:
            Sorted list of team names.
        """
        return sorted(self._teams)

    def _validate_teams(self, home_team: str, away_team: str) -> None:
        """Validate that teams exist in the model."""
        if home_team not in self._teams:
            raise ValueError(
                f"Home team '{home_team}' not found in model. "
                f"Available teams: {sorted(self._teams)[:10]}..."
            )
        if away_team not in self._teams:
            raise ValueError(
                f"Away team '{away_team}' not found in model. "
                f"Available teams: {sorted(self._teams)[:10]}..."
            )
        if home_team == away_team:
            raise ValueError("Home and away teams cannot be the same")

    def _extract_teams(self) -> set[str]:
        """Extract team names from the fitted model."""
        if hasattr(self.model, "teams_"):
            return set(self.model.teams_)
        if hasattr(self.model, "teams"):
            return set(self.model.teams)

        # Fallback: extract from parameters
        params = self.model.get_params() if hasattr(self.model, "get_params") else {}
        teams = set()
        for key in params:
            if key.startswith("attack_") or key.startswith("defence_"):
                team = key.split("_", 1)[1]
                teams.add(team)
        return teams

    def _compute_expected_goals(
        self,
        home_team: str,
        away_team: str,
    ) -> ExpectedGoals:
        """Compute expected goals for each team."""
        # penaltyblog models typically provide lambda parameters
        if hasattr(self.model, "get_params"):
            params = self.model.get_params()

            # Get team attack/defense strengths
            home_attack = params.get(f"attack_{home_team}", 0)
            home_defence = params.get(f"defence_{home_team}", 0)
            away_attack = params.get(f"attack_{away_team}", 0)
            away_defence = params.get(f"defence_{away_team}", 0)

            # Home advantage if available
            home_adv = params.get("home_advantage", 0)

            # Expected goals: exp(attack_home - defence_away + home_adv)
            home_xg = np.exp(home_attack - away_defence + home_adv)
            away_xg = np.exp(away_attack - home_defence)

            return ExpectedGoals(
                home=float(home_xg),
                away=float(away_xg),
                total=float(home_xg + away_xg),
            )

        # Fallback: compute from probability grid
        grid = self.get_probability_grid(home_team, away_team)
        goals = np.arange(grid.shape[0])

        home_xg = np.sum(grid.sum(axis=1) * goals)
        away_xg = np.sum(grid.sum(axis=0) * goals)

        return ExpectedGoals(
            home=float(home_xg),
            away=float(away_xg),
            total=float(home_xg + away_xg),
        )

    def _compute_btts(self, home_team: str, away_team: str) -> float:
        """Compute probability that both teams score."""
        grid = self.get_probability_grid(home_team, away_team)

        # BTTS = 1 - P(home=0) - P(away=0) + P(home=0, away=0)
        # Or equivalently: sum of grid[i,j] where i>0 and j>0
        btts_yes = grid[1:, 1:].sum()

        return float(btts_yes)

    def _compute_totals(
        self,
        home_team: str,
        away_team: str,
        line: float,
    ) -> TotalsLine:
        """Compute over/under probabilities for a goal line."""
        grid = self.get_probability_grid(home_team, away_team)
        n = grid.shape[0]

        # Sum probabilities by total goals
        total_probs = np.zeros(2 * n - 1)
        for i in range(n):
            for j in range(n):
                total_probs[i + j] += grid[i, j]

        # Compute over/under
        under_prob = 0.0
        over_prob = 0.0
        push_prob = None

        for total in range(len(total_probs)):
            if total < line:
                under_prob += total_probs[total]
            elif total > line:
                over_prob += total_probs[total]
            else:  # Exactly on the line (integer line)
                push_prob = float(total_probs[total])

        return TotalsLine(
            line=line,
            over=float(over_prob),
            under=float(under_prob),
            push=push_prob,
        )

    def _compute_totals_from_grid(
        self,
        grid: np.ndarray,
        line: float,
    ) -> TotalsLine:
        """Compute over/under probabilities from a probability grid."""
        n = grid.shape[0]

        # Sum probabilities by total goals
        total_probs = np.zeros(2 * n - 1)
        for i in range(n):
            for j in range(n):
                total_probs[i + j] += grid[i, j]

        # Compute over/under
        under_prob = 0.0
        over_prob = 0.0
        push_prob = None

        for total in range(len(total_probs)):
            if total < line:
                under_prob += total_probs[total]
            elif total > line:
                over_prob += total_probs[total]
            else:  # Exactly on the line (integer line)
                push_prob = float(total_probs[total])

        return TotalsLine(
            line=line,
            over=float(over_prob),
            under=float(under_prob),
            push=push_prob,
        )

    def _reconstruct_grid(
        self,
        home_team: str,
        away_team: str,
    ) -> np.ndarray:
        """
        Reconstruct probability grid from model parameters.

        For Poisson models: P(h,a) = Poisson(lambda_h) * Poisson(lambda_a)
        For Dixon-Coles: includes low-score correlation adjustment.
        """
        from scipy.stats import poisson

        # Get expected goals
        xg = self._compute_expected_goals(home_team, away_team)

        # Create independent Poisson grid
        goals = np.arange(self.max_goals + 1)
        home_probs = poisson.pmf(goals, xg.home)
        away_probs = poisson.pmf(goals, xg.away)

        grid = np.outer(home_probs, away_probs)

        # Apply Dixon-Coles adjustment if applicable
        if self.model_type == "dixon_coles" and hasattr(self.model, "get_params"):
            params = self.model.get_params()
            rho = params.get("rho", 0)

            if rho != 0:
                # Dixon-Coles correlation adjustment for low scores
                grid = self._apply_dc_adjustment(grid, xg.home, xg.away, rho)

        # Normalize to ensure probabilities sum to 1
        grid = grid / grid.sum()

        return grid

    def _apply_dc_adjustment(
        self,
        grid: np.ndarray,
        lambda_h: float,
        lambda_a: float,
        rho: float,
    ) -> np.ndarray:
        """Apply Dixon-Coles low-score correlation adjustment."""
        grid = grid.copy()

        # Adjustment factors (from Dixon-Coles 1997)
        # tau(x,y,lambda,mu,rho) modifies P(x,y) for x,y in {0,1}
        grid[0, 0] *= 1 - lambda_h * lambda_a * rho
        grid[0, 1] *= 1 + lambda_h * rho
        grid[1, 0] *= 1 + lambda_a * rho
        grid[1, 1] *= 1 - rho

        # Ensure non-negative
        grid = np.maximum(grid, 0)

        return grid

    def _get_confidence_note(self, home_team: str, away_team: str) -> Optional[str]:
        """Generate confidence note for edge cases."""
        # This could check for:
        # - Newly promoted teams with limited data
        # - Teams with unusual recent form
        # - Missing contextual factors

        # For the demo, we just return None
        return None


def format_prediction(prediction: MatchPrediction) -> str:
    """
    Format a prediction for display.

    Args:
        prediction: MatchPrediction object.

    Returns:
        Formatted string representation.
    """
    lines = [
        f"\n{'=' * 50}",
        f"Match: {prediction.home_team.title()} vs {prediction.away_team.title()}",
        f"Date: {prediction.match_date}",
        f"Model: {prediction.model_type}",
        f"{'=' * 50}",
        "",
        "1X2 Probabilities:",
        f"  Home Win: {prediction.probabilities.home_win:.1%}",
        f"  Draw:     {prediction.probabilities.draw:.1%}",
        f"  Away Win: {prediction.probabilities.away_win:.1%}",
        "",
        "Expected Goals:",
        f"  Home: {prediction.expected_goals.home:.2f}",
        f"  Away: {prediction.expected_goals.away:.2f}",
        f"  Total: {prediction.expected_goals.total:.2f}",
    ]

    if prediction.btts_yes is not None:
        lines.extend([
            "",
            "BTTS (Both Teams To Score):",
            f"  Yes: {prediction.btts_yes:.1%}",
            f"  No:  {prediction.btts_no:.1%}",
        ])

    if prediction.totals:
        lines.extend(["", "Totals:"])
        for total in prediction.totals:
            push_str = f" (Push: {total.push:.1%})" if total.push else ""
            lines.append(
                f"  {total.line}: Over {total.over:.1%} / Under {total.under:.1%}{push_str}"
            )

    lines.append(f"\nGenerated at: {prediction.generated_at.strftime('%Y-%m-%d %H:%M:%S')}")

    return "\n".join(lines)
